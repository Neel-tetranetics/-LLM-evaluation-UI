"""Web server for the LLM Evaluation UI.

Start it from the project folder:
    .venv/bin/python app.py
Then open http://<this-machine-ip>:7860 from any device on the same network.
"""

import csv
import io
import json
import re
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

import graders
import runner

STATIC_DIR = Path(__file__).parent / "static"

# Counting lines of a big JSONL is slow, so remember counts until the file changes
dataset_count_cache: dict[str, tuple[float, int, str]] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    runner.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    runner.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runner.mark_interrupted_runs()
    runner.start_worker()
    yield
    runner.stop_all_servers()  # Ctrl-C during a run must not leave vLLM holding the GPU


app = FastAPI(title="LLM Evaluation", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def bad_request(error: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(error))


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


# ---------- config: judges and presets ----------

@app.get("/api/config")
def get_config():
    return runner.load_config()


@app.post("/api/judges")
def add_judge(judge: dict = Body(...)):
    config = runner.load_config()
    name = str(judge.get("name") or "").strip()
    command = " ".join(str(judge.get("command") or "").split())
    try:
        if not name:
            raise ValueError("judge needs a name")
        if any(existing["name"] == name for existing in config["judges"]):
            raise ValueError(f"a judge named '{name}' already exists")
        runner.parse_command(command)
    except ValueError as error:
        raise bad_request(error)
    config["judges"].append({"name": name, "command": command})
    runner.save_config(config)
    return {"ok": True}


# Run settings editable in the UI: name -> (type, minimum, maximum). None = no limit.
RUN_SETTING_RULES = {
    "default_system_prompt": (str, None, None),
    "quick_size": (int, 1, None),
    "seed": (int, None, None),
    "temperature": (float, 0, 2),
    "top_p": (float, 0, 1),
    "top_k": (int, -1, None),
    "presence_penalty": (float, -2, 2),
    "max_tokens": (int, 16, None),
    "parallel_requests": (int, 1, 256),
    "request_timeout_sec": (int, 10, None),
    "judge_timeout_sec": (int, 10, None),
    "request_retries": (int, 0, 5),
    "judge_max_tokens": (int, 64, 8192),
    "code_test_timeout_sec": (int, 1, 600),
    "server_start_timeout_sec": (int, 60, None),
    "agent_max_steps": (int, 1, 500),
    "agent_step_max_tokens": (int, 256, None),
    "agent_command_timeout_sec": (int, 5, None),
    "agent_output_chars": (int, 200, None),
    "agent_task_timeout_sec": (int, 60, None),
    "agent_verifier_timeout_sec": (int, 60, None),
}


def check_run_setting(name: str, value):
    number_type, minimum, maximum = RUN_SETTING_RULES[name]
    try:
        value = number_type(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a {number_type.__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


@app.put("/api/settings")
def update_settings(changes: dict = Body(...)):
    """Save run settings and the default judge. Only known settings are accepted."""
    config = runner.load_config()
    try:
        for name, value in (changes.get("run") or {}).items():
            if name not in RUN_SETTING_RULES:
                raise ValueError(f"unknown setting: {name}")
            config["run"][name] = check_run_setting(name, value)
        if changes.get("default_judge"):
            runner.find_judge(config, changes["default_judge"])
            config["default_judge"] = changes["default_judge"]
    except ValueError as error:
        raise bad_request(error)
    runner.save_config(config)
    return {"ok": True}


@app.delete("/api/judges/{name}")
def delete_judge(name: str):
    config = runner.load_config()
    if name == config["default_judge"]:
        raise HTTPException(400, "cannot delete the default judge")
    config["judges"] = [judge for judge in config["judges"] if judge["name"] != name]
    runner.save_config(config)
    return {"ok": True}


@app.post("/api/presets")
def add_preset(preset: dict = Body(...)):
    config = runner.load_config()
    name = str(preset.get("name") or "").strip()
    command = " ".join(str(preset.get("command") or "").split())
    try:
        if not name:
            raise ValueError("preset needs a name")
        runner.parse_command(command)
    except ValueError as error:
        raise bad_request(error)
    presets = [existing for existing in config.get("test_model_presets") or [] if existing["name"] != name]
    presets.append({"name": name, "command": command})
    config["test_model_presets"] = presets
    runner.save_config(config)
    return {"ok": True}


@app.delete("/api/presets/{name}")
def delete_preset(name: str):
    config = runner.load_config()
    config["test_model_presets"] = [preset for preset in config.get("test_model_presets") or [] if preset["name"] != name]
    runner.save_config(config)
    return {"ok": True}


# ---------- datasets ----------

def dataset_info(path: Path) -> dict:
    modified = path.stat().st_mtime
    cached = dataset_count_cache.get(path.name)
    if cached is None or cached[0] != modified:
        with open(path) as file:
            first_line = file.readline()
            count = 1 + sum(1 for _ in file) if first_line else 0
        category = json.loads(first_line).get("category", "custom") if first_line else "custom"
        dataset_count_cache[path.name] = (modified, count, category)
    _, count, category = dataset_count_cache[path.name]
    available = category not in ("coding", "agent") or graders.docker_command() is not None
    return {"name": path.stem, "category": category, "count": count, "available": available}


@app.get("/api/datasets")
def list_datasets():
    return [dataset_info(path) for path in sorted(runner.DATASETS_DIR.glob("*.jsonl"))]


def parse_upload_rows(filename: str, content: str) -> list[dict]:
    """Read uploaded JSONL or CSV. Raises ValueError("line N: ...") on the first bad row."""
    if filename.lower().endswith(".csv"):
        # Line 1 is the header, so data starts at line 2
        rows = list(enumerate(csv.DictReader(io.StringIO(content)), start=2))
    elif filename.lower().endswith(".jsonl"):
        rows = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append((line_number, json.loads(line)))
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: not valid JSON ({error.msg})")
    else:
        raise ValueError("file must end with .jsonl or .csv")

    items = []
    for line_number, row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: must be a JSON object")
        for field in ("question", "ground_truth"):
            if not str(row.get(field) or "").strip():
                raise ValueError(f"line {line_number}: missing {field}")
        items.append(row)
    if not items:
        raise ValueError("file has no rows")
    return items


@app.post("/api/datasets/upload")
def upload_dataset(upload: dict = Body(...)):
    try:
        slug = re.sub(r"[^a-z0-9_]+", "_", str(upload.get("name") or "").lower()).strip("_")
        if not slug:
            raise ValueError("dataset needs a name")
        rows = parse_upload_rows(str(upload.get("filename") or ""), str(upload.get("content") or ""))
    except ValueError as error:
        raise bad_request(error)
    dataset_name = f"custom_{slug}"
    with open(runner.DATASETS_DIR / f"{dataset_name}.jsonl", "w") as file:
        for index, row in enumerate(rows):
            item = {
                "id": f"{dataset_name}-{index}",
                "dataset": dataset_name,
                "category": "custom",
                "grader": "judge",
                "question": str(row["question"]),
                "ground_truth": str(row["ground_truth"]),
            }
            if str(row.get("context") or "").strip():
                item["context"] = str(row["context"])
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"name": dataset_name, "count": len(rows)}


# ---------- runs ----------

@app.post("/api/runs")
def create_run(form: dict = Body(...)):
    try:
        return {"run_id": runner.submit_run(form)}
    except ValueError as error:
        raise bad_request(error)


@app.get("/api/runs")
def list_runs():
    return runner.list_runs()


def existing_run(run_id: str) -> None:
    try:
        runner.run_dir(run_id)
    except ValueError as error:
        raise HTTPException(404, str(error))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    existing_run(run_id)
    return runner.get_run(run_id)


@app.get("/api/runs/{run_id}/results")
def get_results(run_id: str):
    existing_run(run_id)
    return runner.get_results(run_id)


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    existing_run(run_id)
    runner.cancel_run(run_id)
    return {"ok": True}


@app.get("/api/runs/{run_id}/log/{which}", response_class=PlainTextResponse)
def get_log(run_id: str, which: str):
    existing_run(run_id)
    if which not in ("test_model", "judge"):
        raise HTTPException(404, "log must be test_model or judge")
    return runner.last_log_lines(runner.run_dir(run_id) / f"{which}.log", 200) or "(no log yet)"


@app.get("/api/compare")
def compare_runs(ids: str):
    run_ids = [run_id for run_id in ids.split(",") if run_id]
    if len(run_ids) < 2:
        raise HTTPException(400, "pick at least two runs")
    for run_id in run_ids:
        existing_run(run_id)
    verdicts_by_run = {run_id: {result["id"]: result for result in runner.get_results(run_id)} for run_id in run_ids}
    shared_ids = set.intersection(*(set(results) for results in verdicts_by_run.values()))
    differences = []
    for question_id in sorted(shared_ids):
        verdicts = {run_id: verdicts_by_run[run_id][question_id]["verdict"] for run_id in run_ids}
        if len(set(verdicts.values())) > 1:
            first = verdicts_by_run[run_ids[0]][question_id]
            differences.append({"id": question_id, "dataset": first["dataset"],
                                "question": first["question"], "verdicts": verdicts})
    return {"runs": [runner.get_run(run_id) for run_id in run_ids], "differences": differences}


def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


if __name__ == "__main__":
    # uvicorn runs the startup steps (which mark unfinished runs "interrupted" and kill their vLLM)
    # BEFORE it finds out the port is taken. A second `python app.py` once killed a running eval this way.
    if not port_is_free(7860):
        sys.exit("LLM Evaluation is already running on port 7860. Not starting a second copy.")
    uvicorn.run(app, host="0.0.0.0", port=7860)
