"""Run evaluations: start vLLM, ask the questions, grade, judge, save results.

Only one model is on the GPU at a time:
    start test model -> answer all questions -> stop it -> start judge -> judge -> stop it.
Only one run executes at a time (one GPU). Extra runs wait in a queue.
Files for each run live in data/runs/<run_id>/:
    settings.json    what the user picked + status + error
    results.jsonl    one line per question
    summary.json     score per dataset
    test_model.log   vLLM output of the tested model
    judge.log        vLLM output of the judge
"""

import json
import os
import queue
import random
import re
import shlex
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml
from openai import OpenAI

import agent
import graders

PROJECT_DIR = Path(__file__).parent
CONFIG_PATH = PROJECT_DIR / "config.yml"
DATA_DIR = PROJECT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"
VENV_BIN = PROJECT_DIR / ".venv" / "bin"

FINISHED_STATUSES = {"done", "failed", "cancelled", "interrupted"}
SERVER_PID_FILE = "server.pid"  # pid of the vLLM started for a run, so a crashed app can clean up

run_queue: "queue.Queue[str]" = queue.Queue()
live_progress: dict[str, dict] = {}       # run_id -> {"done", "total", "stage"}
cancel_requested: set[str] = set()
results_lock = threading.Lock()
active_servers: set[subprocess.Popen] = set()  # stopped when the app shuts down


# ---------- config ----------

def load_config() -> dict:
    with open(CONFIG_PATH) as file:
        return yaml.safe_load(file)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True, width=1000)


def find_judge(config: dict, judge_name: str) -> dict:
    for judge in config["judges"]:
        if judge["name"] == judge_name:
            return judge
    raise ValueError(f"unknown judge: {judge_name}")


# ---------- vLLM process control ----------

# vLLM flags that load Python code or read arbitrary files. Anyone on the LAN can submit a
# command, so these are refused.
# ponytail: substring block list; switch to an allow list if vLLM adds more code-loading flags.
BLOCKED_FLAG_PARTS = ("trust-remote-code", "plugin", "middleware", "-cls", "logits-processor",
                      "chat-template", "allowed-local-media-path")


def option_value(tokens: list[str], option: str) -> str | None:
    """Value of --option VALUE or --option=VALUE. Last one wins, like vLLM's own argument parser."""
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(option + "="):
            return token.split("=", 1)[1]
    return None


def parse_command(command: str) -> tuple[list[str], int, str]:
    """Check a vLLM command and return (argv, port, served model name).

    Only vLLM server commands are allowed, because anyone on the LAN can submit one.
    """
    tokens = shlex.split(command)
    is_api_server = len(tokens) >= 3 and tokens[0] in ("python", "python3") and tokens[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    is_vllm_serve = len(tokens) >= 2 and tokens[:2] == ["vllm", "serve"]
    if not (is_api_server or is_vllm_serve):
        raise ValueError("command must start with 'python -m vllm.entrypoints.openai.api_server' or 'vllm serve'")
    for token in tokens:
        if token.startswith("--") and any(part in token for part in BLOCKED_FLAG_PARTS):
            raise ValueError(f"flag {token.split('=')[0]} is not allowed (it can run code on this server)")
    # Use this project's vLLM, not whatever is first on PATH
    tokens[0] = str(VENV_BIN / ("vllm" if is_vllm_serve else "python"))

    port_text = option_value(tokens, "--port")
    if port_text is None or not port_text.isdigit():
        raise ValueError("command must contain --port <number>")

    model_name = option_value(tokens, "--served-model-name") or option_value(tokens, "--model")
    if model_name is None and is_vllm_serve and len(tokens) > 2 and not tokens[2].startswith("-"):
        model_name = tokens[2]
    if model_name is None:
        raise ValueError("command must contain --model <path>")
    return tokens, int(port_text), model_name


def is_server_up(port: int) -> bool:
    try:
        return requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def start_server(command: str, log_path: Path) -> subprocess.Popen:
    argv, port, _ = parse_command(command)
    if is_server_up(port):
        raise ValueError(f"port {port} already in use")
    # Like an activated venv: vLLM's FlashInfer compiles kernels with `ninja`, which lives in .venv/bin
    environment = {**os.environ, "PATH": f"{VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"}
    with open(log_path, "a") as log_file:
        # Own process group, so stop_server can kill vLLM and all its worker processes together
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT, env=environment,
                                   start_new_session=True, cwd=PROJECT_DIR)
    active_servers.add(process)
    (log_path.parent / SERVER_PID_FILE).write_text(str(process.pid))
    return process


def last_log_lines(log_path: Path, count: int = 50) -> str:
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-count:])


def wait_until_healthy(process: subprocess.Popen, port: int, timeout_sec: int, log_path: Path,
                       run_id: str = "") -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if run_id in cancel_requested:
            raise RunCancelled()  # loading can take minutes; don't make Cancel wait for it
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}. Last log lines:\n{last_log_lines(log_path)}")
        if is_server_up(port):
            return
        time.sleep(3)
    raise RuntimeError(f"vLLM not ready after {timeout_sec}s. Last log lines:\n{last_log_lines(log_path)}")


def kill_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass  # already gone


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    active_servers.discard(process)
    if process.poll() is None:
        kill_process_group(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    # vLLM worker processes can outlive the main one and still hold GPU memory
    kill_process_group(process.pid, signal.SIGKILL)
    process.wait()


def stop_all_servers() -> None:
    """Called when the app shuts down, so no vLLM keeps holding the GPU."""
    for process in list(active_servers):
        stop_server(process)


def kill_leftover_server(folder: Path) -> None:
    """After an app crash, the vLLM it started may still run. Kill it if the pid is still vLLM."""
    pid_file = folder / SERVER_PID_FILE
    if not pid_file.exists():
        return
    pid = int(pid_file.read_text())
    try:
        command_line = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        command_line = b""
    if b"vllm" in command_line:  # the pid may have been reused by an unrelated process
        kill_process_group(pid, signal.SIGKILL)
    pid_file.unlink()


# ---------- datasets and question picking ----------

def dataset_names() -> set[str]:
    return {path.stem for path in DATASETS_DIR.glob("*.jsonl")}


def load_dataset(name: str) -> list[dict]:
    # Compare with real file names, so names like "../runs/x/results" can't reach other files
    if name not in dataset_names():
        raise ValueError(f"unknown dataset: {name}")
    path = DATASETS_DIR / f"{name}.jsonl"
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def pick_questions(rows: list[dict], size: str | int, seed: int) -> list[dict]:
    """Same seed + same size = same questions every time, so runs can be compared."""
    if size == "full":
        return rows
    count = load_config()["run"]["quick_size"] if size == "quick" else int(size)
    if count >= len(rows):
        return rows
    chosen_positions = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[position] for position in chosen_positions]


# ---------- run files ----------

def run_dir(run_id: str) -> Path:
    path = RUNS_DIR / run_id
    if not path.is_dir() or "/" in run_id or ".." in run_id:
        raise ValueError(f"unknown run: {run_id}")
    return path


def read_settings(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "settings.json").read_text())


def update_settings(run_id: str, **changes) -> None:
    settings = read_settings(run_id)
    settings.update(changes)
    # Write then rename, so the UI never reads a half-written file
    temporary_path = run_dir(run_id) / "settings.json.tmp"
    temporary_path.write_text(json.dumps(settings, indent=2))
    os.replace(temporary_path, run_dir(run_id) / "settings.json")


def set_stage(run_id: str, status: str, done: int = 0, total: int = 0) -> None:
    live_progress[run_id] = {"done": done, "total": total, "stage": status}
    update_settings(run_id, status=status, progress=live_progress[run_id])


def make_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-")[:40] or "run"


# ---------- public API used by app.py ----------

def form_number(form: dict, key: str, default, number_type):
    """Number from the form; empty or missing -> the config default."""
    value = form.get(key)
    if value is None or str(value).strip() == "":
        return number_type(default)
    try:
        return number_type(value)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {value!r}")


def submit_run(form: dict) -> str:
    """Check the form, save settings.json, and put the run in the queue."""
    config = load_config()
    name = str(form.get("name") or "").strip() or "run"
    command = str(form.get("command") or "").strip()
    _, _, model_name = parse_command(command)

    judge = find_judge(config, form.get("judge") or config["default_judge"])

    datasets = form.get("datasets") or []
    if not datasets:
        raise ValueError("pick at least one dataset")
    known_datasets = dataset_names()
    for dataset in datasets:
        if dataset not in known_datasets:
            raise ValueError(f"unknown dataset: {dataset}")

    size = form.get("size", "quick")
    if size not in ("quick", "full") and not (str(size).isdigit() and int(size) > 0):
        raise ValueError("size must be 'quick', 'full', or a positive number")

    run_config = config["run"]
    settings = {
        "name": name,
        "command": command,
        "model_name": model_name,
        "model_path": option_value(shlex.split(command), "--model") or model_name,
        "system_prompt": str(form.get("system_prompt") or ""),
        "datasets": datasets,
        "size": size,
        "seed": int(form.get("seed") or run_config["seed"]),
        "judge": judge["name"],
        "temperature": form_number(form, "temperature", run_config["temperature"], float),
        "top_p": form_number(form, "top_p", run_config.get("top_p", 1.0), float),
        "top_k": form_number(form, "top_k", run_config.get("top_k", -1), int),
        "presence_penalty": form_number(form, "presence_penalty", run_config.get("presence_penalty", 0.0), float),
        "max_tokens": form_number(form, "max_tokens", run_config["max_tokens"], int),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "queued",
        "progress": {"done": 0, "total": 0, "stage": "queued"},
        "error": "",
    }
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + make_slug(name)
    (RUNS_DIR / run_id).mkdir(parents=True)
    (RUNS_DIR / run_id / "settings.json").write_text(json.dumps(settings, indent=2))
    run_queue.put(run_id)
    return run_id


def get_run(run_id: str) -> dict:
    run = read_settings(run_id)
    run["id"] = run_id
    if run_id in live_progress and run["status"] not in FINISHED_STATUSES:
        run["progress"] = live_progress[run_id]
    summary_path = run_dir(run_id) / "summary.json"
    run["summary"] = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return run


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    run_ids = sorted((path.name for path in RUNS_DIR.iterdir() if (path / "settings.json").exists()), reverse=True)
    return [get_run(run_id) for run_id in run_ids]


def get_results(run_id: str) -> list[dict]:
    path = run_dir(run_id) / "results.jsonl"
    if not path.exists():
        return []
    with results_lock, open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def cancel_run(run_id: str) -> None:
    status = read_settings(run_id)["status"]
    if status in FINISHED_STATUSES:
        return
    cancel_requested.add(run_id)
    if status == "queued":
        # Not started yet: the worker will skip it
        update_settings(run_id, status="cancelled")


def mark_interrupted_runs() -> None:
    """After an app restart, runs that were in progress can't continue."""
    for run in list_runs():
        kill_leftover_server(RUNS_DIR / run["id"])
        if run["status"] not in FINISHED_STATUSES:
            update_settings(run["id"], status="interrupted", error="app restarted during the run")


def start_worker() -> None:
    threading.Thread(target=worker_loop, daemon=True).start()


def worker_loop() -> None:
    while True:
        run_id = run_queue.get()
        if read_settings(run_id)["status"] == "cancelled":
            continue
        execute_run(run_id)


# ---------- the run itself ----------

class RunCancelled(Exception):
    pass


def append_result(run_id: str, result: dict) -> None:
    with results_lock, open(run_dir(run_id) / "results.jsonl", "a") as file:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")


def rewrite_results(run_id: str, results: list[dict]) -> None:
    with results_lock, open(run_dir(run_id) / "results.jsonl", "w") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def build_messages(item: dict, system_prompt: str) -> list[dict]:
    user_text = item["question"]
    if item.get("context"):
        user_text += "\n\nContext:\n" + item["context"]
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": user_text})
    return messages


def read_tool_calls(message) -> list[dict]:
    calls = []
    for tool_call in message.tool_calls or []:
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}  # broken JSON from the model; the grader will mark it incorrect
        calls.append({"name": tool_call.function.name, "arguments": arguments})
    return calls


def ask_test_model(client: OpenAI, model_name: str, item: dict, settings: dict) -> dict:
    """Ask one question and grade it with code if possible."""
    started = time.time()
    request = {
        "model": model_name,
        "messages": build_messages(item, settings["system_prompt"]),
        "temperature": settings["temperature"],
        "top_p": settings.get("top_p", 1.0),
        "presence_penalty": settings.get("presence_penalty", 0.0),
        "max_tokens": settings["max_tokens"],
        # top_k is not part of the OpenAI API; vLLM reads it from extra_body
        "extra_body": {"top_k": settings.get("top_k", -1)},
    }
    if item.get("tools"):
        request["tools"] = item["tools"]
    result = {key: item.get(key) for key in ("id", "dataset", "category", "grader", "question", "ground_truth")}
    try:
        choice = client.chat.completions.create(**request).choices[0]
        message = choice.message
        extra_fields = message.model_extra or {}
        result["answer"] = message.content or ""
        # vLLM puts thinking text in "reasoning_content" (older) or "reasoning" (newer)
        result["reasoning"] = extra_fields.get("reasoning_content") or extra_fields.get("reasoning") or ""
        result["tool_calls"] = read_tool_calls(message)
        grade = graders.grade_without_judge(item, result["answer"], result["tool_calls"],
                                            load_config()["run"]["code_test_timeout_sec"])
        if grade["reason"] == "empty answer" and choice.finish_reason == "length":
            # Common with thinking models: all tokens went to thinking. Tells the user to raise max tokens.
            grade["reason"] = "empty answer: hit max_tokens before finishing (thinking too long?)"
    except Exception as error:  # one bad question must not stop the whole run
        result.update(answer="", reasoning="", tool_calls=[])
        grade = {"verdict": "error", "reason": f"request failed: {error}"}
    result.update(grade)
    result["seconds"] = round(time.time() - started, 2)
    return result


def ask_judge(client: OpenAI, model_name: str, item: dict, answer: str) -> dict:
    messages = graders.build_judge_messages(item, answer)
    for _attempt in range(2):  # bad JSON → one retry
        reply = client.chat.completions.create(
            # Temperature 0: the same answer should always get the same verdict.
            # Greedy looping is not a risk here: thinking is off and the reply is one short JSON.
            model=model_name, messages=messages, temperature=0,
            max_tokens=load_config()["run"].get("judge_max_tokens", 512),
            # Thinking makes the judge slow and is not needed for comparing two answers
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).choices[0].message.content or ""
        try:
            return graders.parse_judge_reply(reply)
        except ValueError as error:
            last_error = str(error)
    return {"verdict": "error", "reason": last_error}


def summarize(results: list[dict]) -> dict:
    summary = {}
    for result in results:
        for key in (result["dataset"], "overall"):
            counts = summary.setdefault(key, {"correct": 0, "incorrect": 0, "error": 0, "total": 0})
            verdict = result["verdict"] if result["verdict"] in ("correct", "error") else "incorrect"
            counts[verdict] += 1
            counts["total"] += 1
    for counts in summary.values():
        counts["score"] = round(counts["correct"] / counts["total"], 4) if counts["total"] else 0
    return summary


def check_cancel(run_id: str) -> None:
    if run_id in cancel_requested:
        raise RunCancelled()


AGENT_LIMIT_DEFAULTS = {
    "agent_max_steps": 250,             # commands per task (mini-swe-agent's SWE-bench default)
    "agent_step_max_tokens": 16384,     # per model reply; thinking models need room to think, then write the command
    "agent_command_timeout_sec": 120,   # one shell command
    "agent_output_chars": 10000,        # longer output: model sees start + end (mini-swe-agent uses 10k)
    "agent_task_timeout_sec": 1800,     # whole agent loop (Terminal-Bench tasks use their own limit)
    "agent_verifier_timeout_sec": 1800, # SWE-bench test run
}


def generate_answers(run_id: str, items: list[dict], settings: dict, port: int) -> list[dict]:
    run_config = load_config()["run"]
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="none",
                    timeout=run_config.get("request_timeout_sec", 900),
                    max_retries=run_config.get("request_retries", 0))
    results = []
    set_stage(run_id, "generating", 0, len(items))
    question_items = [item for item in items if item["grader"] != "agent"]
    agent_items = [item for item in items if item["grader"] == "agent"]

    def ask_and_save(item: dict) -> dict:
        if run_id in cancel_requested:
            return {}
        result = ask_test_model(client, settings["model_name"], item, settings)
        append_result(run_id, result)
        with results_lock:  # 8 threads update this counter
            live_progress[run_id]["done"] += 1
        return result

    with ThreadPoolExecutor(max_workers=load_config()["run"]["parallel_requests"]) as pool:
        results = [result for result in pool.map(ask_and_save, question_items) if result]
    check_cancel(run_id)

    # Agent tasks (SWE-bench, Terminal-Bench): one at a time, each image deleted afterwards
    limits = {key: run_config.get(key, default) for key, default in AGENT_LIMIT_DEFAULTS.items()}
    # The agent keeps as much history as fits in the test model's context window
    limits["context_tokens"] = int(option_value(shlex.split(settings["command"]), "--max-model-len") or 32768)
    for item in agent_items:
        check_cancel(run_id)
        result = agent.run_agent_task(client, settings["model_name"], item, settings, limits,
                                      should_stop=lambda: run_id in cancel_requested)
        append_result(run_id, result)
        results.append(result)
        with results_lock:
            live_progress[run_id]["done"] += 1
    check_cancel(run_id)
    return results


def judge_answers(run_id: str, results: list[dict], items_by_id: dict, judge: dict) -> None:
    _, judge_port, judge_model = parse_command(judge["command"])
    run_config = load_config()["run"]
    client = OpenAI(base_url=f"http://127.0.0.1:{judge_port}/v1", api_key="none",
                    timeout=run_config.get("judge_timeout_sec", 600),
                    max_retries=run_config.get("request_retries", 0))
    waiting = [result for result in results if result["verdict"] == "needs_judge"]
    set_stage(run_id, "judging", 0, len(waiting))

    def judge_one(result: dict) -> None:
        if run_id in cancel_requested:
            return
        try:
            result.update(ask_judge(client, judge_model, items_by_id[result["id"]], result["answer"]))
        except Exception as error:
            result.update(verdict="error", reason=f"judge request failed: {error}")
        with results_lock:  # 8 threads update this counter
            live_progress[run_id]["done"] += 1

    with ThreadPoolExecutor(max_workers=load_config()["run"]["parallel_requests"]) as pool:
        list(pool.map(judge_one, waiting))
    check_cancel(run_id)


def execute_run(run_id: str) -> None:
    settings = read_settings(run_id)
    config = load_config()
    timeout_sec = config["run"]["server_start_timeout_sec"]
    folder = run_dir(run_id)
    test_process = None
    judge_process = None
    try:
        judge = find_judge(config, settings["judge"])
        _, test_port, _ = parse_command(settings["command"])
        _, judge_port, _ = parse_command(judge["command"])

        items = []
        for dataset in settings["datasets"]:
            items += pick_questions(load_dataset(dataset), settings["size"], settings["seed"])
        items_by_id = {item["id"]: item for item in items}

        set_stage(run_id, "starting_model")
        test_process = start_server(settings["command"], folder / "test_model.log")
        wait_until_healthy(test_process, test_port, timeout_sec, folder / "test_model.log", run_id)
        check_cancel(run_id)

        results = generate_answers(run_id, items, settings, test_port)
        stop_server(test_process)  # free the GPU before the judge starts

        if any(result["verdict"] == "needs_judge" for result in results):
            set_stage(run_id, "starting_judge")
            judge_process = start_server(judge["command"], folder / "judge.log")
            wait_until_healthy(judge_process, judge_port, timeout_sec, folder / "judge.log", run_id)
            check_cancel(run_id)
            judge_answers(run_id, results, items_by_id, judge)
            rewrite_results(run_id, results)

        (folder / "summary.json").write_text(json.dumps(summarize(results), indent=2))
        set_stage(run_id, "done", len(results), len(results))
    except RunCancelled:
        update_settings(run_id, status="cancelled")
    except Exception as error:
        update_settings(run_id, status="failed", error=str(error))
    finally:
        stop_server(test_process)
        stop_server(judge_process)
        cancel_requested.discard(run_id)
        live_progress.pop(run_id, None)
