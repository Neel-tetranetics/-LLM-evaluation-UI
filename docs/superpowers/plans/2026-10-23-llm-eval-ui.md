# LLM Evaluation UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Browser app that starts a vLLM test model from a pasted command, runs chosen datasets with a user system prompt, grades answers by code or a judge LLM, and compares runs.

**Architecture:** One FastAPI process (`app.py`) serves a static HTML/CSS/JS UI and a JSON API. A single worker thread in `runner.py` runs queued evaluations: start vLLM → ask questions → grade (`graders.py`) → judge → stop vLLM. Everything is stored as JSON/JSONL files under `data/`.

**Tech Stack:** Python 3.12, `.venv` (vllm 0.32.0 brings fastapi, uvicorn, pyyaml, openai, requests), pyarrow (installed, for parquet), Docker (`python:3.12-slim`) for code tests, plain HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-10-23-llm-eval-ui-design.md`

## Global Constraints

- Run everything from `/mnt/newdisk/neel/LLM_EVALUATION` with `.venv/bin/python`. Never use other projects' envs. Never send keys to the user's tmux sessions.
- Simple, readable Python. Meaningful names. Short functions. Comments say *why*.
- Few tests: only `tests/test_graders.py`, plain `assert`s, runnable as `.venv/bin/python tests/test_graders.py` (no pytest). Every test has a comment: why it exists + when it failed during development.
- No new dependencies beyond what is installed (vllm stack + pyarrow). File upload goes as JSON text (no python-multipart).
- NAS `/mnt/truenas/llm_eval_datasets/raw` is read-only. App writes only under `data/`. The only sudo use is the Docker fallback below.
- Docker access order (user decided): 1) plain `docker`; 2) if permission denied and `.env` has `sudo_user`, run `sudo -S -p '' docker ...` with that password on stdin (never logged/printed); 3) neither works → code datasets (humaneval, mbpp) marked unavailable in the UI, everything else still runs. Checked once at startup by `graders.docker_command() -> list[str] | None`.
- UI served on `0.0.0.0:7860` (LAN access).
- Commands accepted only if they start with `python -m vllm.entrypoints.openai.api_server` or `vllm serve`; split with `shlex.split`, run without a shell; leading `python` → `.venv/bin/python`, leading `vllm` → `.venv/bin/vllm`.
- Model-written code runs only in Docker: `--rm --network none --memory 512m --cpus 1`.
- Not a git repo: no commit steps. Do not `git init` unless the user asks.

## Review Focus

1. Thinking model spends all `max_tokens` thinking → `content` empty → must be graded incorrect with reason `"empty answer"`, not crash. (test in Task 2)
2. Model answer without a ```python block → code grader uses the whole answer text as code; still no crash. (test in Task 2)
3. Port in the test-model command is already in use → run fails at once with `"port <N> already in use"` instead of waiting for timeout or testing the wrong model. (check in Task 3, Step 4)
4. Custom upload with a bad JSON line or missing `question`/`ground_truth` → upload rejected with the line number. (check in Task 4, Step 3)
5. Test model and judge use the same port → run rejected at submit. (check in Task 4, Step 3)

---

### Task 1: Config file + dataset conversion

**Files:**
- Create: `config.yml` (content exactly as spec §4)
- Create: `prepare_datasets.py`

**Interfaces:**
- Produces: `data/datasets/<name>.jsonl`, one JSON object per line with keys
  `id, dataset, category, grader, question, ground_truth` and optional `context, choices, tools, tests`.
  `grader` ∈ `"math" | "choice" | "code" | "tool_call" | "judge"`.

Dataset table (name → source → grader → expected count):

| name | source | grader | count |
|---|---|---|---|
| gsm8k | gsm8k/main/test parquet; ground_truth = text after `####`, commas removed | math | 1319 |
| math500 | math500/test.jsonl `problem` / `answer` | math | 500 |
| bfcl | bfcl/BFCL_v3_{simple,multiple,parallel,parallel_multiple}.json + possible_answer/ | tool_call | 1000 |
| hotpotqa | hotpotqa/distractor/validation | judge | 7405 |
| squad_v2 | squad_v2/squad_v2/validation | judge | 11873 |
| humaneval | humaneval/openai_humaneval/test | code | 164 |
| mbpp | mbpp/sanitized/test | code | 257 |
| mmlu_pro | mmlu_pro/data/test (`options`, `answer` letter) | choice | 12032 |
| gpqa_diamond | gpqa/gpqa_diamond.csv | choice | 198 |
| arc_challenge | arc/ARC-Challenge/test (`choices.text/label`, `answerKey`) | choice | 1172 |
| ifeval | ifeval/ifeval_input_data.jsonl | judge | 541 |
| truthfulqa | truthfulqa/generation/validation | judge | 817 |
| nl2bash | nl2bash/all.nl + all.cm (same line number) | judge | 12607 |

Per-grader field rules (decisions the implementer cannot guess):
- **math:** `question` = problem + `"\n\nPut your final answer in \\boxed{}."`
- **choice:** `choices` = list of option texts; `question` = question + lines `"A. ..."`, `"B. ..."` + `"\n\nEnd your reply with 'Answer: X' where X is the letter."`; `ground_truth` = letter. GPQA: shuffle `[Correct, Incorrect 1-3]` with `random.Random(<row index>)` so order is fixed forever.
- **code (humaneval):** `question` = `"Complete this Python function. Reply with the full function in a ```python block.\n\n"` + prompt; `tests` = test + `f"\ncheck({entry_point})\n"`; `context` = prompt (used if model returns only the body).
- **code (mbpp):** `question` = prompt + `"\nYour code must pass this test:\n"` + test_list[0] + same reply instruction; `tests` = test_imports + test_list joined by newlines.
- **tool_call:** `question` = content of first user message of `question[0]`; `tools` = OpenAI format `[{"type":"function","function":{"name": name.replace(".", "_"), "description", "parameters"}}]` with every `"type": "dict"` changed to `"object"` (recursively); `ground_truth` = possible_answer `ground_truth` list with function names also `.`→`_`.
- **judge:** hotpotqa `context` = each title + its sentences joined, separated by blank lines; squad_v2 `ground_truth` = first answer, or `"The question cannot be answered from the context."` when there are none; truthfulqa `ground_truth` = `"Best answer: ...\nAlso correct: ...\nIncorrect (must not say): ..."`; ifeval `question` = prompt, `ground_truth` = `"The reply must follow every instruction in the prompt: "` + instruction_id_list + kwargs as text; nl2bash `ground_truth` = command.
- `id` = `"<name>-<source id or row index>"`, unique per file.

- [ ] **Step 1: Write `config.yml`** exactly as spec §4.
- [ ] **Step 2: Implement `prepare_datasets.py`** — one function per dataset `convert_<name>(raw_dir: Path) -> list[dict]`, a `write_jsonl(path, rows)` helper, and `main()` that writes all files and prints `name count`. Read raw path from `config.yml` `paths.raw_datasets`. Run with `.venv/bin/python -I` style safety not needed (our own script).
- [ ] **Step 3: Run it**

Run: `.venv/bin/python prepare_datasets.py`
Expected: 13 lines, counts equal to the table above. Then `head -c 600 data/datasets/bfcl.jsonl` shows `"type": "object"` and no `.` in function names.

---

### Task 2: Graders

**Files:**
- Create: `graders.py`
- Create: `tests/test_graders.py`

**Interfaces:**
- Produces:
  - `extract_boxed(text: str) -> str | None` — content of the last `\boxed{...}` (brace-balanced), else last number in text.
  - `normalize_math(answer: str) -> str` — strip spaces, `$`, trailing `.`, commas in numbers, `\left`/`\right`; `"0.50"`→`"0.5"`.
  - `grade_math(answer: str, ground_truth: str) -> dict` → `{"verdict": "correct"}` on normalized match, else `{"verdict": "needs_judge"}`.
  - `extract_choice(text: str) -> str | None` — letter from last `Answer: X` (case-insensitive, A–J), else last standalone capital A–J.
  - `grade_choice(answer, ground_truth) -> dict` → correct/incorrect.
  - `extract_code(text: str) -> str` — last ```python block, else any ``` block, else whole text.
  - `grade_code(answer: str, item: dict, timeout_sec: int) -> dict` — program = code (prefixed by `item["context"]` if the code lacks `def <entry>`... simpler: if `item.get("context")` and the code does not start with `def`/`import`/`from`, prepend context) + `"\n" + item["tests"]`; write program to a temp dir and run in Docker (`<docker_command()> run --rm --network none --memory 512m --cpus 1 -v <tmp>:/code:ro python:3.12-slim timeout <timeout_sec> python /code/program.py`; for the sudo variant the password goes on stdin, which is why the program is mounted, not piped); exit 0 → correct; else incorrect with last 300 chars of stderr as reason; Docker missing → `{"verdict": "error", "reason": "docker not available: ..."}`.
  - `docker_command() -> list[str] | None` — `["docker"]` if `docker info` works, else `["sudo", "-S", "-p", "", "docker"]` if that works with the `.env` password, else `None`. Cached after first call.
  - `grade_tool_call(tool_calls: list[dict], ground_truth: list[dict]) -> dict` — `tool_calls` items `{"name": str, "arguments": dict}`. Correct when counts match and each expected call is matched by a distinct actual call (order-free) with same name and every expected param: actual value in allowed list, or param missing and `""` in allowed list. Reason names the first mismatch.
  - `build_judge_messages(item: dict, answer: str) -> list[dict]` — system prompt below + user message with Question / Context (if any) / Ground truth / Model answer.
  - `parse_judge_reply(text: str) -> dict` — first `{...}` JSON in text with `verdict` in (`correct`,`incorrect`); else raise `ValueError`.
  - `grade_without_judge(item: dict, answer: str, tool_calls: list[dict], timeout_sec: int) -> dict` — dispatch on `item["grader"]`; empty `answer` and no `tool_calls` → `{"verdict": "incorrect", "reason": "empty answer"}`; `"judge"` grader → `{"verdict": "needs_judge"}`.

Judge system prompt (exact copy):
```
You are a strict grader. Compare the model answer with the ground truth.
Mark "correct" only if the model answer gives the same final answer or meaning as the ground truth. Extra explanation is fine. Missing, wrong, or contradicting parts make it "incorrect".
If a context is given, the answer must be supported by the context.
Reply with JSON only: {"verdict": "correct" or "incorrect", "reason": "<one short sentence>"}
```

- [ ] **Step 1: Write `tests/test_graders.py`** — plain functions + `if __name__ == "__main__":` calling each and printing `ALL PASSED`. Cases (each with a `# why:` and `# failed when:` comment, filled after running):
  - `extract_boxed("so \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"` (nested braces)
  - `grade_math("The answer is \\boxed{1,000}", "1000")["verdict"] == "correct"`
  - `extract_choice("I think B... Answer: d") == "D"`
  - `extract_code("text\n```python\nx=1\n```") == "x=1"` and `extract_code("x=1") == "x=1"` (Review Focus 2)
  - `grade_tool_call([{"name":"spotify_play","arguments":{"artist":"Maroon 5","duration":15}},{"name":"spotify_play","arguments":{"artist":"Taylor Swift","duration":20}}], [{"spotify_play":{"artist":["Taylor Swift"],"duration":[20]}},{"spotify_play":{"artist":["Maroon 5"],"duration":[15]}}])["verdict"] == "correct"` (order-free)
  - same with `duration` 99 → `"incorrect"`
  - `grade_without_judge({"grader":"math","ground_truth":"3"}, "", [], 10) == {"verdict":"incorrect","reason":"empty answer"}` (Review Focus 1)
  - `parse_judge_reply('<think>x</think> {"verdict": "correct", "reason": "ok"}')["verdict"] == "correct"`
- [ ] **Step 2: Run, expect failure** — `.venv/bin/python tests/test_graders.py` → `ModuleNotFoundError: graders`.
- [ ] **Step 3: Implement `graders.py`** with the interfaces above.
- [ ] **Step 4: Run, expect `ALL PASSED`.** Fill in each `# failed when:` comment with what actually failed (or `never failed` if it passed first time).
- [ ] **Step 5: Docker check** (verified 2026-10-23: plain docker = permission denied, sudo docker works): `docker pull python:3.12-slim`, then in Python `grade_code("```python\ndef f():\n    return 1\n```", {"tests":"assert f()==1"}, 10)` → correct, and with `"assert f()==2"` → incorrect. Not added as a test (needs Docker); just run once.

---

### Task 3: Runner (vLLM control + run loop)

**Files:**
- Create: `runner.py`

**Interfaces:**
- Consumes: `graders.grade_without_judge`, `graders.build_judge_messages`, `graders.parse_judge_reply`; dataset files from Task 1.
- Produces:
  - `load_config() -> dict`, `save_config(config: dict) -> None` (writes `config.yml`).
  - `parse_command(command: str) -> tuple[list[str], int, str]` — returns `(argv, port, served_model_name)`; raises `ValueError` with a plain message for disallowed prefix or missing `--port`. Model name = `--served-model-name` value, else `--model` value (or the positional model for `vllm serve`).
  - `start_server(command: str, log_path: Path) -> subprocess.Popen` — refuses with `ValueError(f"port {port} already in use")` if `GET http://127.0.0.1:{port}/health` already answers; `start_new_session=True`, stdout+stderr to log file.
  - `wait_until_healthy(process, port: int, timeout_sec: int, log_path: Path) -> None` — polls `/health` every 3 s; process exit or timeout → `RuntimeError` containing last 50 log lines.
  - `stop_server(process) -> None` — `os.killpg` SIGTERM, wait 30 s, then SIGKILL.
  - `pick_questions(rows: list[dict], size: str | int, seed: int) -> list[dict]` — `"full"` → all; `"quick"` → `config.run.quick_size`; int → that many; sampling with `random.Random(seed).sample` then sorted by original order.
  - `submit_run(form: dict) -> str` — validates, creates `data/runs/<run_id>/settings.json` with status `queued`, puts id on queue, returns id. `run_id` = `time.strftime("%Y%m%d-%H%M%S") + "-" + slug(name)`.
  - `get_run(run_id) -> dict` (settings + status + progress + summary), `list_runs() -> list[dict]`, `get_results(run_id) -> list[dict]`, `cancel_run(run_id) -> None`.
  - `mark_interrupted_runs() -> None` — at startup, any run whose status is not `done|failed|cancelled` becomes `interrupted`.
  - `start_worker() -> None` — starts the one daemon thread reading the queue.
  - Status values: `queued, starting_model, generating, starting_judge, judging, done, failed, cancelled, interrupted`. Progress = `{"done": int, "total": int, "stage": str}` kept in memory and written into `settings.json` on stage change.

Run loop (`execute_run(run_id)`), in order — this is the part the spec fixes:
1. If chosen judge has `keep_running: false`: stop any kept-running judges first (GPU). Start test model, wait healthy.
2. For each question (ThreadPool, `parallel_requests`): chat completion with `[system prompt (if any), user question (+ "\n\nContext:\n" + context if present)]`, `tools` if present, `temperature`, `max_tokens`. Keep `content`, `reasoning_content` (if any), and tool calls as `{"name", "arguments": json.loads(...)}` (bad JSON args → keep as `{}` and grade incorrect). Request error → `verdict: "error"` with message. Grade with `grade_without_judge`. Append line to `results.jsonl` immediately (one lock for the file). Check cancel flag between questions.
3. Collect `needs_judge` items. If none, skip judge. If judge `keep_running: false` → stop test model now. Start judge if its `/health` does not answer (kept judges stay in a module-level dict `running_judges[name] = process`).
4. Judge each item (`temperature 0`, `max_tokens 512`, `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`), `parse_judge_reply`; on `ValueError` retry once, then `verdict: "error"`. Rewrite `results.jsonl` with final verdicts.
5. Stop test model (if still running) and non-kept judge. Write `summary.json`: per dataset `{"correct", "incorrect", "error", "total", "score": correct/total}` and `"overall"`.
6. Any exception → status `failed`, `error` message saved; always stop the test model in `finally`.

- [ ] **Step 1: Implement `runner.py`.**
- [ ] **Step 2: Smoke check `parse_command`** in a Python shell: the 30B judge command from `config.yml` → port 8008, name `qwen3b`; `"rm -rf /"` → `ValueError`.
- [ ] **Step 3: Real run from Python (no UI):** `submit_run({"name":"smoke","command": <qwen3.5-4b preset>,"system_prompt":"","datasets":["gsm8k","arc_challenge","humaneval","truthfulqa","bfcl"],"size":3,"judge":"qwen3.5-4b-awq (fast)","temperature":0,"max_tokens":2048})`, `start_worker()`, wait. Expected: status `done`, `results.jsonl` has 15 lines each with a verdict, `summary.json` exists, `nvidia-smi` shows test model gone and 4B judge still loaded.
- [ ] **Step 4: Review Focus 3:** while the 4B judge runs on 8009, submit a run whose test command uses `--port 8009` → status `failed`, error `port 8009 already in use`.

---

### Task 4: API server

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: everything `runner.py` produces.
- Routes (all JSON):
  - `GET /api/config` → config dict.
  - `POST /api/judges` `{name, command, keep_running}` → validates with `parse_command`, appends, `save_config`. `DELETE /api/judges/{name}` (cannot delete `default_judge`).
  - `POST /api/presets` `{name, command}`, `DELETE /api/presets/{name}`.
  - `GET /api/datasets` → `[{"name", "category", "count", "available"}]` (`available` false for code datasets when `docker_command()` is None) from `data/datasets/*.jsonl` (count cached by file mtime).
  - `POST /api/datasets/upload` `{name, filename, content}` — `.jsonl` or `.csv`; every row needs non-empty `question` and `ground_truth`; error → HTTP 400 with `"line <N>: <problem>"`; saved as `data/datasets/custom_<slug>.jsonl` with `grader: "judge"`, `category: "custom"`.
  - `POST /api/runs` (form) → `{"run_id"}`; HTTP 400 on validation errors, including test port == judge port.
  - `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/runs/{id}/results`, `POST /api/runs/{id}/cancel`, `GET /api/runs/{id}/log/{test_model|judge}` (last 200 lines, text).
  - `GET /api/compare?ids=a,b,...` → `{"runs": [get_run...], "differences": [{"id", "dataset", "question", "verdicts": {run_id: verdict}}]}` — only ids present in all runs whose verdicts differ.
  - `/` serves `static/index.html`; `/static` mounted.
- On startup: create `data/datasets`, `data/runs`; `mark_interrupted_runs()`; `start_worker()`. `__main__`: `uvicorn.run(app, host="0.0.0.0", port=7860)`.

- [ ] **Step 1: Implement `app.py`.**
- [ ] **Step 2: Start it** — `.venv/bin/python app.py`, then `curl -s localhost:7860/api/datasets` lists 13 datasets with Task 1 counts.
- [ ] **Step 3: Review Focus 4 & 5 with curl:** upload content `'{"question":"a"}\n'` → 400 `line 1: missing ground_truth`; upload `'not json'` → 400 `line 1: ...`; POST run with test `--port 8009` and the 4B judge (port 8009) → 400 mentioning the same port.

---

### Task 5: UI

**Files:**
- Create: `static/index.html`, `static/style.css`, `static/app.js`

**Interfaces:**
- Consumes: Task 4 routes only.
- One page, six sections switched by a top nav (only one visible): **New run, Runs, Run detail, Compare, Datasets, Settings** (spec §6).
- New run: name, preset dropdown (fills command textarea), command textarea, system prompt textarea, dataset checkboxes grouped by category with counts, size radio Quick / Full / Custom N, judge dropdown (default from config), temperature, max tokens, Start. Show `--trust-remote-code` warning text under the command box when present.
- Runs: table (name, model, date, status, overall score), auto-refresh every 3 s while any run is active; active run shows `stage done/total`.
- Run detail: settings, per-dataset score table, results table with "wrong only" filter and expandable answer/reasoning/reason, log viewer buttons.
- Compare: checkboxes on Runs table → Compare button → score table (rows datasets, columns runs) + differences table.
- Datasets: table + upload form (file input read with `FileReader` → JSON POST).
- Settings: judges list (name, command, keep_running, delete) + add form; presets same.
- Errors from API shown in a red message box at the top, never silently ignored.
- CSS: plain, readable, works at phone width (tables scroll horizontally inside their box).

- [ ] **Step 1: Write the three files.**
- [ ] **Step 2: Open `http://<server-LAN-IP>:7860` from another device on the router** and run a Quick run with size 5 on 3 datasets through the UI only. Expected: progress moves, run reaches `done`, scores and per-question table visible.
- [ ] **Step 3: Compare** the Task 3 smoke run with this run → score table + differences render.

---

### Task 6: README

**Files:**
- Create: `README.md`

- [ ] **Step 1:** Write per the user's README rules (what, why, how it works, folder structure, requirements incl. docker group, install, config, run, example, common errors: docker permission denied, port in use, CUDA out of memory, NAS not mounted). Only commands that were actually run in Tasks 1–5.
