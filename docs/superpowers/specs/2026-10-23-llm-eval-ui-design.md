# LLM Evaluation UI — Design

Date: 2026-10-23
Folder: `/mnt/newdisk/neel/LLM_EVALUATION`

## 1. What and why

A web app to test LLMs served by vLLM. The user does everything in the browser:
paste a vLLM command, write a system prompt, pick datasets, pick run size,
pick a judge, start the run, watch progress, and compare old runs.

Why: the user wants to see how well a model (and a system prompt) does on
math, tool calling, RAG, terminal commands, coding, knowledge, and on their
own custom questions, and to compare runs side by side.

**Success:** a user who never touches the terminal can start a run, get a
score per dataset, read each question/answer/verdict, and compare two runs.

**Non-goals (v1):** Terminal-Bench (Docker), multi-GPU scheduling, user
accounts, BFCL multi-turn / live / Java / JS categories.

**Anti-goals:** code the user can't read; heavy test suites; anything that
touches the user's tmux sessions.

## 2. Constraints

- One GPU: NVIDIA L40S, 46 GB.
- Code lives and runs in this folder, with its own `.venv` (uv, Python 3.12, vllm installed).
- Backend: simple Python. Frontend: HTML + CSS (+ a small plain JS file for
  buttons and progress polling — no frameworks, no build step).
- Few tests. Each test has a comment: why it exists, and when it failed in my testing.
- NAS `/mnt/truenas` is read-only for the app (CIFS mount forced to another uid).
  The app uses sudo only as the Docker fallback (§10).

## 3. Data

### Raw datasets (already downloaded, read-only)
`/mnt/truenas/llm_eval_datasets/raw/`

| Category | Dataset | Grading |
|---|---|---|
| Math | gsm8k (test), math500 | code: compare final answer; if not equal → judge |
| Tool calling | bfcl (simple, multiple, parallel, parallel_multiple — python) | code: function names + args vs allowed values |
| RAG | hotpotqa (distractor, validation), squad_v2 (validation) | judge |
| Coding | humaneval, mbpp (test) | code: run unit tests inside Docker (`python:3.12-slim`, `--network none`, memory/CPU limits, removed after) |
| Knowledge | mmlu_pro (test), gpqa (diamond) | code: compare option letter |
| Reasoning | arc (Challenge, test) | code: compare option letter |
| Instruction following | ifeval | judge (v1) |
| Truthfulness | truthfulqa (generation) | judge |
| Terminal | nl2bash | judge |

### Converted datasets (app-writable)
`data/datasets/<name>.jsonl` — one question per line, same fields for all:

```json
{"id": "gsm8k-17", "dataset": "gsm8k", "category": "math",
 "question": "...", "ground_truth": "42",
 "context": "...optional...", "choices": ["..."], "tools": [...], "tests": "...",
 "grader": "math"}
```
Only `id`, `question`, `ground_truth`, `grader` are required.
A one-time script `prepare_datasets.py` reads raw files and writes these.

### Custom uploads
User uploads `.jsonl` or `.csv` with at least `question` and `ground_truth`
columns. Saved to `data/datasets/custom_<name>.jsonl`, grader = `judge`.

### Runs
`data/runs/<run_id>/`
- `settings.json` — everything the user picked (command, system prompt, datasets, size, judge, seed)
- `results.jsonl` — one line per question: question, ground truth, model answer, reasoning (if any), verdict, reason, time
- `summary.json` — score per dataset + total
- `test_model.log`, `judge.log` — vLLM output, shown in UI if something fails

## 4. config.yml (default file, editable in UI)

```yaml
judges:
  - name: qwen3.5-4b-awq (fast)
    command: >-
      python -m vllm.entrypoints.openai.api_server
      --model /mnt/truenas/qwen3_5_AWQ_4bit --served-model-name judge
      --port 8009 --gpu-memory-utilization 0.2 --max-model-len 8192
    keep_running: true     # small: stays loaded next to the test model
  - name: qwen3-30b (accurate)
    command: >-
      python -m vllm.entrypoints.openai.api_server --model /mnt/newdisk/neel/whisper_to_fasterwhisper/model/model_qwen3_6_coding
      --served-model-name qwen3b --port 8008 --dtype bfloat16 --gpu-memory-utilization 0.95
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --kv-cache-dtype auto
      --max-num-seqs 2 --max-num-batched-tokens 2096 --enable-prefix-caching
      --tensor-parallel-size 1 --disable-log-stats --reasoning-parser qwen3
    keep_running: false    # big: test model must stop first
default_judge: qwen3.5-4b-awq (fast)

test_model_presets:        # quick-fill options in the UI; user can still paste any command
  - name: qwen3.5-4b
    command: python -m vllm.entrypoints.openai.api_server --model /mnt/truenas/qwen3-5_4b --served-model-name test --port 8001 --gpu-memory-utilization 0.7

run:
  quick_size: 100
  seed: 42
  temperature: 0
  max_tokens: 4096
  parallel_requests: 8
  code_test_timeout_sec: 10
  server_start_timeout_sec: 900

paths:
  raw_datasets: /mnt/truenas/llm_eval_datasets/raw
  data: data
```
Judges added in the UI are written back into `config.yml`.
`python` in a command is replaced by `.venv/bin/python` so it uses this project's vLLM.

## 5. One run, step by step

1. User fills the form: vLLM command (paste or preset), system prompt,
   datasets (checkboxes, incl. custom), size (Quick 100 / Full / custom N), judge (dropdown).
2. App starts the test model (`subprocess.Popen`, no shell), waits for `GET /health`.
   Port and model name come from `--port` / `--served-model-name` in the command.
   Crash or timeout → run marked failed, last 50 log lines shown.
3. Pick questions: Full = all; Quick/N = random sample with fixed seed → same questions every run.
4. Send each question (system prompt + question [+ context / choices / tools]) to the test model,
   `parallel_requests` at a time. Save each answer to `results.jsonl` immediately.
   Thinking text (`reasoning_content`) is stored separately; only the final answer is graded.
5. Code graders run right after each answer.
6. If the judge has `keep_running: false`: stop test model, then start judge.
   If `keep_running: true`: start judge once (if not up) and leave it running.
7. Judge gets question, ground truth (and context for RAG), model answer.
   Must reply JSON: `{"verdict": "correct"|"incorrect", "reason": "..."}`. Bad JSON → retry once, then `verdict: "error"`.
8. Stop test model (and the judge if `keep_running: false`). Write `summary.json`.
9. UI shows score per dataset and a table of every question (filter: wrong only).

Only one run at a time; extra runs wait in a queue (one GPU).
A run can be cancelled; answers already saved are kept.

## 6. Pages

- **New run** — the form in step 1 + Start.
- **Runs** — list of runs (name, model, date, status, total score), live progress for the active one.
- **Run detail** — per-dataset scores, question table, logs.
- **Compare** — pick 2+ runs → score table side by side + questions where verdicts differ.
- **Datasets** — list of datasets with counts, upload custom.
- **Settings** — edit judges and presets (writes `config.yml`).

## 7. Code layout

```
app.py              FastAPI server: API routes + serves static/   (FastAPI comes with vllm)
runner.py           run queue, start/stop vLLM, ask questions, call graders, call judge
graders.py          math / choice / code-tests / tool-call / judge-prompt
prepare_datasets.py raw NAS files → data/datasets/*.jsonl (run once)
config.yml
static/index.html, style.css, app.js
data/               datasets/, runs/   (created at start)
```
No new dependencies: fastapi, uvicorn, pyyaml, openai all come with vllm.

## 8. Errors

- vLLM fails to start (bad path, out of GPU memory) → run failed, log tail shown.
- Request timeout / error for one question → that question gets `verdict: "error"`, run continues.
- Code test hangs → container killed after `code_test_timeout_sec`, counted incorrect.
- Docker not available → coding datasets show an error; other datasets still run.
- App restart during a run → run marked "interrupted"; saved answers kept.

## 9. Decisions made by Claude (check these)

1. FastAPI + plain JS for buttons/polling (pure HTML+CSS can't show live progress).
2. Converted datasets stored locally in `data/datasets/` (app can't write NAS).
3. IFEval graded by judge in v1, not the official rule checks (they need extra libraries).
4. Math: code compare first, judge only when they don't match (handles `0.5` vs `\frac{1}{2}`).
5. Judge returns correct/incorrect + reason (not 1–10).
6. 4B judge uses 20% GPU memory and stays loaded; presets use 70% for the test model.
7. HotpotQA uses validation split (test has no answers); BFCL only the 4 python single-turn categories.
8. Temperature 0 by default, editable per run.

## 10. Network and safety (user decided)

- UI binds `0.0.0.0:7860` → reachable from any device on the same router/LAN. No login (v1).
- Because anyone on the LAN can start a run: the app only accepts commands that start with
  `python -m vllm.entrypoints.openai.api_server` or `vllm serve`, runs them without a shell
  (`shlex.split`, no `;`/`&&` tricks), and shows a warning if `--trust-remote-code` is present. (Claude's call)
- Model-written code never runs on the host — only in Docker with no network.
- Docker access: plain `docker` → else `sudo docker` with the `sudo_user` password from `.env` → else coding datasets are disabled (optional for users without Docker). (user decided)
