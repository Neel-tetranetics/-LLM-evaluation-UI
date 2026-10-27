"""Agent benchmarks: SWE-bench Verified and Terminal-Bench 2.0.

The model works inside the task's Docker container, one shell command per step:
    pull image -> start container -> agent loop -> run the benchmark's tests -> remove container + image
Only one task runs at a time, and its image is deleted afterwards, so disk use stays at one image.
"""

import os
import re
import tempfile
import time
import uuid
from pathlib import Path

from openai import OpenAI

from graders import run_docker

AGENT_INSTRUCTIONS = """You are working in a Linux shell inside a Docker container to complete a task.
In each reply, think briefly, then give exactly ONE shell command in a ```bash code block. You will get its output.
Every command runs in a fresh bash shell, so `cd` does not carry over: write `cd /some/dir && command`.
Never use interactive programs (vim, nano, less, top, a Python REPL). Edit files with sed, python, or cat <<'EOF' > file.
When the task is fully done, reply with only the word TASK_COMPLETE."""

COMPLETE_WORD = "TASK_COMPLETE"

# SWE-bench images keep the repo in /testbed and its Python env in conda env "testbed"
SWE_SHELL_SETUP = "source /opt/miniconda3/bin/activate && conda activate testbed && cd /testbed && "

PASSING_STATUSES = {"PASSED", "XFAIL"}  # same rule as the official SWE-bench harness


# ---------- docker helpers ----------

def run_in_container(container: str, command: str, timeout_sec: float) -> tuple[int, str]:
    """Run one shell command in the container. Returns (exit code, combined output)."""
    try:
        # `timeout` inside the container stops the command; the outer limit is a safety net
        # `exec 2>&1`: errors stay in the same order as normal output (SWE-bench's test markers go to
        # stderr while test results go to stdout; reading them separately broke the result parsing)
        finished = run_docker(["exec", container, "timeout", str(int(timeout_sec)), "bash", "-lc",
                               "exec 2>&1\n" + command], timeout_sec + 30)
    except Exception as error:  # subprocess timeout or docker failure: report it to the agent as output
        return 124, f"command could not finish: {error}"
    return finished.returncode, finished.stdout + finished.stderr


def start_container(item: dict, pull: bool = True) -> str:
    if pull:
        pulled = run_docker(["pull", item["image"]], 1800)
        if pulled.returncode != 0:
            raise RuntimeError(f"docker pull failed: {pulled.stderr[-300:]}")
    container = "llm-eval-" + uuid.uuid4().hex[:10]
    started = run_docker([
        "run", "-d", "--name", container,
        "--memory", item.get("memory", "4g"), "--cpus", str(item.get("cpus", 2)),
        "--entrypoint", "sleep", item["image"], "infinity",
    ], 300)
    if started.returncode != 0:
        raise RuntimeError(f"docker run failed: {started.stderr[-300:]}")
    return container


def remove_container_and_image(container: str | None, image: str) -> None:
    """Free the disk right away, as the user asked: one task's image on disk at a time."""
    if container:
        run_docker(["rm", "-f", container], 120)
    run_docker(["rmi", "-f", image], 300)


# ---------- the agent loop ----------

def shorten_output(output: str, limit: int) -> str:
    """Whole output if short; otherwise its start and end plus a hint (same idea as mini-swe-agent).

    Keeping only the end hid file headers and the first error lines from the model.
    """
    if len(output) <= limit:
        return output
    half = limit // 2
    return (f"[Output too long: {len(output)} characters. Showing the first and last {half}. "
            f"Use a command with less output, e.g. head, grep, or sed -n.]\n"
            f"{output[:half]}\n... [{len(output) - 2 * half} characters omitted] ...\n{output[-half:]}")


# Rough size estimate: code and logs average about 3 characters per token.
# ponytail: character estimate, not a real tokenizer; the 2,000-token margin covers the error.
CHARS_PER_TOKEN = 3


def recent_history(messages: list[dict], context_tokens: int, reply_tokens: int) -> list[dict]:
    """Keep the system prompt, the task, and as many recent steps as fit in the model's context.

    Dropping only what does not fit matters: when the history was cut to the last 20 steps,
    the 30B model forgot what it had read and re-read the same file 59 times without editing.
    """
    budget_chars = max(0, context_tokens - reply_tokens - 2000) * CHARS_PER_TOKEN
    first_two, steps = messages[:2], messages[2:]
    size = sum(len(message["content"]) for message in messages)
    dropped = 0
    # One step = assistant reply + its output, so drop in pairs and keep the order valid
    while size > budget_chars and len(steps) - dropped > 2:
        size -= len(steps[dropped]["content"]) + len(steps[dropped + 1]["content"])
        dropped += 2
    if dropped == 0:
        return messages
    note = {"role": "user", "content": "[Older steps were removed to save space. Continue from the recent steps below.]"}
    return first_two + [note] + steps[dropped:]

def last_bash_block(text: str) -> str | None:
    blocks = re.findall(r"```(?:bash|sh|shell)?[^\n]*\n(.*?)```", text, re.DOTALL)
    return blocks[-1].strip() if blocks else None


def task_text(item: dict) -> str:
    if item["agent_type"] == "swe":
        return ("The repository is checked out in /testbed (Python env: conda env 'testbed', already activated "
                "for your commands). Change the source code so this issue is fixed. Do not change the tests.\n\n"
                + item["question"])
    return item["question"]


def run_agent(client: OpenAI, model_name: str, item: dict, settings: dict, container: str,
              limits: dict, should_stop) -> tuple[list[str], str]:
    """Talk to the model until it says TASK_COMPLETE or a limit is hit. Returns (transcript, stop reason)."""
    system_prompt = (settings["system_prompt"] + "\n\n" if settings["system_prompt"] else "") + AGENT_INSTRUCTIONS
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": task_text(item)}]
    transcript = []
    deadline = time.time() + item.get("agent_timeout_sec", limits["agent_task_timeout_sec"])
    shell_setup = SWE_SHELL_SETUP if item["agent_type"] == "swe" else ""

    for step in range(1, limits["agent_max_steps"] + 1):
        if should_stop():
            return transcript, "cancelled"
        if time.time() > deadline:
            return transcript, "agent time limit reached"
        try:
            reply = client.chat.completions.create(
                model=model_name,
                messages=recent_history(messages, limits["context_tokens"], limits["agent_step_max_tokens"]),
                temperature=settings["temperature"], top_p=settings.get("top_p", 1.0),
                presence_penalty=settings.get("presence_penalty", 0.0),
                max_tokens=limits["agent_step_max_tokens"],
                extra_body={"top_k": settings.get("top_k", -1)},
            ).choices[0].message.content or ""
        except Exception as error:  # e.g. the conversation no longer fits in --max-model-len
            return transcript, f"model request failed: {error}"
        # Only the final text goes back into the history, not the thinking, to save context space
        messages.append({"role": "assistant", "content": reply})
        command = last_bash_block(reply)

        if command is None:
            if COMPLETE_WORD in reply:
                transcript.append(f"--- step {step}: model says {COMPLETE_WORD}")
                return transcript, "model finished"
            transcript.append(f"--- step {step}: no command in reply:\n{reply[-500:]}")
            messages.append({"role": "user", "content": f"Reply with one ```bash block, or only {COMPLETE_WORD} when done."})
            continue

        exit_code, output = run_in_container(container, shell_setup + command, limits["agent_command_timeout_sec"])
        shown_output = shorten_output(output, limits["agent_output_chars"])
        transcript.append(f"--- step {step}: $ {command}\n[exit {exit_code}]\n{shown_output[-1500:]}")
        messages.append({"role": "user", "content": f"[exit code {exit_code}]\n{shown_output}"})

    return transcript, "step limit reached"


# ---------- grading ----------

def parse_test_log(parser_name: str, log: str) -> dict[str, str]:
    """Test name -> status from the test runner's output.

    ponytail: covers the three output styles used by SWE-bench Verified (pytest -rA, django, sympy);
    edge cases the official swebench parsers handle (e.g. django docstrings split over lines) may be missed.
    """
    statuses = {}
    last_django_test = None
    # Some images make pytest print in color ("\x1b[32mPASSED\x1b[0m"); remove the codes first,
    # as the official parsers do. Without this, correct sklearn/astropy fixes were graded as failed.
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    for line in log.splitlines():
        line = line.strip()
        if parser_name == "parse_log_django":
            if " ... " in line:
                last_django_test = line.split(" ... ")[0]
            match = re.match(r"^(.+?) \.\.\.\s+(ok|OK|FAIL|ERROR|skipped.*|expected failure)$", line)
            if match:
                word = match.group(2)
                status = {"ok": "PASSED", "OK": "PASSED", "expected failure": "XFAIL"}.get(word, word.split()[0].upper())
                statuses[match.group(1)] = status
            elif line == "ok" and last_django_test:
                # Some tests print extra lines between "name ..." and "ok" (same rule as the official parser)
                statuses[last_django_test] = "PASSED"
        elif parser_name == "parse_log_sympy":
            match = re.match(r"^(test_\w+)\s+(ok|F|E|f|s)\b", line)
            if match:
                statuses[match.group(1)] = {"ok": "PASSED", "F": "FAILED", "E": "ERROR", "f": "XFAIL", "s": "SKIPPED"}[match.group(2)]
        else:  # pytest -rA style: "PASSED tests/x.py::test_a" or "FAILED tests/x.py::test_a - message"
            # Same rule as the official harness: the test name is the 2nd whitespace-separated word.
            # SWE-bench's FAIL_TO_PASS / PASS_TO_PASS names were produced this way, so a name with
            # spaces (e.g. "test_x[text-file.py:4:4: W0612: ...]") is cut at its first space.
            words = line.replace(" - ", " ").split() if line.startswith("FAILED") else line.split()
            if len(words) > 1 and words[0] in ("PASSED", "FAILED", "ERROR", "XFAIL", "SKIPPED", "XPASS"):
                statuses[words[1]] = words[0]
    return statuses


def read_model_patch(container: str) -> str:
    """The agent's changes as a git patch. `git add -A` so new files the agent created are included."""
    _, patch_text = run_in_container(container, "cd /testbed && git add -A && git -c core.fileMode=false diff --cached", 120)
    return patch_text


def apply_patch_in_fresh_container(item: dict, patch_text: str) -> tuple[str, str]:
    """Start a clean container from the task image and apply the patch there, like the official harness.

    Grading in the agent's own container could count packages it installed or files it changed outside git.
    Returns (container name, problem text or "" if the patch applied).
    """
    container = start_container(item, pull=False)  # same image, already on disk
    if not patch_text.strip():
        return container, ""  # no changes: run the tests on the untouched repo
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as patch_file:
        patch_file.write(patch_text if patch_text.endswith("\n") else patch_text + "\n")
    try:
        # docker cp, not stdin: with sudo, stdin carries the password
        copied = run_docker(["cp", patch_file.name, f"{container}:/tmp/model.patch"], 120)
    finally:
        os.unlink(patch_file.name)
    if copied.returncode != 0:
        return container, f"could not copy patch: {copied.stderr[-200:]}"
    exit_code, output = run_in_container(container, "cd /testbed && git apply --whitespace=nowarn /tmp/model.patch", 120)
    return container, "" if exit_code == 0 else f"patch did not apply: {output[-300:]}"


def grade_swe_bench(item: dict, container: str, limits: dict) -> tuple[dict, str]:
    """Run the official eval script on the patch in a fresh container; check FAIL_TO_PASS and PASS_TO_PASS."""
    model_patch = read_model_patch(container)
    fresh_container, problem = apply_patch_in_fresh_container(item, model_patch)
    try:
        if problem:
            return {"verdict": "incorrect", "reason": problem}, model_patch
        exit_code, log = run_in_container(fresh_container, item["eval_script"], limits["agent_verifier_timeout_sec"])
    finally:
        run_docker(["rm", "-f", fresh_container], 120)
    # Only the part between the markers is test output (the script also prints git diff etc.)
    test_output = log.split(">>>>> Start Test Output")[-1].split(">>>>> End Test Output")[0]
    statuses = parse_test_log(item["log_parser"], test_output)

    def passed(test_name: str) -> bool:
        return statuses.get(test_name) in PASSING_STATUSES

    fixed = sum(passed(name) for name in item["fail_to_pass"])
    kept = sum(passed(name) for name in item["pass_to_pass"])
    resolved = fixed == len(item["fail_to_pass"]) and kept == len(item["pass_to_pass"])
    reason = (f"{'resolved' if resolved else 'not resolved'}: FAIL_TO_PASS {fixed}/{len(item['fail_to_pass'])}, "
              f"PASS_TO_PASS {kept}/{len(item['pass_to_pass'])}")
    if not statuses:
        reason += f" (no test results found; eval exit code {exit_code})"
    return {"verdict": "correct" if resolved else "incorrect", "reason": reason}, model_patch


def grade_terminal_bench(item: dict, container: str) -> tuple[dict, str]:
    """Copy the task's tests into the container, run test.sh, read the reward file (1 = pass)."""
    run_in_container(container, "rm -rf /tests && mkdir -p /logs/verifier", 60)
    copied = run_docker(["cp", str(Path(item["task_dir"]) / "tests"), f"{container}:/tests"], 300)
    if copied.returncode != 0:
        return {"verdict": "error", "reason": f"could not copy tests: {copied.stderr[-200:]}"}, ""
    # test.sh expects to start in the image's working directory, which `docker exec` uses by default
    exit_code, log = run_in_container(container, "bash /tests/test.sh", item.get("verifier_timeout_sec", 900))
    _, reward = run_in_container(container, "cat /logs/verifier/reward.txt", 30)
    passed = reward.strip() == "1"
    reason = f"tests {'passed' if passed else 'failed'} (reward {reward.strip() or 'missing'}, exit {exit_code})"
    return {"verdict": "correct" if passed else "incorrect", "reason": reason}, log[-3000:]


# ---------- one task, start to finish ----------

def run_agent_task(client: OpenAI, model_name: str, item: dict, settings: dict, limits: dict, should_stop) -> dict:
    started = time.time()
    result = {key: item.get(key) for key in ("id", "dataset", "category", "grader", "question", "ground_truth")}
    result.update(answer="", reasoning="", tool_calls=[])
    container = None
    try:
        container = start_container(item)
        transcript, stop_reason = run_agent(client, model_name, item, settings, container, limits, should_stop)
        if item["agent_type"] == "swe":
            grade, final_output = grade_swe_bench(item, container, limits)
        else:
            grade, final_output = grade_terminal_bench(item, container)
        grade["reason"] = f"{grade['reason']}; agent stopped: {stop_reason}, {len(transcript)} steps"
        result["answer"] = final_output  # SWE: the model's git diff; Terminal-Bench: test output
        result["reasoning"] = "\n\n".join(transcript)  # every command and its output
    except Exception as error:  # docker/pull problems: mark this task as error and keep the run going
        grade = {"verdict": "error", "reason": f"agent task failed: {error}"}
    finally:
        try:
            remove_container_and_image(container, item["image"])
        except Exception:
            pass  # cleanup must never hide the real result; worst case one image stays on disk
    result.update(grade)
    result["seconds"] = round(time.time() - started, 2)
    return result
