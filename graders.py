"""Grade model answers.

Each grade function returns a dict: {"verdict": ..., "reason": ...}
verdict is one of: "correct", "incorrect", "error", "needs_judge".
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).parent

JUDGE_SYSTEM_PROMPT = """You are a strict grader. Compare the model answer with the ground truth.
Mark "correct" only if the model answer gives the same final answer or meaning as the ground truth. Extra explanation is fine. Missing, wrong, or contradicting parts make it "incorrect".
If a context is given, the answer must be supported by the context.
Reply with JSON only: {"verdict": "correct" or "incorrect", "reason": "<one short sentence>"}"""

# Long answers are cut before sending to the judge so they fit its context window
MAX_ANSWER_CHARS_FOR_JUDGE = 6000


# ---------- math ----------

def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (handles nested braces). Falls back to the last number."""
    start = text.rfind("\\boxed{")
    if start != -1:
        depth = 0
        content_start = start + len("\\boxed{")
        for position in range(content_start, len(text)):
            if text[position] == "{":
                depth += 1
            elif text[position] == "}":
                if depth == 0:
                    return text[content_start:position]
                depth -= 1
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return numbers[-1] if numbers else None


def normalize_math(answer: str) -> str:
    answer = answer.strip().strip("$").rstrip(".")
    answer = answer.replace("\\left", "").replace("\\right", "").replace(" ", "")
    if re.fullmatch(r"-?\d[\d,]*\.?\d*", answer):
        answer = answer.replace(",", "")
        number = float(answer)
        # "0.50" and "0.5" and "2.0" and "2" should all match
        answer = str(int(number)) if number.is_integer() else str(number)
    return answer


def grade_math(answer: str, ground_truth: str) -> dict:
    extracted = extract_boxed(answer)
    if extracted is not None and normalize_math(extracted) == normalize_math(ground_truth):
        return {"verdict": "correct", "reason": f"final answer {extracted} matches"}
    # Different text can still be the same value (e.g. 0.5 vs \frac{1}{2}), so let the judge decide
    return {"verdict": "needs_judge", "reason": f"extracted {extracted!r}, not an exact match"}


# ---------- multiple choice ----------

# "Answer: C", "**Answer:** C", "The answer is: C". A lowercase letter counts only when nothing
# follows it ("Answer: d"), so "Answer: a bit unclear" is not read as option A.
ANSWER_WITH_COLON = r"(?i:answer)(?:\s+(?i:is))?[\s*_]*[:：][\s*_]*\(?([A-J]|[a-j](?=[).]?\s*$))\)?(?![A-Za-z])"
# "The answer is C" / "the answer is (C)". Uppercase only: "the answer is a ..." is just English.
ANSWER_IS = r"(?i:answer\s+is)[\s*_]*\(?([A-J])\)?(?![A-Za-z])"


def extract_choice(text: str) -> str | None:
    matches = re.findall(f"{ANSWER_WITH_COLON}|{ANSWER_IS}", text, re.MULTILINE)
    if matches:
        colon_letter, is_letter = matches[-1]
        return (colon_letter or is_letter).upper()
    # Last resort: the reply's last line is only a letter, e.g. "C" or "(C)".
    # Not "any capital letter in the text": "I" and "A" are also normal English words.
    lines = [line.strip(" *_") for line in text.strip().splitlines() if line.strip()]
    last_line = re.fullmatch(r"\(?([A-J])\)?\.?", lines[-1]) if lines else None
    return last_line.group(1) if last_line else None


def grade_choice(answer: str, ground_truth: str) -> dict:
    letter = extract_choice(answer)
    if letter == ground_truth:
        return {"verdict": "correct", "reason": f"chose {letter}"}
    return {"verdict": "incorrect", "reason": f"chose {letter}, expected {ground_truth}"}


# ---------- code (runs only inside Docker) ----------

def extract_code(text: str) -> str:
    # Only strip blank lines, not spaces: a body-only HumanEval reply must keep its indent
    python_blocks = re.findall(r"```python[^\n]*\n(.*?)```", text, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip("\n").rstrip()
    any_blocks = re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if any_blocks:
        return any_blocks[-1].strip("\n").rstrip()
    return text.strip("\n").rstrip()


def read_sudo_password() -> str | None:
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        if line.startswith("sudo_user="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


_docker_checked = False
_docker_prefix: list[str] | None = None


def docker_command() -> list[str] | None:
    """How to call docker: plain, through sudo (password from .env), or None if neither works."""
    global _docker_checked, _docker_prefix
    if _docker_checked:
        return _docker_prefix
    _docker_checked = True
    if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
        _docker_prefix = ["docker"]
    elif read_sudo_password():
        sudo_prefix = ["sudo", "-S", "-p", "", "docker"]
        check = subprocess.run(sudo_prefix + ["info"], input=read_sudo_password() + "\n",
                               capture_output=True, text=True)
        if check.returncode == 0:
            _docker_prefix = sudo_prefix
    return _docker_prefix


def run_docker(arguments: list[str], timeout_sec: float) -> subprocess.CompletedProcess:
    """Run `docker <arguments>` (plain or through sudo). Raises RuntimeError if Docker is unavailable.

    For sudo, the password goes on stdin, so docker commands here never read stdin themselves.
    """
    prefix = docker_command()
    if prefix is None:
        raise RuntimeError("docker not available: plain docker and sudo docker both failed")
    password = read_sudo_password() + "\n" if prefix[0] == "sudo" else None
    return subprocess.run(prefix + arguments, input=password, capture_output=True, text=True,
                          timeout=timeout_sec, errors="replace")


def build_program(answer: str, item: dict) -> str:
    code = extract_code(answer)
    # HumanEval: if the model sent only the function body, put it under the original signature
    if item.get("context") and not code.lstrip().startswith(("def ", "import ", "from ", "class ", "@")):
        code = item["context"] + code
    return code + "\n\n" + item["tests"]


def grade_code(answer: str, item: dict, timeout_sec: int) -> dict:
    with tempfile.TemporaryDirectory() as code_dir:
        Path(code_dir, "program.py").write_text(build_program(answer, item))
        # The program is mounted, not piped: with sudo, stdin carries the password
        arguments = [
            "run", "--rm", "--network", "none", "--memory", "512m", "--cpus", "1", "--pids-limit", "64",
            "-v", f"{code_dir}:/code:ro", "python:3.12-slim",
            "timeout", str(timeout_sec), "python", "/code/program.py",
        ]
        try:
            finished = run_docker(arguments, timeout_sec + 30)
        except RuntimeError as error:
            return {"verdict": "error", "reason": str(error)}
        except subprocess.TimeoutExpired:
            return {"verdict": "incorrect", "reason": "tests timed out"}
    if finished.returncode == 0:
        return {"verdict": "correct", "reason": "all tests passed"}
    if finished.returncode == 125:  # Docker itself failed (e.g. image missing), not the model's code
        return {"verdict": "error", "reason": "docker failed: " + finished.stderr[-300:].strip()}
    if finished.returncode == 124:  # exit code of `timeout`
        return {"verdict": "incorrect", "reason": f"tests timed out after {timeout_sec}s"}
    return {"verdict": "incorrect", "reason": finished.stderr[-300:].strip() or "tests failed"}


# ---------- tool calling ----------

def values_match(actual, allowed) -> bool:
    if actual == allowed:
        return True
    if isinstance(actual, str) and isinstance(allowed, str):
        return actual.strip().lower() == allowed.strip().lower()
    if isinstance(actual, (int, float)) and isinstance(allowed, (int, float)):
        return float(actual) == float(allowed)
    return False


def call_matches(actual_call: dict, expected_call: dict) -> str | None:
    """None if the call matches, else a short reason why not."""
    expected_name, expected_params = next(iter(expected_call.items()))
    if actual_call["name"] != expected_name:
        return f"called {actual_call['name']}, expected {expected_name}"
    arguments = actual_call["arguments"]
    for param, allowed_values in expected_params.items():
        if param not in arguments:
            # "" in the allowed list means the parameter is optional
            if "" not in allowed_values:
                return f"{expected_name}: missing {param}"
        elif not any(values_match(arguments[param], allowed) for allowed in allowed_values):
            return f"{expected_name}: {param}={arguments[param]!r}, allowed {allowed_values}"
    invented = sorted(set(arguments) - set(expected_params))
    if invented:
        return f"{expected_name}: unexpected argument(s) {invented}"
    return None


def grade_tool_call(tool_calls: list[dict], ground_truth: list[dict]) -> dict:
    if len(tool_calls) != len(ground_truth):
        return {"verdict": "incorrect", "reason": f"made {len(tool_calls)} calls, expected {len(ground_truth)}"}
    unused_calls = list(tool_calls)
    for expected_call in ground_truth:
        reasons = []
        for actual_call in unused_calls:
            reason = call_matches(actual_call, expected_call)
            if reason is None:
                unused_calls.remove(actual_call)
                break
            reasons.append(reason)
        else:
            return {"verdict": "incorrect", "reason": reasons[0] if reasons else "no matching call"}
    return {"verdict": "correct", "reason": "all calls match"}


# ---------- judge ----------

def build_judge_messages(item: dict, answer: str) -> list[dict]:
    parts = [f"Question:\n{item['question']}"]
    if item.get("context"):
        parts.append(f"Context:\n{item['context']}")
    parts.append(f"Ground truth:\n{item['ground_truth']}")
    parts.append(f"Model answer:\n{answer[-MAX_ANSWER_CHARS_FOR_JUDGE:]}")
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_judge_reply(text: str) -> dict:
    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            reply = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if reply.get("verdict") in ("correct", "incorrect"):
            return {"verdict": reply["verdict"], "reason": str(reply.get("reason", ""))}
    raise ValueError(f"judge reply has no valid verdict JSON: {text[:200]!r}")


# ---------- dispatch ----------

def grade_without_judge(item: dict, answer: str, tool_calls: list[dict], timeout_sec: int) -> dict:
    if not answer.strip() and not tool_calls:
        return {"verdict": "incorrect", "reason": "empty answer"}
    grader = item["grader"]
    if grader == "math":
        return grade_math(answer, item["ground_truth"])
    if grader == "choice":
        return grade_choice(answer, item["ground_truth"])
    if grader == "code":
        return grade_code(answer, item, timeout_sec)
    if grader == "tool_call":
        return grade_tool_call(tool_calls, item["ground_truth"])
    return {"verdict": "needs_judge", "reason": ""}
