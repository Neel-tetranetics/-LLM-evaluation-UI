"""Small checks for the answer graders.

Run from the project folder:
    .venv/bin/python tests/test_graders.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import parse_test_log, shorten_output
from graders import (
    build_program,
    extract_boxed,
    extract_choice,
    extract_code,
    grade_math,
    grade_tool_call,
    grade_without_judge,
    parse_judge_reply,
)


def test_boxed_with_nested_braces():
    # why: MATH answers like \frac{1}{2} have braces inside \boxed{}; a simple regex cuts them off.
    # failed when: never failed after graders.py was written.
    assert extract_boxed("so \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"


def test_math_ignores_thousands_comma():
    # why: models often write 1,000 while GSM8K ground truth is 1000.
    # failed when: never failed after graders.py was written.
    assert grade_math("The answer is \\boxed{1,000}", "1000")["verdict"] == "correct"


def test_choice_letter_from_answer_line():
    # why: the reply mentions other letters while reasoning; only the final "Answer: X" counts,
    #      and the word "a" in "the answer is a ..." must not count as option A.
    # failed when: first version of extract_choice returned "A" for "The answer is a bit tricky." (case-insensitive regex); fixed by allowing lowercase only after "Answer:".
    assert extract_choice("I think B... Answer: d") == "D"
    assert extract_choice("The answer is a bit tricky.") is None


def test_code_with_and_without_fence():
    # why: some models forget the ```python block; we still need code to run (Review Focus 2).
    # failed when: never failed after graders.py was written.
    assert extract_code("text\n```python\nx=1\n```") == "x=1"
    assert extract_code("x=1") == "x=1"


def test_tool_calls_any_order():
    # why: parallel BFCL calls may come back in any order; both orders are correct.
    # failed when: never failed after graders.py was written.
    expected = [{"spotify_play": {"artist": ["Taylor Swift"], "duration": [20]}},
                {"spotify_play": {"artist": ["Maroon 5"], "duration": [15]}}]
    actual = [{"name": "spotify_play", "arguments": {"artist": "Maroon 5", "duration": 15}},
              {"name": "spotify_play", "arguments": {"artist": "Taylor Swift", "duration": 20}}]
    assert grade_tool_call(actual, expected)["verdict"] == "correct"
    actual[0]["arguments"]["duration"] = 99
    assert grade_tool_call(actual, expected)["verdict"] == "incorrect"


def test_empty_answer_is_incorrect():
    # why: thinking models can use all max_tokens on thinking and return empty content (Review Focus 1).
    # failed when: never failed after graders.py was written.
    result = grade_without_judge({"grader": "math", "ground_truth": "3"}, "", [], 10)
    assert result == {"verdict": "incorrect", "reason": "empty answer"}


def test_judge_reply_after_thinking_text():
    # why: the judge may write <think> text before the JSON.
    # failed when: never failed after graders.py was written.
    assert parse_judge_reply('<think>x</think> {"verdict": "correct", "reason": "ok"}')["verdict"] == "correct"


def test_choice_markdown_and_answer_is_colon():
    # why: Qwen often writes "**Answer:** C" or "The correct answer is: C"; the old fallback
    #      then picked the pronoun "I" (also a valid MMLU-Pro option) from the text after it.
    # failed when: final review found both returned "I" instead of "C".
    assert extract_choice("**Answer:** C\n\nI hope this helps.") == "C"
    assert extract_choice("The correct answer is: C. I think") == "C"
    assert extract_choice("Answer: a bit unclear, maybe B") is None


def test_humaneval_body_only_keeps_indent():
    # why: a body-only reply must stay indented under the prompt's "def", or it is a SyntaxError.
    # failed when: final review found .strip() removed the first line's indent -> "'return' outside function".
    item = {"context": "def add_one(x):\n", "tests": "assert add_one(1) == 2"}
    for answer in ["    return x + 1", "```python\n    return x + 1\n```"]:
        program = build_program(answer, item)
        compile(program, "program.py", "exec")
        scope = {}
        exec(program, scope)


def test_tool_call_extra_argument_is_wrong():
    # why: BFCL marks a call wrong when the model invents a parameter; ignoring it inflated scores.
    # failed when: final review found an invented parameter still graded "correct".
    expected = [{"get_weather": {"city": ["Paris"]}}]
    actual = [{"name": "get_weather", "arguments": {"city": "Paris", "planet": "Mars"}}]
    assert grade_tool_call(actual, expected)["verdict"] == "incorrect"


def test_swe_pytest_log_names_cut_at_first_space():
    # why: SWE-bench expected test names stop at the first space; matching whole log lines made
    #      14 untouched pylint tests count as missing, so the task looked unsolvable.
    # failed when: first real agent run: pylint-7080 with NO edits scored PASS_TO_PASS 106/120.
    log = ("PASSED tests/t.py::test_a[text-file.py:4:4: W0612: Unused variable 'v']\n"
           "FAILED tests/t.py::test_b - AssertionError: nope\n")
    statuses = parse_test_log("parse_log_pytest", log)
    assert statuses["tests/t.py::test_a[text-file.py:4:4:"] == "PASSED"
    assert statuses["tests/t.py::test_b"] == "FAILED"


def test_long_output_keeps_start_and_end():
    # why: keeping only the end of long output hid file headers and first errors from the agent.
    # failed when: never failed after shorten_output was written (added with the fix).
    output = "HEAD" + "x" * 50_000 + "TAIL"
    shown = shorten_output(output, 10_000)
    assert "HEAD" in shown and "TAIL" in shown and "Output too long" in shown and len(shown) < 10_400
    assert shorten_output("short", 10_000) == "short"


def test_swe_log_with_color_codes():
    # why: some SWE-bench images print pytest results in color; the codes hid every PASSED line.
    # failed when: sklearn-26323 passed all 189 tests but was graded "no test results found".
    log = "\x1b[32mPASSED\x1b[0m sklearn/t.py::\x1b[1mtest_a[x]\x1b[0m\n"
    assert parse_test_log("parse_log_pytest", log) == {"sklearn/t.py::test_a[x]": "PASSED"}


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
    print("ALL PASSED")
