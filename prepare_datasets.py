"""Convert the raw datasets on the NAS into one simple JSONL format.

Run once (from the project folder):
    .venv/bin/python prepare_datasets.py

Every output line looks like:
    {"id", "dataset", "category", "grader", "question", "ground_truth",
     optional: "context", "choices", "tools", "tests"}
"""

import csv
import json
import random
import tomllib
from pathlib import Path

import pyarrow.parquet as pq
import yaml

LETTERS = "ABCDEFGHIJ"
CODE_REPLY_RULE = "Reply with the full code in a ```python block."


def read_parquet(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_item(dataset: str, category: str, grader: str, item_id, question: str, ground_truth, **extra) -> dict:
    item = {
        "id": f"{dataset}-{item_id}",
        "dataset": dataset,
        "category": category,
        "grader": grader,
        "question": question,
        "ground_truth": ground_truth,
    }
    item.update(extra)
    return item


def choice_question(question: str, options: list[str]) -> str:
    lines = [question, ""]
    lines += [f"{LETTERS[index]}. {option}" for index, option in enumerate(options)]
    lines.append("\nEnd your reply with 'Answer: X' where X is the letter.")
    return "\n".join(lines)


# ---------- math ----------

def convert_gsm8k(raw: Path) -> list[dict]:
    rows = read_parquet(raw / "gsm8k/main/test-00000-of-00001.parquet")
    items = []
    for index, row in enumerate(rows):
        # GSM8K solutions end with "#### <number>"
        final_answer = row["answer"].split("####")[-1].strip().replace(",", "")
        question = row["question"] + "\n\nPut your final answer in \\boxed{}."
        items.append(make_item("gsm8k", "math", "math", index, question, final_answer))
    return items


def convert_math500(raw: Path) -> list[dict]:
    items = []
    for index, row in enumerate(read_jsonl(raw / "math500/test.jsonl")):
        question = row["problem"] + "\n\nPut your final answer in \\boxed{}."
        items.append(make_item("math500", "math", "math", index, question, row["answer"]))
    return items


# ---------- multiple choice ----------

def convert_mmlu_pro(raw: Path) -> list[dict]:
    items = []
    for row in read_parquet(raw / "mmlu_pro/data/test-00000-of-00001.parquet"):
        options = list(row["options"])
        items.append(make_item("mmlu_pro", "knowledge", "choice", row["question_id"],
                               choice_question(row["question"], options), row["answer"], choices=options))
    return items


def convert_gpqa_diamond(raw: Path) -> list[dict]:
    items = []
    with open(raw / "gpqa/gpqa_diamond.csv", newline="") as file:
        for index, row in enumerate(csv.DictReader(file)):
            correct = row["Correct Answer"].strip()
            options = [correct] + [row[f"Incorrect Answer {n}"].strip() for n in (1, 2, 3)]
            # Seeded by row index so the option order never changes between runs
            random.Random(index).shuffle(options)
            answer_letter = LETTERS[options.index(correct)]
            items.append(make_item("gpqa_diamond", "knowledge", "choice", index,
                                   choice_question(row["Question"].strip(), options), answer_letter, choices=options))
    return items


def convert_arc_challenge(raw: Path) -> list[dict]:
    items = []
    for row in read_parquet(raw / "arc/ARC-Challenge/test-00000-of-00001.parquet"):
        options = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        # Some ARC rows use labels "1","2","3"; we always show A, B, C...
        answer_letter = LETTERS[labels.index(row["answerKey"])]
        items.append(make_item("arc_challenge", "reasoning", "choice", row["id"],
                               choice_question(row["question"], options), answer_letter, choices=options))
    return items


# ---------- coding ----------

def convert_humaneval(raw: Path) -> list[dict]:
    items = []
    for row in read_parquet(raw / "humaneval/openai_humaneval/test-00000-of-00001.parquet"):
        question = "Complete this Python function. " + CODE_REPLY_RULE + "\n\n" + row["prompt"]
        tests = row["test"] + f"\ncheck({row['entry_point']})\n"
        item_id = row["task_id"].split("/")[-1]
        # context = original prompt, used when the model replies with only the function body
        items.append(make_item("humaneval", "coding", "code", item_id, question, row["canonical_solution"],
                               tests=tests, context=row["prompt"]))
    return items


def convert_mbpp(raw: Path) -> list[dict]:
    items = []
    for row in read_parquet(raw / "mbpp/sanitized/test-00000-of-00001.parquet"):
        test_list = list(row["test_list"])
        question = row["prompt"] + "\nYour code must pass this test:\n" + test_list[0] + "\n" + CODE_REPLY_RULE
        tests = "\n".join(list(row["test_imports"]) + test_list) + "\n"
        items.append(make_item("mbpp", "coding", "code", row["task_id"], question, row["code"], tests=tests))
    return items


# ---------- tool calling ----------

def fix_bfcl_schema(value):
    """BFCL uses type "dict"; OpenAI-style tools need "object"."""
    if isinstance(value, dict):
        fixed = {key: fix_bfcl_schema(inner) for key, inner in value.items()}
        if fixed.get("type") == "dict":
            fixed["type"] = "object"
        return fixed
    if isinstance(value, list):
        return [fix_bfcl_schema(inner) for inner in value]
    return value


def safe_tool_name(name: str) -> str:
    # Tool names with "." are rejected by many chat templates
    return name.replace(".", "_")


def convert_bfcl(raw: Path) -> list[dict]:
    items = []
    for category in ["simple", "multiple", "parallel", "parallel_multiple"]:
        questions = read_jsonl(raw / f"bfcl/BFCL_v3_{category}.json")
        answers = {row["id"]: row["ground_truth"] for row in read_jsonl(raw / f"bfcl/possible_answer/BFCL_v3_{category}.json")}
        for row in questions:
            first_turn = row["question"][0]
            user_message = next(message["content"] for message in first_turn if message["role"] == "user")
            tools = [
                {"type": "function", "function": {
                    "name": safe_tool_name(function["name"]),
                    "description": function.get("description", ""),
                    "parameters": fix_bfcl_schema(function["parameters"]),
                }}
                for function in row["function"]
            ]
            expected_calls = [
                {safe_tool_name(name): params for name, params in call.items()}
                for call in answers[row["id"]]
            ]
            items.append(make_item("bfcl", "tool_calling", "tool_call", row["id"], user_message, expected_calls, tools=tools))
    return items


# ---------- judged by the judge model ----------

def convert_hotpotqa(raw: Path) -> list[dict]:
    items = []
    for row in read_parquet(raw / "hotpotqa/distractor/validation-00000-of-00001.parquet"):
        passages = [
            title + "\n" + "".join(sentences)
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"])
        ]
        items.append(make_item("hotpotqa", "rag", "judge", row["id"], row["question"], row["answer"],
                               context="\n\n".join(passages)))
    return items


def convert_squad_v2(raw: Path) -> list[dict]:
    items = []
    for row in read_parquet(raw / "squad_v2/squad_v2/validation-00000-of-00001.parquet"):
        answers = row["answers"]["text"]
        ground_truth = answers[0] if len(answers) > 0 else "The question cannot be answered from the context."
        items.append(make_item("squad_v2", "rag", "judge", row["id"], row["question"], ground_truth,
                               context=row["context"]))
    return items


def convert_truthfulqa(raw: Path) -> list[dict]:
    items = []
    for index, row in enumerate(read_parquet(raw / "truthfulqa/generation/validation-00000-of-00001.parquet")):
        ground_truth = (
            f"Best answer: {row['best_answer']}\n"
            f"Also correct: {'; '.join(row['correct_answers'])}\n"
            f"Incorrect (must not say): {'; '.join(row['incorrect_answers'])}"
        )
        items.append(make_item("truthfulqa", "truthfulness", "judge", index, row["question"], ground_truth))
    return items


def convert_ifeval(raw: Path) -> list[dict]:
    items = []
    for row in read_jsonl(raw / "ifeval/ifeval_input_data.jsonl"):
        rules = []
        for instruction, kwargs in zip(row["instruction_id_list"], row["kwargs"]):
            settings = {key: value for key, value in (kwargs or {}).items() if value is not None}
            rules.append(f"{instruction} {settings}" if settings else instruction)
        ground_truth = "The reply must follow every instruction in the prompt: " + "; ".join(rules)
        items.append(make_item("ifeval", "instruction_following", "judge", row["key"], row["prompt"], ground_truth))
    return items


def convert_nl2bash(raw: Path) -> list[dict]:
    descriptions = (raw / "nl2bash/all.nl").read_text().splitlines()
    commands = (raw / "nl2bash/all.cm").read_text().splitlines()
    items = []
    for index, (description, command) in enumerate(zip(descriptions, commands)):
        question = "Write one bash command for this task. Reply with only the command.\n\n" + description
        items.append(make_item("nl2bash", "terminal", "judge", index, question, command))
    return items


# ---------- agent benchmarks (run inside Docker, graded by the benchmark's own tests) ----------

def convert_swe_bench_verified(raw: Path) -> list[dict]:
    items = []
    for row in read_parquet(raw / "swe_bench_verified/data/test-00000-of-00001.parquet"):
        items.append(make_item(
            "swe_bench_verified", "agent", "agent", row["instance_id"], row["problem_statement"], row["patch"],
            agent_type="swe", image=row["image"], eval_script=row["eval_script"], log_parser=row["log_parser"],
            fail_to_pass=list(row["FAIL_TO_PASS"]), pass_to_pass=list(row["PASS_TO_PASS"]),
        ))
    return items


def convert_terminal_bench_2(raw: Path) -> list[dict]:
    items = []
    for task_dir in sorted(path for path in (raw / "terminal_bench_2").iterdir() if (path / "task.toml").exists()):
        task = tomllib.loads((task_dir / "task.toml").read_text())
        environment = task.get("environment", {})
        items.append(make_item(
            "terminal_bench_2", "agent", "agent", task_dir.name, (task_dir / "instruction.md").read_text(),
            "The task's tests/test.sh must pass.",
            agent_type="terminal", image=environment["docker_image"], task_dir=str(task_dir),
            memory=environment.get("memory", "2G"), cpus=environment.get("cpus", 1),
            agent_timeout_sec=task.get("agent", {}).get("timeout_sec", 900),
            verifier_timeout_sec=task.get("verifier", {}).get("timeout_sec", 900),
        ))
    return items


CONVERTERS = {
    "gsm8k": convert_gsm8k,
    "math500": convert_math500,
    "bfcl": convert_bfcl,
    "hotpotqa": convert_hotpotqa,
    "squad_v2": convert_squad_v2,
    "humaneval": convert_humaneval,
    "mbpp": convert_mbpp,
    "mmlu_pro": convert_mmlu_pro,
    "gpqa_diamond": convert_gpqa_diamond,
    "arc_challenge": convert_arc_challenge,
    "ifeval": convert_ifeval,
    "truthfulqa": convert_truthfulqa,
    "nl2bash": convert_nl2bash,
    "swe_bench_verified": convert_swe_bench_verified,
    "terminal_bench_2": convert_terminal_bench_2,
}


def main() -> None:
    config = yaml.safe_load(open(Path(__file__).parent / "config.yml"))
    raw_dir = Path(config["paths"]["raw_datasets"])
    output_dir = Path(__file__).parent / config["paths"]["data"] / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, convert in CONVERTERS.items():
        items = convert(raw_dir)
        write_jsonl(output_dir / f"{name}.jsonl", items)
        print(name, len(items))


if __name__ == "__main__":
    main()
