#!/bin/bash
# Download raw eval datasets to the NAS. Run as root (the NAS folder is not writable by neel):
#   sudo bash download_datasets.sh
export HF_TOKEN=$(cat /home/neel/.cache/huggingface/token)
OUT=/mnt/truenas/llm_eval_datasets/raw
mkdir -p $OUT
HF=/home/neel/.local/bin/hf
export PYTHONPATH=/home/neel/.local/lib/python3.12/site-packages
get() { name=$1; repo=$2; shift 2
  echo "=== $name ($repo)"
  $HF download "$repo" --repo-type dataset --local-dir "$OUT/$name" "$@" && echo "OK $name" || echo "FAIL $name"
}
get gsm8k openai/gsm8k
get math500 HuggingFaceH4/MATH-500
get bfcl gorilla-llm/Berkeley-Function-Calling-Leaderboard
get hotpotqa hotpotqa/hotpot_qa --include "distractor/*"
get squad_v2 rajpurkar/squad_v2
get humaneval openai/openai_humaneval
get mbpp google-research-datasets/mbpp
get mmlu_pro TIGER-Lab/MMLU-Pro
get gpqa Idavidrein/gpqa
get arc allenai/ai2_arc
get ifeval google/IFEval
get truthfulqa truthfulqa/truthful_qa
# NL2Bash is not on HuggingFace as plain files; take the two text files from GitHub
mkdir -p "$OUT/nl2bash"
for f in all.cm all.nl; do
  curl -sfL -o "$OUT/nl2bash/$f" "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/$f" && echo "OK nl2bash/$f" || echo "FAIL nl2bash/$f"
done
echo ALL_DONE
