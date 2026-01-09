export CUDA_VISIBLE_DEVICES=0,1
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION

set -e
set -o pipefail
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

### File related hparams
LLM_DIR=llms
DATA_DIR=data
SAVE_DIR=checkpoints
VAL_DIR=evaluation


# Define an array of model names
ckpts=(
  llms/qwen2-text-1.5b
)

# hi

# List of test datasets
datasets=(
  "noformat-gsm_symbolic"
  "noformat-math"
  "noformat-openr1"
  "noformat-aime"
  "noformat-gsm8k"
)

max_length=4096



parse_model_alias() {
  local ckpt_path="$1"
  local rel_path model identifier step_info remaining

  # Finetuned pattern: $SAVE_DIR/model/identifier/step_info/remaining
  if [[ "$ckpt_path" == "$SAVE_DIR/"* ]]; then
    rel_path="${ckpt_path#$SAVE_DIR/}"
    IFS='/' read -r model identifier step_info remaining <<< "$rel_path"

    if [[ -n "$model" && -n "$identifier" && -n "$step_info" ]]; then
      # Include step information in the alias to avoid collisions
      echo "${model}/${identifier}/${step_info}"
      return 0
    elif [[ -n "$model" && -n "$identifier" ]]; then
      # Fallback for cases without step info
      echo "${model}/${identifier}"
      return 0
    fi
  fi

  # Base model pattern: $LLM_DIR/model
  if [[ "$ckpt_path" == "$LLM_DIR/"* ]]; then
    rel_path="${ckpt_path#$LLM_DIR/}"
    IFS='/' read -r model _ <<< "$rel_path"

    if [[ -n "$model" ]]; then
      echo "${model}/orig"
      return 0
    fi
  fi
}


for ckpt in "${ckpts[@]}"; do
  if alias=$(parse_model_alias "$ckpt"); then

    for test_ds in "${datasets[@]}"; do

      out_dir="$VAL_DIR/$alias/${test_ds}"
      mkdir -p "$out_dir"
      echo "=== Evaluating $alias on $test_ds ==="

        python verl/eval/inference.py \
            --model-path $ckpt \
            --input-file $DATA_DIR/${test_ds}/test.jsonl \
            --output-dir $out_dir \
            --max-new-token ${max_length} \
            --num-gpus-per-model $N_GPUS \
            --query-key "question" \
            --answer-key "answer" \
            --skip


    done
  else
    echo "Skipping invalid checkpoint: $ckpt" >&2
  fi
done
