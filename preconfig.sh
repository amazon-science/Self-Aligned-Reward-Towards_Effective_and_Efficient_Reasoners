# Prepare the environment
conda create -n verl python=3.10 # remove this if you already have a conda environment
source activate verl
bash scripts/install.sh
pip install -r requirements.txt
pip install -e .

# Prepare the datasets
mkdir -p data
python examples/data_preprocess/aime.py --noformat
python examples/data_preprocess/math_data.py --noformat
python examples/data_preprocess/openr1.py --noformat # This is NuminaMath in the paper
python examples/data_preprocess/gsm8k.py --noformat
python examples/data_preprocess/gsm_symbolic.py --noformat

python examples/data_preprocess/merge_datasets.py \
    --input-dirs data/math data/openr1 data/gsm8k \
    --output-dir data/math_combined_all \
    --difficulty all

# Model download - Qwen-4B-Base and Qwen-1.7B-Base for example
mkdir llms
cd llms
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B-Base
git clone https://huggingface.co/Qwen/Qwen3-1.7B-Base
# Return to main directory
cd ../
# Run sample training script
./scripts/SA-grpo-1.7B.sh
# The checkpoints from the above runs are saved under the checkpoints directory
# To evaluate use the below command - modify N_GPUs and SAVE_PATH accordingly
# (modify the save path to the required actor_hf folder within checkpoints)
N_GPUS=8
SAVE_PATH="checkpoints/Qwen3-1.7B-Base/SA-grpo-1.7B/global_step_500/actor_hf"
bash scripts/auto_validate.sh ${SAVE_PATH} ${N_GPUS}