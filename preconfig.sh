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