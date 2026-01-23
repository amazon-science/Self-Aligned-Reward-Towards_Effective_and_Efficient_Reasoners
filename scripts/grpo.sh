export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
# export NCCL_P2P_DISABLE=1 # Uncomment this line if you encounter deadlock issues
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION
set -e
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')


### File related hparams
LLM_DIR=llms
DATA_DIR=data
SAVE_DIR=checkpoints


model_name=Qwen3-4B-Base
model_path=${LLM_DIR}/${model_name}



sampling_bsz=128
gradient_bsz=64
val_batch_size=ERR
forward_bsz=16
step=500

exp_name="test"


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files=[data/math_combined_all_all/train.parquet] \
    data.val_files=[data/math_combined_all_all/test.parquet] \
    data.train_batch_size=${sampling_bsz} \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.prompt_key="question" \
    data.val_proportion=0.1 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.model.path=${model_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.optim.total_training_steps=${step} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${gradient_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size=${forward_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${forward_bsz} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${forward_bsz} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=6 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.total_training_steps=${step} \
    trainer.project_name=math-reasoning-rl \
    trainer.experiment_name=${model_name}-${exp_name} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir=${SAVE_DIR}/${model_name}/${exp_name} \
    trainer.reward_types=["base","ppl_qa"] \
    trainer.reward_factors=[0.0,1] \
    trainer.total_epochs=100



bash scripts/auto_validate.sh ${SAVE_DIR}/${model_name}/${exp_name}/global_step_${step}/actor_hf ${N_GPUS}

