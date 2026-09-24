#!/usr/bin/env bash
# AdaPPO on MBPP
# Usage: OPENRLHF_ROOT=/path/to/OpenRLHF bash scripts/run_mbpp_adappo.sh
# Requires openrlhf_patch/ to be copied over OpenRLHF (see README).

set -euo pipefail
cd "$(dirname "$0")/.."

NGPU=${NGPU:-4}                 # GPUs per node for actor / critic / ref
VLLM_TP=${VLLM_TP:-4}                    # vLLM tensor parallel size
VLLM_ENGINES=${VLLM_ENGINES:-1}            # with --train.colocate_all: NGPU == VLLM_ENGINES * VLLM_TP
MODEL=${MODEL:-Qwen/Qwen3-8B}
DATA=${DATA:-data/MBPP/train.arrow}
INPUT_KEY=${INPUT_KEY:-question}
LABEL_KEY=${LABEL_KEY:-answer}
OUT=${OUT:-ckpt/mbpp_adappo}

python -m openrlhf.cli.train_ppo_ray \
  --actor.model_name_or_path "$MODEL" \
  --critic.model_name_or_path "$MODEL" \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$NGPU" \
  --critic.num_nodes 1 \
  --critic.num_gpus_per_node "$NGPU" \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$NGPU" \
  --train.agent_func_path rewards/mbpp_reward.py \
  --data.prompt_dataset "$DATA" \
  --data.input_key "$INPUT_KEY" \
  --data.label_key "$LABEL_KEY" \
  --data.max_len 1024 \
  --data.max_samples 512 \
  --rollout.max_new_tokens 512 \
  --rollout.batch_size 32 \
  --rollout.n_samples_per_prompt 8 \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --train.micro_batch_size 4 \
  --train.batch_size 32 \
  --train.num_episodes 2 \
  --train.max_epochs 1 \
  --train.seed 42 \
  --actor.eps_clip 0.2 \
  --critic.value_clip 0.2 \
  --algo.advantage.estimator gae \
  --algo.advantage.lambd 0.95 \
  --algo.advantage.gamma 1.0 \
  --algo.kl.init_coef 0.3 \
  --algo.kl.horizon 10000 \
  --actor.entropy_coef 0.01 \
  --actor.adam.lr 2e-6 \
  --critic.adam.lr 9e-6 \
  --vllm.num_engines "$VLLM_ENGINES" \
  --vllm.tensor_parallel_size "$VLLM_TP" \
  --vllm.gpu_memory_utilization 0.14 \
  --vllm.sync_backend nccl \
  --vllm.enforce_eager \
  --train.colocate_all \
  --ds.enable_sleep \
  --vllm.enable_sleep \
  --actor.gradient_checkpointing_enable \
  --critic.save_value_network \
  --ds.zero_stage 3 \
  --ds.param_dtype bf16 \
  --ckpt.path "$OUT" \
  --ckpt.save_steps -1 \
  --ckpt.max_num 10 \
  --ckpt.save_every_grad_steps 64 \
  --adaptive_critic \
  --adaptive_acc_growth_threshold 0.01 \
  --adaptive_kl_threshold 0.03 \
  --adaptive_entropy_threshold 0.1 \
  --adaptive_acc_growth_window 5 \
  --adaptive_kl_ceiling 0.35 \
  --adaptive_kl_delta 0.25 \
  --adaptive_max_skip 6 \
  --adaptive_ent_floor 0.05
