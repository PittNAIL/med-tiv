#!/bin/bash

# Search-R1 style training with verl-tool
# This script demonstrates how to train an LLM to use search capabilities
eval "$(conda shell.bash hook)"
conda activate verl-tool

# Path to base model or checkpoint from previous iteration
# Example: 'Qwen/Qwen2.5-7B-Instruct' or './checkpoints/iter1/actor/huggingface'
model_name=''

# Path to training data in parquet format with (question, trace, label) tuples
# Example: './data/med_tiv/train.parquet'
train_data=''

# Path to validation data in the same format
# Example: './data/med_tiv/test.parquet'
val_data=''

unset ROCR_VISIBLE_DEVICES
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Weights & Biases API key for experiment tracking (optional)
export WANDB_API_KEY=''
export no_proxy="localhost,127.0.0.1,0.0.0.0,$(hostname -i)"

rl_alg=grpo
n_gpus_per_node=4
n_nodes=1
n=8
total_epochs=5
batch_size=256
ppo_mini_batch_size=256
max_prompt_length=2048
max_action_length=1024
max_response_length=8192
max_obs_length=2048
temperature=1.0
top_p=1.0
enable_agent=True
strategy="fsdp"
action_stop_tokens="</search>,</answer>"
max_turns=4
kl_loss_coef=0.0001
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6
reward_manager=search_r1_qa_em
ppo_micro_batch_size_per_gpu=16
log_prob_micro_batch_size_per_gpu=16
tensor_model_parallel_size=4
gpu_memory_utilization=0.65
do_offload=True
use_dynamic_bsz=True
ulysses_sequence_parallel_size=1
fsdp_size=-1
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=False
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="local_retrieval"  # Changed from "debug"
if [ "$enable_agent" = "True" ]; then
    run_name="${reward_manager}-${strategy}-agent-qwen_2nd-${rl_alg}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}-${run_name_postfix}-medcritic-medcpt"
else
    run_name="${reward_manager}-${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}-${run_name_postfix}"
fi
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN  # Changed from INFO to reduce log spam
export VLLM_USE_V1=1
rollout_mode='async'

# temp file for action tokens as verl cannot pass special strs as params
action_stop_tokens_file="$(pwd)$(mktemp)"
mkdir -p $(dirname $action_stop_tokens_file)
echo -e -n "$action_stop_tokens" | tee $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

# ============================================================================
# LOCAL RETRIEVAL SETUP WITH LOGGING
# ============================================================================

echo ""
echo "================================================================"
echo "Starting Multi-GPU Retrieval Server (Single Endpoint)"
echo "================================================================"
echo ""

# Retrieval configuration
file_path=./data/med_tiv/retriever_index
index_file=$file_path/medcpt_Flat.index
corpus_file=$file_path/medical_combined.jsonl
retriever_name=medcpt

# Path to MedCPT Query Encoder for dense retrieval
# Download from: https://huggingface.co/ncbi/MedCPT-Query-Encoder
retriever_path=''

port=8000

echo "Starting retrieval server using GPUs 0,1,2,3..."

# Launch retrieval server with multi-GPU support
# The server will internally distribute the index across GPUs
# Note: Your retrieval server needs to support multi-GPU mode
python -u \
    ./verl_tool/servers/tools/utils/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --port $port \
    --faiss_gpu &

retriever_pid=$!
retrieval_url="http://127.0.0.1:$port"

echo "✓ Multi-GPU retrieval server started on port $port (PID: $retriever_pid)"

MAX_WAIT=1500  # 30 minutes
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if process died
    if ! ps -p $retriever_pid > /dev/null 2>&1; then
        echo "✗ Retrieval server process died!"
        echo "Last 50 lines of log:"
        tail -50 retrieval_server.log
        exit 1
    fi
    
    # Check if server is responding
    if curl -s "http://127.0.0.1:${port}/docs" > /dev/null 2>&1; then
        echo "✓ Retrieval server is ready!"
        break
    fi
    
    sleep 15
    WAITED=$((WAITED + 15))
    
    # Show progress every minute
    if [ $((WAITED % 60)) -eq 0 ]; then
        MINUTES=$((WAITED / 60))
        echo "  Waited ${MINUTES} min / $((MAX_WAIT / 60)) min"
        echo "  Last 3 lines from log:"
        tail -3 retrieval_server.log 2>/dev/null || echo "    (no log output yet)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "✗ Retrieval server failed to start within $((MAX_WAIT / 60)) minutes"
    echo "Last 100 lines of log:"
    tail -100 retrieval_server.log
    kill -9 $retriever_pid 2>/dev/null || true
    exit 1
fi

echo ""
echo "✓ Retrieval server ready at: $retrieval_url"
echo ""

# ============================================================================
# TOOL SERVER SETUP
# ============================================================================

host=$(hostname -i | awk '{print $1}')
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation

# Export configuration for tool
export RETRIEVER_URL=$retrieval_url
export RETRIEVER_CACHE_SIZE=100000  # Large cache for single server

python -m verl_tool.servers.serve \
    --host $host \
    --port $port \
    --tool_type "search_retrieval" \
    --workers_per_tool 8 &

server_pid=$!
echo "✓ Tool server started (PID: $server_pid)"

# Wait for tool server to be ready
sleep 30

echo "Testing retrieval with correct format..."
test_response=$(curl -s --max-time 15 -X POST $tool_server_url \
    -H "Content-Type: application/json" \
    -d '{
      "trajectory_ids": ["test_trajectory"],
      "actions": ["<search>what is diabetes</search>"],
      "extra_fields": [{}]
    }')

if echo "$test_response" | grep -q '"observations"'; then
    echo "✓ Tool server responding correctly"
    # Show a snippet of the retrieved information
    echo "$test_response" | python3 -m json.tool 2>/dev/null | head -n 20
else
    echo "⚠ Response: $test_response"
fi


# ============================================================================
# START TRAINING
# ============================================================================

PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=True \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=2048 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    reward_model.reward_manager=$reward_manager \
    reward_model.launch_reward_fn_async=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.loss_agg_mode='seq-mean-token-sum-norm' \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.agent.tool_server_url=$tool_server_url \
    actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.agent.max_response_length=$max_response_length \
    actor_rollout_ref.agent.max_start_length=$max_prompt_length \
    actor_rollout_ref.agent.max_obs_length=$max_obs_length \
    actor_rollout_ref.agent.max_turns=$max_turns \
    actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
    actor_rollout_ref.agent.mask_observations=$mask_observations \
    actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
    actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
    actor_rollout_ref.agent.max_action_length=$max_action_length \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['wandb'] \
    trainer.project_name=$reward_manager \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=$total_epochs


kill -9 $retriever_pid 2>/dev/null || true
pkill -9 -P $retriever_pid 2>/dev/null || true
kill -9 $server_pid 2>/dev/null || true
pkill -9 -P $server_pid 2>/dev/null || true
