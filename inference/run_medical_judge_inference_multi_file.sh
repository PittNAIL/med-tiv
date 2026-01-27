#!/bin/bash
# Complete workflow for running medical judge inference with dense retrieval
# Place this file in: your-project/inference/run_inference.sh
# Run from: your-project/inference/

eval "$(conda shell.bash hook)"
conda activate search-retriever
# module load gcc/13.3.1-p20240614

echo "Cleaning up any existing processes..."
lsof -ti:5000 | xargs -r kill -9 2>/dev/null || true
lsof -ti:8000 | xargs -r kill -9 2>/dev/null || true
lsof -ti:30150 | xargs -r kill -9 2>/dev/null || true
pkill -9 -f "retrieval_server.py" 2>/dev/null || true
pkill -9 -f "medical_dense_retrieval_tool.py" 2>/dev/null || true
pkill -9 -f "eval_service/app.py" 2>/dev/null || true
sleep 2
echo "✓ Cleanup complete"

echo "=========================================="
echo "Medical Judge Inference with Dense Retrieval"
echo "=========================================="

# ============================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================
export no_proxy="localhost,127.0.0.1,0.0.0.0,$(hostname -i)"

# Project root (parent directory of inference/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Retrieval Server Configuration
# Path to FAISS index built with build_index.py
# Example: './data/med_tiv/retriever_index/medcpt_Flat.index'
RETRIEVER_INDEX_PATH=''

# Path to corpus JSONL file used to build the index
# Example: './data/med_tiv/retriever_index/medical_combined.jsonl'
RETRIEVER_CORPUS_PATH=''

# Path to MedCPT Query Encoder model
# Download from: https://huggingface.co/ncbi/MedCPT-Query-Encoder
RETRIEVER_MODEL_PATH=''

RETRIEVER_NAME="medcpt"  # bm25 or e5 or medcpt
RETRIEVER_PORT=8000
RETRIEVER_TOPK=3

# Tool Server Configuration
TOOL_SERVER_HOST="0.0.0.0"
TOOL_SERVER_PORT=30150
TOOL_SERVER_URL="http://${TOOL_SERVER_HOST}:${TOOL_SERVER_PORT}/get_observation"

# API Service Configuration
# Path to trained Med-TIV verifier checkpoint
# Example: './checkpoints/med_tiv/iter_2/actor/huggingface'
MODEL_PATH=''

API_HOST="0.0.0.0"
API_PORT=5000
MAX_TURNS=5
MIN_TURNS=0
TENSOR_PARALLEL_SIZE=4
NUM_MODELS=1  # Number of model instances (should equal number of GPUs)
ENABLE_MTRL=False

# Action stop tokens - model outputs </search> to trigger retrieval
ACTION_STOP_TOKENS="</search>"

# Inference Configuration
MAX_TOKENS=16384
MAX_CONCURRENT=32  # Concurrent requests (recommended: num_models * 4)

# Input files: JSON files containing reasoning traces to evaluate
# Each file should have format: [{"question": ..., "trace": ..., "label": ...}, ...]
# Example: './data/eval/medqa_traces.json'
INPUT_FILES=(
    ''  # MedQA traces
    ''  # MedMCQA traces
    ''  # MedXpertQA traces
    ''  # MMLU-Med traces
)

# Output files: Where to save evaluation results
# Example: './results/medqa_evaluations.json'
OUTPUT_FILES=(
    ''  # MedQA results
    ''  # MedMCQA results
    ''  # MedXpertQA results
    ''  # MMLU-Med results
)

# ============================================
# Verify Paths
# ============================================
echo ""
echo "Verifying paths..."

if [ ! -f "$RETRIEVER_INDEX_PATH" ]; then
    echo "ERROR: Index file not found: $RETRIEVER_INDEX_PATH"
    exit 1
fi

if [ ! -f "$RETRIEVER_CORPUS_PATH" ]; then
    echo "ERROR: Corpus file not found: $RETRIEVER_CORPUS_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: Model not found: $MODEL_PATH"
    exit 1
fi

# Verify all input files
for file in "${INPUT_FILES[@]}"; do
    if [ -n "$file" ] && [ ! -f "$file" ]; then
        echo "ERROR: Input file not found: $file"
        exit 1
    fi
done

echo "✓ All paths verified"

# ============================================
# Step 1: Start the FAISS Retrieval Server
# ============================================
echo ""
echo "Step 1: Starting FAISS retrieval server..."

python inference/retrieval_server.py \
    --index_path "$RETRIEVER_INDEX_PATH" \
    --corpus_path "$RETRIEVER_CORPUS_PATH" \
    --topk "$RETRIEVER_TOPK" \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL_PATH" \
    --faiss_gpu \
    --port "$RETRIEVER_PORT" \
    >> inference/retrieval_server.log 2>&1 &

RETRIEVER_PID=$!
echo "Retrieval server started with PID: $RETRIEVER_PID"

# Wait for retrieval server to be ready
echo "Waiting for retrieval server to initialize (this may take a while for large indices)..."
MAX_WAIT=1500  # 25 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${RETRIEVER_PORT}/docs" > /dev/null 2>&1; then
        echo "✓ Retrieval server is ready!"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "  Waiting... ${WAITED}s"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "✗ Retrieval server failed to start within ${MAX_WAIT}s"
    kill -9 $RETRIEVER_PID 2>/dev/null || true
    exit 1
fi


# ============================================
# Step 2: Start the Tool Server
# ============================================
echo ""
echo "Step 2: Starting medical dense retrieval tool server..."

python inference/medical_dense_retrieval_tool.py \
    --host "$TOOL_SERVER_HOST" \
    --port "$TOOL_SERVER_PORT" \
    --retriever_url "http://localhost:${RETRIEVER_PORT}" \
    --topk "$RETRIEVER_TOPK" \
    > inference/tool_server.log 2>&1 &

TOOL_SERVER_PID=$!
echo "Tool server started with PID: $TOOL_SERVER_PID"

# Wait for tool server
echo "Waiting for tool server to start..."
sleep 10
if curl -s "http://localhost:${TOOL_SERVER_PORT}/health" > /dev/null 2>&1; then
    echo "✓ Tool server is ready!"
else
    echo "✗ Tool server failed to start"
    kill -9 $RETRIEVER_PID 2>/dev/null || true
    kill -9 $TOOL_SERVER_PID 2>/dev/null || true
    exit 1
fi

# ============================================
# Step 3: Start the API Service with Model
# ============================================
echo ""
echo "Step 3: Starting API service with medical judge model..."

# Create temp file for action tokens
ACTION_STOP_TOKENS_FILE=$(mktemp)
echo "$ACTION_STOP_TOKENS" > $ACTION_STOP_TOKENS_FILE
echo "Action stop tokens file: $ACTION_STOP_TOKENS_FILE"

python eval_service/app.py \
    --gpu_memory_utilization 0.75 \
    --host "$API_HOST" \
    --port "$API_PORT" \
    --tool_server_url "$TOOL_SERVER_URL" \
    --model "$MODEL_PATH" \
    --max_turns "$MAX_TURNS" \
    --min_turns "$MIN_TURNS" \
    --action_stop_tokens "$ACTION_STOP_TOKENS_FILE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --num_models "$NUM_MODELS" \
    --enable_mtrl "$ENABLE_MTRL" \
    --max_obs_length 2048 \
    --max_model_len 16384 \
    > inference/api_server_debug.log 2>&1 &

API_SERVER_PID=$!
echo "API server started with PID: $API_SERVER_PID"

# Wait for API server (vLLM takes time to load)
echo "Waiting for API server to load model (this may take a few minutes)..."
MAX_WAIT=900  # 15 minutes for model loading
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${API_PORT}/health" > /dev/null 2>&1; then
        echo "✓ API server is ready!"
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    echo "  Waiting... ${WAITED}s"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "✗ API server failed to start within ${MAX_WAIT}s"
    # Cleanup
    kill -9 $RETRIEVER_PID 2>/dev/null || true
    kill -9 $TOOL_SERVER_PID 2>/dev/null || true
    kill -9 $API_SERVER_PID 2>/dev/null || true
    pkill -9 -P $API_SERVER_PID 2>/dev/null || true
    exit 1
fi


# ============================================
# Step 4: Run Inference
# ============================================
TEMPERATURE=0.7
TOP_P=0.9

echo "Waiting for API server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "✓ API server is ready!"
        break
    fi
    echo "Attempt $i/60: API server not ready yet..."
    sleep 5
done

# Additional safety: check one more time
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✗ API server health check failed after waiting"
    kill -9 $RETRIEVER_PID $TOOL_SERVER_PID $API_SERVER_PID 2>/dev/null || true
    exit 1
fi

sleep 60

# Get the length of the array
NUM_FILES=${#INPUT_FILES[@]}

for (( i=0; i<${NUM_FILES}; i++ ));
do
    CURRENT_INPUT="${INPUT_FILES[$i]}"
    CURRENT_OUTPUT="${OUTPUT_FILES[$i]}"
    
    # Skip empty entries
    if [ -z "$CURRENT_INPUT" ] || [ -z "$CURRENT_OUTPUT" ]; then
        continue
    fi
    
    echo "------------------------------------------"
    echo "Processing File [$((i+1))/$NUM_FILES]"
    echo "Input:  $CURRENT_INPUT"
    echo "Output: $CURRENT_OUTPUT"
    echo "------------------------------------------"

    # Create output directory for this specific file if it doesn't exist
    mkdir -p "$(dirname "$CURRENT_OUTPUT")"

    # Run the inference command
    python inference/medical_judge_inference.py \
        --input_file "$CURRENT_INPUT" \
        --output_file "$CURRENT_OUTPUT" \
        --api_base_url "http://localhost:${API_PORT}" \
        --model_name "$MODEL_PATH" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_concurrent "$MAX_CONCURRENT" \
        --resume \
        >> inference/inference.log 2>&1

    # Check if the specific run failed
    if [ $? -eq 0 ]; then
        echo "✓ Finished processing file $((i+1))"
    else
        echo "!! Warning: Inference failed for file $((i+1)). Continuing to next file..."
    fi
done

INFERENCE_EXIT_CODE=$?

# ============================================
# Step 5: Cleanup
# ============================================
echo ""
echo "Step 5: Cleaning up..."

# Kill all processes
echo "Stopping API server..."
kill -9 $API_SERVER_PID 2>/dev/null || true
pkill -9 -P $API_SERVER_PID 2>/dev/null || true

echo "Stopping tool server..."
kill -9 $TOOL_SERVER_PID 2>/dev/null || true

echo "Stopping retrieval server..."
kill -9 $RETRIEVER_PID 2>/dev/null || true

# Clean up temp file
rm -f "$ACTION_STOP_TOKENS_FILE"

exit $INFERENCE_EXIT_CODE
