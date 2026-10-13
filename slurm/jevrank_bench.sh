#!/bin/bash
#SBATCH --job-name=jevrank
#SBATCH --gpus=h200:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=logs/jevrank_%j.out
#SBATCH --error=logs/jevrank_%j.err
#
# EVIE-4.5B + JEV-27B-VL one-shot re-rank (src/jevrank_pipeline.py) on ViDoRe v3 through vidore-benchmark's own
# CLI: English queries, all 8 datasets, the pipeline-leaderboard protocol.
#   mkdir -p logs; sbatch slurm/jevrank_bench.sh     (from the repository root)
# Set first (sbatch passes the environment on):
#   VLLM_SIF          Apptainer image of vllm/vllm-openai:v0.30.0 (used by bin/jev-serve)
#   EVIE_PYTHON       python of an env with the EVIE colpali fork's dependencies
#   EVIE_COLPALI      colpali/ directory of a github.com/Tencent/EVIE checkout
#   VIDORE_BENCH_BIN  bin/ directory of an env with vidore-benchmark (commit a70f23a) installed
#   VIDORE_BENCHMARK_DIR  checkout of vidore-benchmark, for its CI check (default external/vidore-benchmark)
#   HF_HOME           Hugging Face cache with the models (default ~/.cache/huggingface)
# Needs apptainer, python3 and curl on PATH.
# One JEV server (bin/jev-serve, started first: vLLM checks free memory) and one EVIE server share the GPU.
# Outputs:
#   runs/$EXP/submission/results/{metrics,pipeline_descriptions}/<name>/   PR-ready, checked by the repo's CI
#   runs/$EXP/decisions/<dataset>.jsonl   per query: EVIE top-16, JEV probabilities, tokens, times
#   runs/$EXP/{run_config.json, jev.log, evie.log, cli_<dataset>.log}
# Finished datasets are skipped on re-submit. The EXIT trap stops both servers, so the GPU is released at once.
set -u
JOB=${SLURM_JOB_ID:-local$$}
export TRITON_CACHE_DIR=/tmp/${USER}_triton_$JOB
ROOT=${SLURM_SUBMIT_DIR:-$PWD}
: "${EVIE_PYTHON:?}" "${EVIE_COLPALI:?}" "${VIDORE_BENCH_BIN:?}" "${VLLM_SIF:?}"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export VIDORE_BENCHMARK_DIR=${VIDORE_BENCHMARK_DIR:-$ROOT/external/vidore-benchmark}
cd "$ROOT"

DATASETS=${DATASETS:-"computer_science energy hr finance_en finance_fr industrial pharmaceuticals physics"}
NUM_CONCURRENT=${NUM_CONCURRENT:-8}   # = the JEV server's --max-num-seqs (required by the model card)
CAND_K=${CAND_K:-16}
PX=${PX:-896}
QUERY_LIMIT=${QUERY_LIMIT:-}
SMOKE=${SMOKE:-1}
JEV_MEM=${JEV_MEM:-0.55}             # of the 141 GB: weights 56 GB + KV; the rest is EVIE's (largest corpus ~22 GB)
EXP=${EXP:-jevrank_k16}
PNAME=${PNAME:-jevrank_evie-4.5b_jev-27b-vl}
mkdir -p "runs/$EXP"
RUN=$(realpath "runs/$EXP")
SUB=$RUN/submission
LOCAL=/tmp/${USER}_jevrank_$JOB
VB=$VIDORE_BENCH_BIN
mkdir -p "$RUN/decisions" "$SUB/results/metrics/$PNAME" "$LOCAL"
echo "Start: $(date) host: $(hostname) job: $JOB datasets: $DATASETS k: $CAND_K px: $PX concurrent: $NUM_CONCURRENT ${QUERY_LIMIT:+query_limit: $QUERY_LIMIT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

JPID=""; EPID=""
cleanup() {
    for p in $EPID $JPID; do kill "$p" 2>/dev/null; done
    sleep 5
    for p in $EPID $JPID; do kill -9 "$p" 2>/dev/null; done
    rm -rf "$LOCAL" "/tmp/${USER}_jev_$JOB" "$TRITON_CACHE_DIR"
    echo "End Time: $(date)"
}
trap cleanup EXIT
free_port() { python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1])"; }
VLLM_API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(24))")
export VLLM_API_KEY

# ---------- 1. JEV ----------
JPORT=$(free_port)
MAX_MODEL_LEN=32768 bin/jev-serve "$JPORT" --gpu-memory-utilization "$JEV_MEM" --limit-mm-per-prompt "{\"image\": $CAND_K}" > "$RUN/jev.log" 2>&1 &
JPID=$!
for i in $(seq 1 360); do
    curl -sf -m 5 -H @<(printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY") "http://127.0.0.1:$JPORT/v1/models" >/dev/null && break
    kill -0 $JPID 2>/dev/null || { echo "JEV server exited"; tail -40 "$RUN/jev.log"; exit 1; }
    sleep 10
done
curl -sf -m 5 -H @<(printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY") "http://127.0.0.1:$JPORT/v1/models" >/dev/null || { echo "JEV not ready after 60 min"; exit 1; }
echo "JEV ready on $JPORT after $((i * 10))s"
grep -hE "GPU KV cache size|Maximum concurrency" "$RUN/jev.log" | tail -2

# ---------- 2. EVIE retrieval server (empty; the pipeline's index() fills it) ----------
PYTHONPATH=$EVIE_COLPALI HF_HUB_OFFLINE=1 \
  "$EVIE_PYTHON" -u src/vdr_server.py \
  --backend evie --domains --port-file "$LOCAL/evie.port" > "$RUN/evie.log" 2>&1 &
EPID=$!
for i in $(seq 1 90); do [ -f "$LOCAL/evie.port" ] && break; kill -0 $EPID 2>/dev/null || break; sleep 10; done
[ -f "$LOCAL/evie.port" ] || { echo "EVIE server failed"; tail -20 "$RUN/evie.log"; exit 1; }
EURL="http://127.0.0.1:$(cat "$LOCAL/evie.port")"
echo "EVIE at $EURL"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

python3 - "$RUN/run_config.json" <<PYEOF
import json, subprocess, sys, time
commit = subprocess.run(["git", "-C", "$VIDORE_BENCHMARK_DIR", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"time": time.strftime("%F %T"), "job": "$JOB", "host": "$(hostname)", "pipeline_name": "$PNAME",
           "reranker": "autotrust/JEV-27B-VL@fee8c1d", "vllm_image": "vllm/vllm-openai:v0.30.0", "jev_gpu_memory_utilization": "$JEV_MEM",
           "candidate_k": $CAND_K, "page_max_side": $PX, "num_concurrent": $NUM_CONCURRENT, "query_limit": "$QUERY_LIMIT",
           "datasets": "$DATASETS".split(), "vidore_benchmark_commit": commit}, open(sys.argv[1], "w"), indent=1)
PYEOF

# ---------- 3. the official CLI, one dataset at a time ----------
export JEVRANK_JEV_URL="http://127.0.0.1:$JPORT" JEVRANK_JEV_KEY_ENV=VLLM_API_KEY JEVRANK_RETRIEVER_URL="$EURL" JEVRANK_WORK_DIR="$LOCAL/work"
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$LOCAL/hf_datasets
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
pargs() {  # $1 = query limit (may be empty)
    python3 -c "
import json, sys
d = {'retriever_model': 'tencent/EVIE-4.5B', 'reranker_model': 'autotrust/JEV-27B-VL', 'candidate_k': $CAND_K,
     'retriever_top_k': 100, 'page_max_side': $PX, 'page_jpeg_quality': 85, 'num_concurrent': $NUM_CONCURRENT, 'max_retries': 2}
if sys.argv[1]: d['query_limit'] = int(sys.argv[1])
print(json.dumps(d))" "$1"
}
evaluate() {  # $1 = dataset, $2 = output json, $3 = query limit, $4 = log, $5 = decision-log dir
    env JEVRANK_RUN_DIR="$5" "$VB/vidore-benchmark" pipeline evaluate --dataset-name "vidore/vidore_v3_$1" --language english \
        --module-path "$ROOT/src/jevrank_pipeline.py" --class-name JevRerankPipeline \
        --pipeline-args "$(pargs "$3")" --output-file "$2" > "$4" 2>&1
}

if [ "$SMOKE" = 1 ]; then  # two HR queries through the full official path; JEV must answer both
    echo "== smoke $(date +%T)"
    evaluate hr "$RUN/smoke_vidore_v3_hr.json" 2 "$RUN/cli_smoke.log" "$RUN/smoke_decisions"
    if ! python3 -I - "$RUN/smoke_decisions/hr.jsonl" <<'PYEOF'
import json, sys
try:
    r = [json.loads(l) for l in open(sys.argv[1])]
except FileNotFoundError:
    r = []
ok = len(r) == 2 and all(x["probs"] and len(x["probs"]) == len(x["evie"]) == 16 for x in r)
print("smoke", "PASS" if ok else "FAIL", [(x["qid"], x["prompt_tokens"], x["jev_ms"], x["error"]) for x in r])
sys.exit(0 if ok else 1)
PYEOF
    then
        echo "smoke test failed; stopping before the full run"; tail -30 "$RUN/cli_smoke.log"; exit 1
    fi
fi

STATUS=0
for D in $DATASETS; do
    OUT=$SUB/results/metrics/$PNAME/vidore_v3_$D.json
    [ -f "$OUT" ] && { echo "== $D: done earlier, skipping"; continue; }
    echo "== $D $(date +%T)"
    if ! evaluate "$D" "$OUT" "$QUERY_LIMIT" "$RUN/cli_$D.log" "$RUN/decisions"; then
        echo "evaluation failed for $D:"; grep -v -i "warn" "$RUN/cli_$D.log" | tail -30; STATUS=1; break
    fi
    grep -E "ndcg_cut_10|recall_10 |search_throughput|indexed:" "$RUN/cli_$D.log" | head -6
    rm -rf "$LOCAL/work/$D"
done

# ---------- 4. PR check ----------
python3 src/finalize_submission.py "$SUB" "$PNAME" || echo "PR check not passed (expected until all 8 datasets are in)"
echo "Exit Code: $STATUS"
exit $STATUS
