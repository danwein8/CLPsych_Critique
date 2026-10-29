#!/usr/bin/env bash
# =============================================================================
# Cluster driver — ABCD-fidelity F1 sweep over the full Task B system set:
# 13 systems (5 new cluster reimplementations + BLUE + official baseline +
# 6 "ours" runs) + 5 degenerate baselines = 18 submissions, one NLI load.
#
# Runs pipelines/abcd_fidelity.py from the repo ROOT (it imports
# pipelines.pipeline and eval.evaluation). The NLI model is the SAME DeBERTa
# the official Consistency Score uses (already in the cluster HF cache from
# the evaluation runs), so fidelity-vs-CS differences are WHAT is measured,
# not HOW.
#
# NEW-PROMPT MAPPING for the "ours" systems (user decision 2026-10-29; see
# PROJECT_CONTEXT §0). The OLD-prompt copies in eval_systems/ are NOT used
# and must NOT be overwritten:
#   gold_structural   <- outputs/sweep_rerun/gold_structural.json        CS 0.8706
#   gold_semantic     <- outputs/sweep_rerun/gold_semantic.json          CS 0.8666
#   gold_hybrid-a0.50 <- eval_systems/gold_hybrid-a0.50.json             CS 0.8629
#                        (SHA-identical to outputs/sweep_rerun/gold_hybrid-a0.50.json)
#   random_structural <- outputs/sweep_rerun_random/random_structural.json CS 0.8721
#   gold_zeroshot     <- eval_systems/gold_zeroshot.json                CS 0.8743
#                        (2026-10-11 run = the NEW-prompt zero-shot, per
#                         run_zeroshot_label_sources.py; the 0.858 May run
#                         is old-prompt and NOT used)
#   gold_rand-retr    <- eval_systems/gold_rand-retr.json               CS 0.8676
#                        (2026-09-17 run, same updated prompt)
#   eaionflux         <- eval_systems/eaionflux.json                    CS 0.9033
#                        (SHA-identical to the cluster llama3.3:70b rerun)
#
# Files needed on the cluster (in addition to tracked pipelines/ + eval/):
#   post_data.csv                (repo root; gitignored)
#   eval_systems/                (all 13 systems; already scp'd)
#   clpsych2025/evaluation/official_run/{baseline_constant,baseline_goldsummary,
#       baseline_empty,constant-timeline,gold-timeline}.json   (already scp'd)
#   outputs/sweep_rerun/{gold_structural,gold_semantic}.json   (scp if missing)
#   outputs/sweep_rerun_random/random_structural.json          (scp if missing)
# The preflight below hard-fails and lists anything missing.
#
# Usage — run inside tmux, log to file:
#   bash pipelines/run_abcd_fidelity.sh 2>&1 | tee /users/arozovskaya/dweiner/fidelity_log.txt
# GPU override (default: GPU 4):
#   GPU=5 bash pipelines/run_abcd_fidelity.sh 2>&1 | tee fidelity_log.txt
#
# Runtime: ~120 s per system on a 4080-class card (~40 min for 18, single
# GPU, batch 16). Output -> $OUT_DIR (default: outputs/abcd_fidelity_full_<ts>/)
# with abcd_fidelity_per_summary.csv + abcd_fidelity_by_system.csv. Copy
# $OUT_DIR back to the workstation afterwards.
# =============================================================================
set -euo pipefail

# ── Cluster safeguards ────────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/users/arozovskaya/dweiner/hf_cache}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-4}"
used=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 1000 ]; then echo "GPU $CUDA_VISIBLE_DEVICES busy (${used} MiB), not starting"; exit 1; fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONIOENCODING=utf-8

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# abcd_fidelity.py imports pipelines.* and eval.* — repo root must be importable
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
POST_DATA="${POST_DATA:-$REPO_ROOT/post_data.csv}"
BATCH="${BATCH:-16}"
TAU_RECALL="${TAU_RECALL:-0.5}"
TAU_PRECISION="${TAU_PRECISION:-0.5}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/outputs/abcd_fidelity_full_$STAMP}"

# ── The 18 systems (label=path, official submission format) ──────────────────
SYSTEMS=(
  "BLUE=$REPO_ROOT/eval_systems/BLUE.json"
  "BASELINE=$REPO_ROOT/eval_systems/BASELINE.json"
  "bulusi=$REPO_ROOT/eval_systems/bulusi.json"
  "aquarius=$REPO_ROOT/eval_systems/aquarius.json"
  "ism=$REPO_ROOT/eval_systems/ism.json"
  "uottawa_s2=$REPO_ROOT/eval_systems/uottawa_s2.json"
  "eaionflux=$REPO_ROOT/eval_systems/eaionflux.json"
  "gold_structural=$REPO_ROOT/outputs/sweep_rerun/gold_structural.json"
  "gold_semantic=$REPO_ROOT/outputs/sweep_rerun/gold_semantic.json"
  "gold_hybrid-a0.50=$REPO_ROOT/eval_systems/gold_hybrid-a0.50.json"
  "random_structural=$REPO_ROOT/outputs/sweep_rerun_random/random_structural.json"
  "gold_zeroshot=$REPO_ROOT/eval_systems/gold_zeroshot.json"
  "gold_rand-retr=$REPO_ROOT/eval_systems/gold_rand-retr.json"
  "constant=$REPO_ROOT/clpsych2025/evaluation/official_run/baseline_constant.json"
  "gold-summary=$REPO_ROOT/clpsych2025/evaluation/official_run/baseline_goldsummary.json"
  "empty=$REPO_ROOT/clpsych2025/evaluation/official_run/baseline_empty.json"
  "constant_timeline=$REPO_ROOT/clpsych2025/evaluation/official_run/constant-timeline.json"
  "gold_timeline=$REPO_ROOT/clpsych2025/evaluation/official_run/gold-timeline.json"
)

# ── Preflight: hard-fail if anything is missing (a partial sweep is useless) ──
missing=()
[ -f "$POST_DATA" ] || missing+=("$POST_DATA")
for spec in "${SYSTEMS[@]}"; do
  path="${spec#*=}"
  [ -f "$path" ] || missing+=("$path")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "MISSING FILES — the sweep must run over all 18 systems + post_data.csv:"
  for m in ${missing[@]+"${missing[@]}"}; do echo "  $m"; done
  exit 1
fi

# ── Run (single NLI model load, all 18 systems) ───────────────────────────────
ARGS=()
for spec in "${SYSTEMS[@]}"; do ARGS+=(--system "$spec"); done

echo "ABCD-fidelity sweep: 18 systems -> $OUT_DIR (GPU $CUDA_VISIBLE_DEVICES, batch $BATCH)"
"$PYTHON" "$REPO_ROOT/pipelines/abcd_fidelity.py" \
  --post_data "$POST_DATA" \
  --tau_recall "$TAU_RECALL" --tau_precision "$TAU_PRECISION" \
  --batch_size "$BATCH" \
  --output_dir "$OUT_DIR" \
  "${ARGS[@]}"

# ── Post-run check block (values only; no pass/fail verdicts) ────────────────
BY_SYS="$OUT_DIR/abcd_fidelity_by_system.csv"
echo ""
echo "=== Check values (from $BY_SYS) ==="
echo "system / recall / f1:"
awk -F, '$1=="constant" || $1=="constant_timeline" || $1=="gold-summary" || $1=="empty" \
        {printf "  %-18s recall=%s f1=%s\n", $1, $2, $4}' "$BY_SYS"
echo "system / recall / precision:"
awk -F, '$1=="eaionflux" {printf "  %-18s recall=%s precision=%s\n", $1, $2, $3}' "$BY_SYS"

echo ""
echo "Done. Copy back to the workstation:"
echo "  scp -r '<cluster>:$OUT_DIR' outputs/"
echo "Then run the LOCAL specificity analysis (see PROJECT_CONTEXT section 0)."
