#!/bin/bash
# Re-benchmark GD + Jacobi on the amortization-retargeted suitesparse-iter values
# (VBR+CSR SABLE configs + fully-sparse CSR baselines), in cost tiers.
#
#   bash run_amortized_bench.sh            # quick tier only (~4 h)
#   bash run_amortized_bench.sh medium     # quick + medium (~+12 h)
#   bash run_amortized_bench.sh heavy      # everything (WEEKS - see below)
#
# All tiers write/merge into results/iterative_eval_amortized.json; completed
# configurations are skipped on re-invocation, so the script is safe to
# interrupt and re-run. Solves are pinned to core 0 by bench_iterative.py
# (unless under SLURM). Run inside tmux/nohup.
set -euo pipefail
cd "$(dirname "$0")"

OUT=results/iterative_eval_amortized.json
LOG=results/iterative_eval_amortized.log
TIER="${1:-quick}"

run() {
    python3 bench_iterative.py "$@" \
        --compositions vbr_csr,csr --output "$OUT" 2>&1 | tee -a "$LOG"
}

# Quick tier: solves of 0.6 s - 2 min. Full sampling (bench 5). ~4 h total
# including the one-off scipy reference solves.
run eris1176 heart2 heart3 heart1 c-30 exdata_1 --bench 5

if [[ "$TIER" == "medium" || "$TIER" == "heavy" ]]; then
    # Medium tier: solves of 2.5-20 min. bench 2 keeps it ~12 h; per-solve
    # times are stable anyway (each averages over >= 2e5 iterations).
    run TSC_OPF_1047 TSC_OPF_300 bundle1 c-45 --bench 2
fi

if [[ "$TIER" == "heavy" ]]; then
    # Heavy tier: solves of 3-11 HOURS each. Even at bench 1 this is
    # ~2.5-3 WEEKS single-core (6 configs per matrix + scipy references of
    # comparable length). Run selectively, e.g. one matrix at a time:
    #   bash run_amortized_bench.sh heavy  # or edit the list below
    run c-57 gupta1 net100 net125 net150 --bench 1
fi

echo "DONE tier=$TIER -> $OUT"
