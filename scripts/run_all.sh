#!/bin/bash
set -e  # Exit immediately on error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Output path
CSV="exp/C_ddsp/ckps/full_eval_all_C3.csv"

# Back up the previous table if it exists
if [ -f "$CSV" ]; then
  mv "$CSV" "${CSV%.csv}_backup_$(date +%Y%m%d_%H%M%S).csv"
  echo "[INFO] Previous CSV has been backed up."
fi

# Experiment list
declare -a MODES=("b1" "b2" "main" "no_pde" "no_probe" "no_treg")
declare -a RIDS=("ddsp_only_v1" "pinn_only_v1" "joint_v1" "abl_nopde_v1" "abl_noprobe_v1" "abl_notreg_v1")

# Run experiments sequentially
for i in "${!MODES[@]}"; do
  MODE="${MODES[$i]}"
  RID="${RIDS[$i]}"
  echo "========== Running MODE=$MODE RID=$RID =========="
  MODE=$MODE RID=$RID python exp/A_static/run_train.py
done

# Final summary
python tools/summarize_c3.py
