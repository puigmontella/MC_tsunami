#!/usr/bin/env bash

NTRIALS=200000

TRANSECT_DIR="transects"
MC_SCRIPT="3_translational_mc.py"
REG_SCRIPT="4_regional_filter.py"

echo "=== Running all transects (sequential) ==="
echo "Trials per transect: $NTRIALS"
echo

for tr in ${TRANSECT_DIR}/tr_*.csv; do
  base="${tr%.csv}"
  out_mc="${base}_grilli_mcs.csv"
  out_reg="${base}_grilli_mcs_REGIONAL.csv"

  echo "------------------------------------------------------------"
  echo "Transect: $tr"

  # 1) Monte Carlo
  if [[ -f "$out_mc" ]]; then
    echo "  [skip] MC exists: $out_mc"
  else
    echo "  [run ] Monte Carlo -> $out_mc"
    python "$MC_SCRIPT" "$tr" --n "$NTRIALS" --keep_fs_checks
  fi

  # 2) Regional filter
  if [[ -f "$out_reg" ]]; then
    echo "  [skip] REG exists: $out_reg"
  else
    echo "  [run ] Regional filter -> $out_reg"
    python "$REG_SCRIPT" "$out_mc"
  fi
done

echo
echo "=== Done ==="
