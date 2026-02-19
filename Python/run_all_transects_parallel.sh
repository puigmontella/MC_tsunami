#!/usr/bin/env bash

NTRIALS=200000
MAX_JOBS=100 #select the number of cores available

TRANSECT_DIR="transects"
MC_SCRIPT="3_translational_mc.py"
REG_SCRIPT="4_regional_filter.py"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "=== Running all transects (parallel) ==="
echo "Trials per transect: $NTRIALS"
echo "Max concurrent jobs : $MAX_JOBS"
echo "Logs -> $LOG_DIR/"
echo

run_one_transect () {
  local tr="$1"
  local base="${tr%.csv}"
  local out_mc="${base}_grilli_mcs.csv"
  local out_reg="${base}_grilli_mcs_REGIONAL.csv"
  local name="$(basename "$base")"
  local log="${LOG_DIR}/${name}.log"

  {
    echo "------------------------------------------------------------"
    echo "Transect: $tr"

    if [[ -f "$out_mc" ]]; then
      echo "  [skip] MC exists: $out_mc"
    else
      echo "  [run ] Monte Carlo -> $out_mc"
      python "$MC_SCRIPT" "$tr" --n "$NTRIALS" --keep_fs_checks
    fi

    if [[ -f "$out_reg" ]]; then
      echo "  [skip] REG exists: $out_reg"
    else
      echo "  [run ] Regional filter -> $out_reg"
      python "$REG_SCRIPT" "$out_mc"
    fi

    echo "  [done] $tr"
  } >> "$log" 2>&1
}

# Simple job pool: start jobs, but never exceed MAX_JOBS running at once
running=0

for tr in ${TRANSECT_DIR}/tr_*.csv; do
  run_one_transect "$tr" &

  ((running+=1))

  # If too many running jobs, wait for at least one to finish
  if (( running >= MAX_JOBS )); then
    wait -n
    ((running-=1))
  fi
done

# Wait for remaining jobs
wait

echo
echo "=== Done ==="
echo "Check logs in $LOG_DIR/ (one log per transect)"
