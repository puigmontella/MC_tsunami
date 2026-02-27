python Python/run_pipeline.py --config cases/config.json \
  --n-jobs 30 --n-trials 5000 --seed 1234 \
  --rp-min 100 --rp-max 750 --rp-step 10 --rp-cap 750 \
  --keep-fs-checks \
  --rps 100,200,475,750 --eta-min 0.5 \
  --map-types any --eta-pct 99
