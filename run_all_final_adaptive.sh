#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

run_plain() {
  local name="$1"; shift
  echo "=== $name ==="
  "$PYTHON" "$ROOT/run_experiment.py" "$@" 2>&1 | tee "$ROOT/adaptive_run_logs/${name}.log"
}

mkdir -p "$ROOT/adaptive_run_logs"

run_plain final_2017_reconstruction \
  --preprocessed "$ROOT/data/cic2017_preprocessed.npz" \
  --dataset CIC-IDS2017 --mode reconstruction \
  --output_dir "$ROOT/final_2017_reconstruction" \
  --hidden_dim 160 --n_layers 6 --n_harmonics 16 --n_state_harmonics 48 \
  --epochs 4000 --min_epochs 1200 --patience 700 \
  --loss_weight_lr_scale 0.1 \
  --lambda_ode 5e-5 --lambda_ic 0.1 --lambda_forcing 1e-4 \
  --lambda_smooth 1e-3 --lambda_d_smooth 5e-6 --peak_weight 5 \
  --log_every 1000

run_plain final_2017_holdout \
  --preprocessed "$ROOT/data/cic2017_preprocessed.npz" \
  --dataset CIC-IDS2017 --mode holdout \
  --output_dir "$ROOT/final_2017_holdout" \
  --hidden_dim 160 --n_layers 6 --n_harmonics 16 --n_state_harmonics 48 \
  --epochs 4000 --min_epochs 1200 --patience 700 \
  --loss_weight_lr_scale 0.1 \
  --lambda_ode 5e-5 --lambda_ic 0.1 --lambda_forcing 1e-4 \
  --lambda_smooth 1e-3 --lambda_d_smooth 5e-6 --peak_weight 5 \
  --log_every 1000

run_plain final_2018_reconstruction \
  --preprocessed "$ROOT/data/cse2018_preprocessed/cse2018_preprocessed.npz" \
  --dataset CSE-CIC-IDS2018 --mode reconstruction \
  --output_dir "$ROOT/final_2018_reconstruction" \
  --hidden_dim 160 --n_layers 6 --n_harmonics 32 --n_state_harmonics 128 \
  --epochs 5000 --min_epochs 1500 --patience 900 \
  --loss_weight_lr_scale 0.1 \
  --lambda_ode 5e-6 --lambda_ic 0.1 --lambda_forcing 5e-6 \
  --lambda_smooth 5e-5 --lambda_d_smooth 0 --peak_weight 300 \
  --log_every 1000

run_plain final_2018_holdout \
  --preprocessed "$ROOT/data/cse2018_preprocessed/cse2018_preprocessed.npz" \
  --dataset CSE-CIC-IDS2018 --mode holdout \
  --output_dir "$ROOT/final_2018_holdout" \
  --hidden_dim 160 --n_layers 6 --n_harmonics 32 --n_state_harmonics 128 \
  --epochs 5000 --min_epochs 1500 --patience 900 \
  --loss_weight_lr_scale 0.1 \
  --lambda_ode 5e-6 --lambda_ic 0.1 --lambda_forcing 5e-6 \
  --lambda_smooth 5e-5 --lambda_d_smooth 0 --peak_weight 300 \
  --log_every 1000
