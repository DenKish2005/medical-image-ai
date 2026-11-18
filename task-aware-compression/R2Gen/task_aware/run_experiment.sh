#!/usr/bin/env bash
# Full experiment from the technical note (conditions A, B, C, D) on IU X-Ray.
# Run from the R2Gen folder:   bash task_aware/run_experiment.sh
# Override anything via env vars, e.g.  TARGETS="1000 2000" EPOCHS_B=60 bash task_aware/run_experiment.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
R2GEN_ARGS=(--image_dir data/iu_xray/images/ --ann_path data/iu_xray/annotation.json --dataset_name iu_xray
            --load data/model_iu_xray.pth --num_workers 2 --beam_size 3 --sample_method beam_search
            --block_trigrams 1 --max_seq_length 60)
FEAT=${FEAT:-../results/ta_features}
RUNS=${RUNS:-../results/ta_runs}
EVAL=${EVAL:-../results/ta_eval}
TARGETS=${TARGETS:-"500 1000 2000 4000 8000"}   # ACTUAL coded bits per study (2 views); raw = 6,422,528 bits
C_Y=${C_Y:-128}
EPOCHS_B=${EPOCHS_B:-100}
EPOCHS_C=${EPOCHS_C:-40}
LAMBDA_F=${LAMBDA_F:-1.0}
LAMBDA_T=${LAMBDA_T:-1.0}
ALPHA=${ALPHA:-2.0}
COMMON=(--feature_dir "$FEAT" --rate_mode target --c_y "$C_Y" --batch_size 16)

# 1. cache frozen layer4 features, verify cached path reproduces R2Gen's own reports
[ -f "$FEAT/manifest.json" ] || $PY task_aware/extract_features.py --out_dir "$FEAT" --verify 16 -- "${R2GEN_ARGS[@]}"
# 2. safeguards (polarity tests, gradient, roundtrip, leakage)
$PY task_aware/check_pipeline.py --feature_dir "$FEAT" -- "${R2GEN_ARGS[@]}"

for T in $TARGETS; do
  # B: standard codec, rate + feature distortion only
  [ -f "$RUNS/B_r$T/best.pt" ] || $PY task_aware/train_codec.py --stage baseline "${COMMON[@]}" \
      --out_dir "$RUNS/B_r$T" --target_bits "$T" --epochs "$EPOCHS_B" --lr 1e-4 --lambda_f "$LAMBDA_F" -- "${R2GEN_ARGS[@]}"
  # B+: control -- baseline fine-tuned for the same extra epochs (rules out "just more training")
  [ -f "$RUNS/Bft_r$T/best.pt" ] || $PY task_aware/train_codec.py --stage baseline "${COMMON[@]}" \
      --init "$RUNS/B_r$T/best.pt" --out_dir "$RUNS/Bft_r$T" --target_bits "$T" --epochs "$EPOCHS_C" --lr 5e-5 \
      --lambda_f "$LAMBDA_F" -- "${R2GEN_ARGS[@]}"
  # C: task-aware, initialised from B, same target rate
  [ -f "$RUNS/C_r$T/best.pt" ] || $PY task_aware/train_codec.py --stage task "${COMMON[@]}" \
      --init "$RUNS/B_r$T/best.pt" --out_dir "$RUNS/C_r$T" --target_bits "$T" --epochs "$EPOCHS_C" --lr 5e-5 \
      --lambda_f "$LAMBDA_F" --lambda_t "$LAMBDA_T" --alpha "$ALPHA" -- "${R2GEN_ARGS[@]}"
  # D1: no feature loss
  [ -f "$RUNS/Cnofeat_r$T/best.pt" ] || $PY task_aware/train_codec.py --stage task "${COMMON[@]}" \
      --init "$RUNS/B_r$T/best.pt" --out_dir "$RUNS/Cnofeat_r$T" --target_bits "$T" --epochs "$EPOCHS_C" --lr 5e-5 \
      --lambda_f 0 --lambda_t "$LAMBDA_T" --alpha "$ALPHA" -- "${R2GEN_ARGS[@]}"
  # D2: no span weighting (plain teacher-forced NLL)
  [ -f "$RUNS/Cnospan_r$T/best.pt" ] || $PY task_aware/train_codec.py --stage task "${COMMON[@]}" \
      --init "$RUNS/B_r$T/best.pt" --out_dir "$RUNS/Cnospan_r$T" --target_bits "$T" --epochs "$EPOCHS_C" --lr 5e-5 \
      --lambda_f "$LAMBDA_F" --lambda_t "$LAMBDA_T" --alpha 0 -- "${R2GEN_ARGS[@]}"
done

# 3. validation evaluation of every run (select lambda/alpha/epochs here, never on test)
CONDS=(--cond A=original)
PAIRS=()
for T in $TARGETS; do
  for F in B Bft C Cnofeat Cnospan; do CONDS+=(--cond "${F}_r$T=$RUNS/${F}_r$T/best.pt"); done
  PAIRS+=(--pair "C_r$T:Bft_r$T" --pair "C_r$T:B_r$T" --pair "C_r$T:Cnospan_r$T")
done
$PY task_aware/evaluate.py --feature_dir "$FEAT" --split val --out_dir "$EVAL/val" "${CONDS[@]}" "${PAIRS[@]}" -- "${R2GEN_ARGS[@]}"

# 4. Only after freezing all choices on val:   RUN_TEST=1 bash task_aware/run_experiment.sh
if [ "${RUN_TEST:-0}" = "1" ]; then
  $PY task_aware/evaluate.py --feature_dir "$FEAT" --split test --out_dir "$EVAL/test" "${CONDS[@]}" "${PAIRS[@]}" \
      -- "${R2GEN_ARGS[@]}"
fi
