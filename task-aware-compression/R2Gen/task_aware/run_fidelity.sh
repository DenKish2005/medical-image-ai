#!/usr/bin/env bash
# Fidelity experiment: train the codec to preserve the UNCOMPRESSED R2Gen report, then compare
# with the feature-only control (Bft) and the earlier reference-loss codec (C) at matched rates.
# Run from the R2Gen folder:   bash task_aware/run_fidelity.sh
# Override with env vars, e.g.  TARGETS="500" LAMBDA_KD=300 TAG=F300 bash task_aware/run_fidelity.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
R2GEN_ARGS=(--image_dir data/iu_xray/images/ --ann_path data/iu_xray/annotation.json --dataset_name iu_xray
            --load data/model_iu_xray.pth --num_workers 2 --beam_size 3 --sample_method beam_search
            --block_trigrams 1 --max_seq_length 60)
FEAT=${FEAT:-../results/ta_features}
RUNS=${RUNS:-../results/ta_full}
TARGETS=${TARGETS:-"500 1000 2000"}
EPOCHS=${EPOCHS:-40}
LAMBDA_F=${LAMBDA_F:-1.0}
LAMBDA_T=${LAMBDA_T:-1.0}
LAMBDA_KD=${LAMBDA_KD:-100}
ALPHA=${ALPHA:-2.0}
TAG=${TAG:-F}
TEACHER=$RUNS/teacher_reports.json

# 1. uncompressed reports for train + val (checked against the saved condition A)
[ -f "$TEACHER" ] || $PY task_aware/make_teacher_reports.py --feature_dir "$FEAT" --out "$TEACHER" \
    --check "$RUNS/eval_compare_r500/per_study.json" -- "${R2GEN_ARGS[@]}"

for T in $TARGETS; do
  # 2. fidelity codec, initialised from the same baseline as Bft and C, same target rate and epochs
  [ -f "$RUNS/${TAG}_c64_r$T/best.pt" ] || $PY task_aware/train_codec_fidelity.py --teacher_reports "$TEACHER" \
      --stage task --feature_dir "$FEAT" --rate_mode target --batch_size 16 \
      --init "$RUNS/B_c64_r$T/best.pt" --out_dir "$RUNS/${TAG}_c64_r$T" --target_bits "$T" \
      --epochs "$EPOCHS" --lr 5e-5 --lambda_f "$LAMBDA_F" --lambda_t "$LAMBDA_T" --lambda_kd "$LAMBDA_KD" \
      --alpha "$ALPHA" -- "${R2GEN_ARGS[@]}"
  # 3. validation: real bitstreams, beam-search reports, paired bootstrap against the control
  $PY task_aware/evaluate.py --feature_dir "$FEAT" --split val --out_dir "$RUNS/eval_fidelity_${TAG}_r$T" \
      --cond A=original --cond "Bft=$RUNS/Bft_c64_r$T/best.pt" --cond "C=$RUNS/C_c64_r$T/best.pt" \
      --cond "F=$RUNS/${TAG}_c64_r$T/best.pt" --pair F:Bft --pair F:C -- "${R2GEN_ARGS[@]}"
done

# 4. chart + table
$PY ../plot_fidelity.py "$RUNS" ../codec_rate_vs_fidelity.png "$TAG"
