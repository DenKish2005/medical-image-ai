#!/usr/bin/env bash
# Seed replication on IU X-Ray: retrain the baseline (B), the feature-only control (Bft) and the
# fidelity codec (F) with new random seeds, then evaluate on validation and test.
# Seed 1234 is the run you already have (B_c64_r500, Bft_c64_r500, F_c64_r500).
# Run from the R2Gen folder:   bash task_aware/run_seeds.sh
# Override with env vars, e.g.  SEEDS="1 2 3" TARGETS="500 1000" bash task_aware/run_seeds.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
R2GEN_ARGS=(--image_dir data/iu_xray/images/ --ann_path data/iu_xray/annotation.json --dataset_name iu_xray
            --load data/model_iu_xray.pth --num_workers 2 --beam_size 3 --sample_method beam_search
            --block_trigrams 1 --max_seq_length 60)
FEAT=${FEAT:-../results/ta_features}
RUNS=${RUNS:-../results/ta_full}
SEEDS=${SEEDS:-"1 2"}
TARGETS=${TARGETS:-"500"}
LAMBDA_KD=${LAMBDA_KD:-100}
TEACHER=$RUNS/teacher_reports.json
COMMON=(--feature_dir "$FEAT" --rate_mode target --batch_size 16)

[ -f "$TEACHER" ] || { echo "Missing $TEACHER - run task_aware/run_fidelity.sh first"; exit 1; }

for T in $TARGETS; do
  for S in $SEEDS; do
    B=$RUNS/B_c64_r${T}_s$S; BFT=$RUNS/Bft_c64_r${T}_s$S; F=$RUNS/F_c64_r${T}_s$S
    # baseline from scratch (same settings as B_c64_r$T, new seed)
    [ -f "$B/best.pt" ] || $PY task_aware/train_codec.py --stage baseline "${COMMON[@]}" --c_y 64 \
        --out_dir "$B" --target_bits "$T" --epochs 100 --lr 1e-4 --lambda_f 1.0 --seed "$S" -- "${R2GEN_ARGS[@]}"
    # feature-only control: same extra epochs as F
    [ -f "$BFT/best.pt" ] || $PY task_aware/train_codec.py --stage baseline "${COMMON[@]}" \
        --init "$B/best.pt" --out_dir "$BFT" --target_bits "$T" --epochs 40 --lr 5e-5 --lambda_f 1.0 \
        --seed "$S" -- "${R2GEN_ARGS[@]}"
    # fidelity codec: trained toward the uncompressed R2Gen report
    [ -f "$F/best.pt" ] || $PY task_aware/train_codec_fidelity.py --teacher_reports "$TEACHER" --stage task \
        "${COMMON[@]}" --init "$B/best.pt" --out_dir "$F" --target_bits "$T" --epochs 40 --lr 5e-5 \
        --lambda_f 1.0 --lambda_t 1.0 --lambda_kd "$LAMBDA_KD" --alpha 2.0 --seed "$S" -- "${R2GEN_ARGS[@]}"
    for SPLIT in val test; do
      OUT=$RUNS/eval_seed_${SPLIT}_r${T}_s$S
      [ -f "$OUT/per_study.json" ] || $PY task_aware/evaluate.py --feature_dir "$FEAT" --split "$SPLIT" \
          --out_dir "$OUT" --cond A=original --cond "Bft=$BFT/best.pt" --cond "F=$F/best.pt" --pair F:Bft \
          -- "${R2GEN_ARGS[@]}"
    done
  done
done

$PY ../summarize_seeds.py "$RUNS"
