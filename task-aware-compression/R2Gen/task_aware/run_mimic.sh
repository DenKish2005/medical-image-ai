#!/usr/bin/env bash
# MIMIC-CXR experiment with the frozen MIMIC R2Gen checkpoint: feature-only codec (Bft) vs the
# fidelity codec (F, trained toward the uncompressed report) at matched actual bits per image.
# Run from the R2Gen folder:
#   SMOKE=1 bash task_aware/run_mimic.sh     # few-minute end-to-end test on a handful of images
#   bash task_aware/run_mimic.sh             # real run (default target: 500 bits per image)
#   TARGETS="250 1000" bash task_aware/run_mimic.sh
# Finished steps are skipped, so it is safe to re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
R2GEN_ARGS=(--image_dir data/mimic_cxr/images/ --ann_path data/mimic_cxr/annotation.json --dataset_name mimic_cxr
            --load data/model_mimic_cxr.pth --num_workers 2 --beam_size 3 --sample_method beam_search
            --block_trigrams 1 --max_seq_length 100 --threshold 10)
FEAT=${FEAT:-../results/mimic_features}
TARGETS=${TARGETS:-"500"}
LAMBDA_KD=${LAMBDA_KD:-100}
SEED=${SEED:-1234}
if [ "${SMOKE:-0}" = "1" ]; then
  RUNS=../results/mimic_smoke; EB=1; EF=1; LIMIT=(--max_train 64 --max_val 32); EVALN=(--max_n 32); TMAX=(--max_n 64)
  SPLITS="val"; FEAT=$RUNS/features; SUBSET=data/mimic_cxr/subset_smoke.json
else
  RUNS=${RUNS:-../results/mimic_full}; EB=${EB:-30}; EF=${EF:-10}; LIMIT=(); EVALN=(); TMAX=()
  SPLITS="val test"; SUBSET=data/mimic_cxr/subset.json
fi
TEACHER=$RUNS/teacher_reports.json
COMMON=(--feature_dir "$FEAT" --rate_mode target --batch_size 16 --seed "$SEED")
mkdir -p "$RUNS"

# which subset images are on disk? (smoke: a few of them; real run: all must be there)
$PY - "$SUBSET" "${SMOKE:-0}" "${ALLOW_PARTIAL:-0}" <<'PYEOF'
import json, os, sys
sub = json.load(open('data/mimic_cxr/subset.json'))
ann = json.load(open('data/mimic_cxr/annotation.json'))
have, total = {}, 0
for split in ('train', 'val', 'test'):
    path = {e['id']: e['image_path'][0] for e in ann[split]}
    have[split] = [i for i in sub[split] if os.path.isfile('data/mimic_cxr/images/' + path[i])]
    total += len(sub[split])
    print(f'{split}: {len(have[split])}/{len(sub[split])} images on disk')
if sys.argv[2] == '1':
    json.dump({'train': have['train'][:96], 'val': have['val'][:48], 'test': have['test'][:48]}, open(sys.argv[1], 'w'))
elif sum(map(len, have.values())) < total and sys.argv[3] != '1':
    sys.exit('The download is not complete. Wait for download_mimic.py to finish (or re-run it), '
             'or set ALLOW_PARTIAL=1 to train on what is there.')
PYEOF

# 1. cache layer4 features of the downloaded subset (one 2048x7x7 map per image)
[ -f "$FEAT/manifest.json" ] || $PY task_aware/extract_features.py --out_dir "$FEAT" \
    --subset "$SUBSET" --splits train,val,test -- "${R2GEN_ARGS[@]}"

# 2. reports of the uncompressed model for train + val (targets of the fidelity codec)
[ -f "$TEACHER" ] || $PY task_aware/make_teacher_reports.py --feature_dir "$FEAT" --out "$TEACHER" \
    ${TMAX[@]+"${TMAX[@]}"} -- "${R2GEN_ARGS[@]}"

for T in $TARGETS; do
  B=$RUNS/B_r$T; BFT=$RUNS/Bft_r$T; F=$RUNS/F_r$T
  [ -f "$B/best.pt" ] || $PY task_aware/train_codec.py --stage baseline "${COMMON[@]}" ${LIMIT[@]+"${LIMIT[@]}"} --c_y 64 \
      --out_dir "$B" --target_bits "$T" --epochs "$EB" --lr 1e-4 --lambda_f 1.0 -- "${R2GEN_ARGS[@]}"
  [ -f "$BFT/best.pt" ] || $PY task_aware/train_codec.py --stage baseline "${COMMON[@]}" ${LIMIT[@]+"${LIMIT[@]}"} \
      --init "$B/best.pt" --out_dir "$BFT" --target_bits "$T" --epochs "$EF" --lr 5e-5 --lambda_f 1.0 \
      -- "${R2GEN_ARGS[@]}"
  [ -f "$F/best.pt" ] || $PY task_aware/train_codec_fidelity.py --teacher_reports "$TEACHER" --stage task \
      "${COMMON[@]}" ${LIMIT[@]+"${LIMIT[@]}"} --init "$B/best.pt" --out_dir "$F" --target_bits "$T" --epochs "$EF" \
      --lr 5e-5 --lambda_f 1.0 --lambda_t 1.0 --lambda_kd "$LAMBDA_KD" --alpha 2.0 -- "${R2GEN_ARGS[@]}"
  for SPLIT in $SPLITS; do
    OUT=$RUNS/eval_${SPLIT}_r$T
    [ -f "$OUT/per_study.json" ] || $PY task_aware/evaluate.py --feature_dir "$FEAT" --split "$SPLIT" \
        --out_dir "$OUT" ${EVALN[@]+"${EVALN[@]}"} --cond A=original --cond "Bft=$BFT/best.pt" --cond "F=$F/best.pt" \
        --pair F:Bft -- "${R2GEN_ARGS[@]}"
  done
done
echo "done: $RUNS"
