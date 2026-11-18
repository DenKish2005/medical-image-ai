# Task-aware feature compression for R2Gen (implementation of the technical note)

Drop this `task_aware/` folder into your `R2Gen/` folder (next to `main_test.py`). Nothing in R2Gen or
LaMoFCBench is modified. Requirements: your existing R2Gen env + `compressai` (1.2.x), `matplotlib`.

## Why the earlier pilots showed "no success"

Your `results/` runs (`iu_task_aware_pilot`, `iu_controlled_*`, `iu_weighted_v3_*`, `iu_strong_*`) behave
exactly as their setup predicts. The method isn't the problem; the experiment couldn't show anything:

| Problem in the pilots | Evidence from your results | Fix here |
|---|---|---|
| **No headroom.** The codec ran at rates where R2Gen isn't degraded, so the report loss had nothing to recover. | val NLL uncompressed 1.400; compressed 1.395 (λ=0.01, 19×) and 1.408 (λ=0.001, 200×). R2Gen tolerated feature NMSE ≈ 0.61. | Sweep **much lower** rates (500–8000 bits/study ≈ 800×–12,800×), where baseline B actually loses clinical content. |
| **Wrong codec.** A pretrained RGB *image* codec (bmshj2018) ran on 8‑bit, nonlinearly mapped, tiled feature planes. It wasn't the note's feature codec. | `FeatureAdapter` + `codec._quantize` + LaMoFC mapping | `feature_codec.py`: a native 2048→C_y scale/mean hyperprior on the 7×7 grid (note §3). |
| **Essentially no training.** 1 epoch × 100 studies × lr 1e‑5 = 100 small steps. | before/after differences of ±0.003–0.008 NLL across 50 studies (noise) | Full train split (≈2k studies), cached features, 100 + 40 epochs, lr 1e‑4/5e‑5 |
| **Rate not really in the objective.** 0.001 × bits/feature-value ≈ 0.002, compared with a text loss of ≈1. | rate barely moved; "after" bytes changed by chance | Lagrangian rate term with a **rate controller** that hits a target *actual* bits/study |
| **No real baseline B, no matched-rate comparison.** before/after compared a pretrained codec with a fine-tuned one at different rates. | 19.20× vs 20.02×, 199.9× vs 208.2× | B and C trained to the **same target rate**, plus a `Bft` control (B fine-tuned for the same extra epochs) |
| **Span weighting was nearly uniform.** Sentence-level rules upweighted 64% of tokens. | α=0 and α=2 runs: −0.00838 vs −0.00823 | Phrase-level spans (cue + finding + modifiers), ≈47% of tokens, with polarity unit tests |
| **Evaluation was teacher-forced loss only.** | no generated-report clinical metrics | Beam-search reports from decoded files, plus NLG + clinical P/R/F1, omissions, polarity/uncertainty errors, and unsupported findings |
| JSON headers ≈ 363 B/view counted as rate | 36,300 header bytes / 100 views; ≈18% of bytes at λ=0.001 | 4‑byte binary container + varint lengths. Every byte is counted. |

## Files

| File | Note § | What it does |
|---|---|---|
| `feature_codec.py` | 3 | g_a / h_a / h_s / g_s + factorized p(z) + conditional Gaussian p(y|z). Modes: `noise` (training), `ste`, and `round` (= deployed). Includes real `compress`/`decompress` and the binary container. |
| `clinical_spans.py` | 4.3, 7 | Span rules (negation/uncertainty scope + modifiers), token weights, report-level concept labels, error counting, 21 polarity unit tests (`python task_aware/clinical_spans.py`) |
| `ta_common.py` | 5 | Frozen R2Gen wrapper (same feature path as `forward_iu_xray`; grads flow to F̂, never to R2Gen), feature store, weighted NLL, NMSE |
| `extract_features.py` | 5 | Caches layer4 maps once. `--verify` checks that the cached path reproduces R2Gen's own reports. |
| `train_codec.py` | 4, 5 | Stage `baseline` (B) and `task` (C/D). Uses a target-rate controller or fixed λ. Selects on validation only. Saves weights, CDF tables, config, seed, and hashes. |
| `evaluate.py` | 6 | Runs compress → `.bin` file → decode → beam search for every condition. Reports NLG + clinical metrics, paired bootstrap CIs, matched-rate (BD-style) deltas, and `rate_quality.png`. |
| `check_pipeline.py` | 7 | Gradient, roundtrip, leakage, polarity, and checkpoint checks |
| `run_experiment.sh` | 6 | The whole A/B/Bft/C/D grid at TARGETS bits/study. Test split only with `RUN_TEST=1`. |

## Run

```bash
cd R2Gen
R2="--image_dir data/iu_xray/images/ --ann_path data/iu_xray/annotation.json --dataset_name iu_xray --load data/model_iu_xray.pth"
# smoke test (minutes): tiny subsets, one target
python task_aware/extract_features.py --out_dir ../results/ta_features --verify 8 -- $R2
python task_aware/check_pipeline.py   --feature_dir ../results/ta_features -- $R2
python task_aware/train_codec.py --stage baseline --feature_dir ../results/ta_features --out_dir ../results/ta_smoke/B \
       --target_bits 2000 --epochs 3 --max_train 200 --max_val 50 -- $R2
# full experiment
bash task_aware/run_experiment.sh                 # val only
RUN_TEST=1 bash task_aware/run_experiment.sh      # after freezing choices on val
```

Script arguments go **before** `--`. R2Gen arguments (`main_test.parse_agrs`) go **after** it.

## How to read the results (`../results/ta_eval/val/`)

* `summary.csv`: one row per condition with actual bits/study, compression ratio, NLL, BLEU/METEOR/ROUGE/CIDEr,
  `ref_*` (clinical metrics vs the reference report), and `A_*` (agreement with the *uncompressed* R2Gen
  report, which is the cleanest measure of what compression destroys).
* `summary.json → matched_rate["C-B"]`: average metric difference between curves C and B over their
  overlapping actual-rate range. `pairs`: paired bootstrap 95% CIs on the same studies.
* `rate_quality.png`: the plot the note asks for.
* What success looks like: at rates where B's `A_mention_f1` / `ref_abn_f1` clearly drops below A, C sits above
  B **and** above `Bft`, and the CI on `C:Bft` excludes 0. If B matches A at every target, lower TARGETS.

## Knobs worth tuning on val (not test)
`LAMBDA_T` (0.3 / 1 / 3), `ALPHA` (1 / 2 / 4), `C_Y` (32–512), `--quant ste` (often better at low rate),
`--feature_norm channel`. The controller (`--mu_lr`) only sets the rate. If `best.pt` warns that the rate
tolerance was not met, train longer or raise `--mu_lr`.

## Caveats
* The span/concept rules are a documented heuristic, not ground truth. Audit a sample by hand and, for the
  paper, also score the generated reports with CheXbert/RadGraph.
* rANS decoding must run on the same platform/device class as encoding (CompressAI caveat).
* Codec weights and CDF tables are pre-shared and excluded from the per-study rate, as in your pilot.
