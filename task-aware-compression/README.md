# Task-aware compression of R2Gen features

Spend fewer bits while preserving the clinically meaningful content of the report generated from the
compressed feature.

R2Gen (ResNet-101 + memory-driven Transformer, Chen et al., EMNLP 2020) is **frozen**. Only a learned codec
between its image encoder and its report decoder is trained:

```
X-ray -> ResNet-101 (frozen) -> feature 2048x7x7 (3,211,264 bits) -> codec -> 500-2,000 bits
      -> codec decoder -> approximate feature -> Transformer (frozen) -> report
```

Two codecs are compared at the same actual bit rate:

| codec | training loss |
|---|---|
| **Bft** (feature-only) | rate + feature reconstruction error |
| **F** (task-aware) | rate + feature error + span-weighted NLL of the **uncompressed model's own report** + token-level KL to the uncompressed model |

The codec is a mean-scale hyperprior (Balle/Minnen) on the 7x7 grid with CompressAI entropy models; the
contribution is the training objective.

## Main result (MIMIC-CXR test, 3,858 images, one run per codec)

Abnormal-finding F1 of the generated report against the CheXpert labels of the radiologist report
(uncompressed R2Gen: 0.281), and share of images whose findings equal the uncompressed report.

| target bits | F1 Bft | F1 F | F - Bft (95% CI) | same findings Bft | same findings F |
|---|---|---|---|---|---|
| 500 | 0.218 | 0.268 | +0.051 [+0.039, +0.064] | 28.8% | 44.1% |
| 1,000 | 0.258 | 0.274 | +0.015 [+0.005, +0.024] | 49.6% | 61.6% |
| 2,000 | 0.276 | 0.284 | +0.008 [+0.001, +0.017] | 67.4% | 73.0% |

Actual rates (all bytes of the bitstream file): Bft 512 / 1,020 / 2,046, F 535 / 1,037 / 2,069 bits per image.
Findings come from a rule-based labeller (`clinical_spans.py`, micro-F1 0.67 vs CheXpert on radiologist reports).
On validation the F1 gain is clear only at 500 bits. Known limitations: about 160 of the ~500 bits are
container and coder overhead, and at 500 bits the hyperprior appears to carry no information (z costs ~0 bits).

## Repository layout

```
R2Gen/                      files to ADD to a clone of https://github.com/zhjohnchan/R2Gen (R2Gen itself is unmodified)
  task_aware/               codec, losses, training, evaluation, experiment scripts
    feature_codec.py          codec + bitstream container
    clinical_spans.py         clinical span rules / labels (ta-spans-1.0)
    ta_common.py              frozen R2Gen wrapper, cached features, targets, losses
    extract_features.py       cache ResNet layer4 features
    make_teacher_reports.py   uncompressed reports = training targets of F
    train_codec.py            B / Bft (rate + feature error)
    train_codec_distill.py    adds KL distillation
    train_codec_fidelity.py   F: targets replaced by the teacher reports
    evaluate.py               deployed-path evaluation (bytes on disk -> decode -> beam search -> metrics)
    run_mimic.sh              full MIMIC-CXR experiment (resumable; SMOKE=1 for a quick test)
    run_fidelity.sh, run_seeds.sh   IU X-Ray experiments
  build_mimic_annotation.py rebuild R2Gen's MIMIC annotation.json from PhysioNet files
  make_mimic_subset.py      choose the image subset (12,000 train / 1,000 val / full test)
  download_mimic.py         download those images from PhysioNet
  mimic_txt/                report section parser (MIT-LCP mimic-cxr code)
task_aware_codec_walkthrough.ipynb   code walkthrough (outputs cleared, see below)
check_labeller_vs_chexpert.py, analyze_mimic_reports.py, plot_*.py, summarize_seeds.py   analysis
results/                    aggregate results only: summary.json/csv, configs, training histories
figures/                    result charts
```

## Data and weights are NOT in this repository

MIMIC-CXR is credentialed data (PhysioNet data use agreement). This repo contains **no** images, reports,
annotation files, cached features, per-study outputs (they contain report text), or model weights.
Do not add them; `.gitignore` blocks the usual file names. Only people with their own PhysioNet
MIMIC-CXR access can reproduce the MIMIC experiments.

## Reproducing (requires PhysioNet access to MIMIC-CXR and MIMIC-CXR-JPG)

1. `git clone https://github.com/zhjohnchan/R2Gen` and copy this repo's `R2Gen/` files into it.
   Put the pretrained `model_mimic_cxr.pth` (from the R2Gen README) in `R2Gen/data/`.
2. Environment: PyTorch + torchvision, `compressai==1.2.8`, numpy, matplotlib (and `notebook` for the walkthrough).
3. From the project folder (parent of `R2Gen/`), with `mimic-cxr-reports.zip` and `mimic-cxr-2.0.0-split.csv` there:
   `cd R2Gen && python build_mimic_annotation.py --scan`, then `--write <variant> --out data/mimic_cxr/annotation.json`
   with the variant whose vocabulary is exactly 4,335 tokens (required by the checkpoint).
4. `python make_mimic_subset.py --train 12000 --val 1000` then `python download_mimic.py --user <physionet user>`
5. `bash task_aware/run_mimic.sh` (add `SMOKE=1` first for a few-minute test; `TARGETS="500 1000 2000"` for all rates).
   Takes many hours on one GPU. Trained codecs and evaluations go to `../results/mimic_full/`.
6. CheXpert check and figures: put `mimic-cxr-2.0.0-chexpert.csv.gz` in the project folder, then
   `python check_labeller_vs_chexpert.py` and `python plot_mimic.py`.

The notebook walks through the code and reproduces the reports and tables from the trained checkpoints.
Run it from the project folder after step 5. The committed copy has its outputs cleared, because the outputs
print MIMIC report text.
