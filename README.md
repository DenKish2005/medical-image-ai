# Medical Image AI

AI-Based Medical Image Classification and Diagnostic Report Generation

CSCI 408 Senior Project I — Fall 2026
CSCI 409 Senior Project II — Spring 2027

## Goal

Develop an AI-based prototype that analyzes medical images
(X-ray or CT), predicts clinically relevant abnormalities,
and generates diagnostic report text.

## Team

- Daniyar Kshibekov
- Ayaulym Zhumagul
- Daulet Kaskenov
- Assel Derbissova

## Supervisor

Min-Ho Lee

## New: CXR feature-compression research module (MICCAI 2027 proposals)

An **additive, research-only** experimental package is available under
[`research_compression/`](research_compression/). It addresses three related
hypotheses: diagnostic-risk-aware feature compression, spatial uncertainty /
importance bit allocation, and a single variable-rate feature codec.

**Start here:** [`RESEARCH_COMPRESSION_GUIDE.md`](RESEARCH_COMPRESSION_GUIDE.md).

The original BLIP/R2Gen notebooks, images and existing IU pilot measurements are
not altered. The new diagnosis/feature-rate experiments require their own
patient-level manifest, disease labels and trained classifier checkpoint;
existing IU X-Ray report-generation scores must not be interpreted as AUROC or
clinical validation of these new methods. A separate hook can be used to try
this codec at a compatible intermediate layer in an existing notebook.
