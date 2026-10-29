# IU X-Ray feature-compression pilot results

## Experiment setup

Evaluation used 10 IU X-Ray test studies, with two images per study
(20 test images). The 256-centroid mapping was calibrated separately
using 10 training images from five training studies; test images were
excluded.

The experiment used the pretrained IU R2Gen checkpoint and the
hyperprior-featurecoding codec with lambda = 0.01 and the existing
8-bit nonlinear mapping. Model weights remained fixed.


## Before/after results

| Measure | Original | Compressed |
| --- | ---: | ---: |
| Feature bytes per study, mean | 802,816 | 40,899.6 payload |
| Header bytes per study | 0 | 792 |
| Payload + header bytes per study, mean | 802,816 | 41,691.6 |
| All-study bytes including headers | 8,028,160 | 416,916 |
| Aggregate compression ratio including headers | 1x | 19.2561x |
| Feature MSE against original | 0 | 0.12113370 |
| Corpus BLEU-4, 0–1 | 0.089004 | 0.204639 |
| Mean ROUGE-L, 0–1 | 0.322424 | 0.388714 |
| Mean encoding time per image | — | 20.23 ms |
| Mean decoding time per image | — | 42.56 ms |
| Maximum peak allocated GPU memory during codec execution | — | 439.86 MiB |

Aggregate ratio is total original bytes / total compressed bytes. Payload-only ratio is 19.6289x. Including headers, the feature size reduction is 94.8068%.

Scores were computed with R2Gen's vendored pycocoevalcap BLEU and ROUGE implementations. BLEU-4 is a corpus score, not an average of sentence scores. ROUGE-L uses beta=1.2. Both use exactly the same saved reference_eval strings, which include tokenizer normalization, unknown-word replacement and max_seq_length=60 truncation. They are not scores against the full raw report strings or published full-test-set results.

Runtime was measured around DTUFCCodec.encode() and decode() in
run_iu_codec.py using GPU-synchronized wall-clock timing. One measurement
was taken per image after one codec warm-up. Packing, file I/O, feature
extraction and report generation are excluded.

Peak allocated GPU memory includes resident models, features and codec
temporaries. It is not codec-exclusive memory. The maximum increase
above the pre-codec allocation was 91.27 MiB.


## Interpretation

- Four reports are exactly unchanged; six differ textually.
- Higher BLEU-4 and ROUGE-L demonstrate higher reference text overlap on this small selected sample, not improved clinical accuracy.
- Some changes are paraphrases with retained negation. Other changes introduce stronger normality claims or unsupported findings (examples below).

## Three report examples

### Example 1: CXR1004_IM-0005

**Reference (raw):**

The heart, pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is no focal air space opacity to suggest a pneumonia. The aorta is tortuous and ectatic. There are degenerative changes of the acromioclavicular joints. There degenerative changes of the spine. There is an IVC XXXX identified.

**Original-model report:**

the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

**Compressed-model report:**

the lungs are clear . there is no pleural effusion or pneumothorax . the heart and mediastinum are normal . the skeletal structures are normal .

**Observation:** The reference mentions a tortuous/ectatic aorta and degenerative changes of the joints and spine. The original report already omits these findings. Compression changes “no acute bony findings” to the broader “skeletal structures are normal”, which conflicts with the reference description of degeneration. Pleural-effusion and pneumothorax negations are retained.

### Example 2: CXR1008_IM-0009

**Reference (raw):**

Heart size and mediastinal contours are normal in appearance. No consolidative airspace opacities. No radiographic evidence of pleural effusion or pneumothorax. Visualized osseous structures appear intact.

**Original-model report:**

the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

**Compressed-model report:**

heart size and mediastinal contour are normal . pulmonary vascularity is normal . lungs are clear . no pleural effusions or pneumothoraces . degenerative changes in the thoracic spine .

**Observation:** Compression adds “degenerative changes in the thoracic spine”. This finding is not supported by the reference report, which says the visualized osseous structures appear intact. This is a report-level comparison, not an independent image diagnosis.

### Example 3: CXR1015_IM-0001

**Reference (raw):**

Streaky and patchy bibasilar opacities, triangular density projected over the heart on the lateral view. No definite pleural effusion seen, no typical findings of pulmonary edema. Considering differences in technical factors XXXX stable cardiomediastinal silhouette with normal heart size.

**Original-model report:**

heart size and mediastinal contour are normal . pulmonary vascularity is normal . lungs are clear . no pleural effusions or pneumothoraces . degenerative changes in the thoracic spine .

**Compressed-model report:**

heart size and mediastinal contour are normal . pulmonary vascularity is normal . lungs are clear . no pleural effusions or pneumothoraces . degenerative changes in the thoracic spine .

**Observation:** The generated report is exactly unchanged, but both outputs say “lungs are clear” despite reference bibasilar opacities. This illustrates a baseline-model error, not a new error caused by compression.



## Setup changes and limitations to document

- Substituted IU X-Ray and its pretrained checkpoint/annotations after professor approval because matching MIMIC-CXR annotations were unavailable.
- Reused the existing Windows lamofc environment with working CompressAI/ANS; installed missing OpenCV dependency.
- Corrected baseline dataloader unpacking from five values to the actual four values.
- Added R2Gen handler, layer4 replacement hook, training-image mapping construction, payload/header saving and measurements.
- Torchvision pretrained/weights deprecation warnings were non-fatal.
- Different PyTorch versions were checked by baseline report comparison; 10/10 reports matched. Use the lamofc environment for the paired final runs.


## Effect of compression on size and report content

Compression reduced feature size from 8,028,160 bytes to 416,916 bytes
across the 20 images, including decoding headers. This corresponds to
19.26× compression and a 94.8% size reduction. Pre-shared codec weights
and mapping files are excluded.

Four of the 10 study reports were unchanged, while six changed textually.
Some changes were paraphrases, but others affected findings. For example,
CXR1008_IM-0009 gained “degenerative changes in the thoracic spine,” which
was not reported in the reference. In CXR1004_IM-0005, “no acute bony
findings” became the broader statement “the skeletal structures are
normal,” despite degenerative changes described in the reference.

Manual comparison did not identify a direct positive-to-negative or
negative-to-positive reversal for pleural effusion or pneumothorax.
However, broader normality statements and added findings show that the
changes were not purely stylistic.

In CXR1046_IM-0036, “no large pleural effusion” became “no pleural
effusion.” Negation was retained, but removing “large” broadened the
negative claim.

Some errors existed before compression. For example, both versions of
CXR1015_IM-0001 stated that the lungs were clear, although the reference
described bibasilar opacities.



# All 10 reports comparisons


===========================================================================
1. CXR1004_IM-0005
===========================================================================

REFERENCE:
The heart, pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is no focal air space opacity to suggest a pneumonia. The aorta is tortuous and ectatic. There are degenerative changes of the acromioclavicular joints. There degenerative changes of the spine. There is an IVC XXXX identified.

ORIGINAL:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

COMPRESSED:
the lungs are clear . there is no pleural effusion or pneumothorax . the heart and mediastinum are normal . the skeletal structures are normal .

===========================================================================
2. CXR1005_IM-0006
===========================================================================

REFERENCE:
Cardiac and mediastinal contours are within normal limits. The lungs are clear. Bony structures are intact.

ORIGINAL:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

COMPRESSED:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

===========================================================================
3. CXR1008_IM-0009
===========================================================================

REFERENCE:
Heart size and mediastinal contours are normal in appearance. No consolidative airspace opacities. No radiographic evidence of pleural effusion or pneumothorax. Visualized osseous structures appear intact.

ORIGINAL:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

COMPRESSED:
heart size and mediastinal contour are normal . pulmonary vascularity is normal . lungs are clear . no pleural effusions or pneumothoraces . degenerative changes in the thoracic spine .

===========================================================================
4. CXR1011_IM-0013
===========================================================================

REFERENCE:
The heart is top normal in size. The mediastinum is unremarkable. The lungs are clear.

ORIGINAL:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

COMPRESSED:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

===========================================================================
5. CXR1015_IM-0001
===========================================================================

REFERENCE:
Streaky and patchy bibasilar opacities, triangular density projected over the heart on the lateral view. No definite pleural effusion seen, no typical findings of pulmonary edema. Considering differences in technical factors XXXX stable cardiomediastinal silhouette with normal heart size.

ORIGINAL:
heart size and mediastinal contour are normal . pulmonary vascularity is normal . lungs are clear . no pleural effusions or pneumothoraces . degenerative changes in the thoracic spine .

COMPRESSED:
heart size and mediastinal contour are normal . pulmonary vascularity is normal . lungs are clear . no pleural effusions or pneumothoraces . degenerative changes in the thoracic spine .

===========================================================================
6. CXR1017_IM-0013
===========================================================================

REFERENCE:
Both lungs are clear and expanded with no infiltrates. Basilar focal atelectasis is present in the lingula. Heart size normal. Calcified right hilar XXXX are present

ORIGINAL:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

COMPRESSED:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

===========================================================================
7. CXR1030_IM-0024
===========================================================================

REFERENCE:
Normal cardiomediastinal silhouette. There is no focal consolidation. There are no XXXX of a large pleural effusion. There is no pneumothorax. There is no acute bony abnormality seen. Mild degenerative changes of the spine.

ORIGINAL:
the cardiomediastinal silhouette is within normal limits for size and contour . the lungs are normally inflated without evidence of focal airspace disease pleural effusion or pneumothorax . osseous structures are within normal limits for patient age .

COMPRESSED:
the lungs are clear bilaterally . specifically no evidence of focal consolidation pneumothorax or pleural effusion . cardio mediastinal silhouette is unremarkable . visualized osseous structures of the thorax are without acute abnormality .

===========================================================================
8. CXR1034_IM-0028
===========================================================================

REFERENCE:
The heart and lungs have XXXX XXXX in the interval. Both lungs are clear and expanded. Right middle lobe calcified granuloma is unchanged. Heart and mediastinum unchanged. No change hiatus hernia.

ORIGINAL:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

COMPRESSED:
heart size and mediastinal contour are normal . pulmonary vascularity is normal . lungs are clear . no pleural effusions or pneumothoraces . degenerative changes in the thoracic spine .

===========================================================================
9. CXR1046_IM-0036
===========================================================================

REFERENCE:
Heart and mediastinum within normal limits. Negative for focal pulmonary consolidation, pleural effusion, or pneumothorax.

ORIGINAL:
the cardiomediastinal silhouette is normal in size and contour . no focal consolidation pneumothorax or large pleural effusion . negative for acute bone abnormality .

COMPRESSED:
the lungs are clear bilaterally . specifically no evidence of focal consolidation pneumothorax or pleural effusion . cardio mediastinal silhouette is unremarkable . visualized osseous structures of the thorax are without acute abnormality .

===========================================================================
10. CXR106_IM-0042
===========================================================================

REFERENCE:
Heart size and mediastinal contour are normal. Pulmonary vascularity is normal. Lungs are clear. No pleural effusions or pneumothoraces.

ORIGINAL:
the cardiomediastinal silhouette and pulmonary vasculature are within normal limits in size . the lungs are clear of focal airspace disease pneumothorax or pleural effusion . there are no acute bony findings .

COMPRESSED:
heart size and mediastinal contour are normal . pulmonary vascularity is normal . lungs are clear . no pleural effusions or pneumothoraces . degenerative changes in the thoracic spine .
