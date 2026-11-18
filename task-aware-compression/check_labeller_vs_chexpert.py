"""Two checks with the official MIMIC-CXR-JPG CheXpert labels (mimic-cxr-2.0.0-chexpert.csv.gz).

1. Labeller check: on the radiologists' reports of the test split, how well do our rule-based
   labels (clinical_spans, ta-spans-1.0) agree with the CheXpert labels, finding by finding?
2. Codec check with an independent reference: abnormal-finding F1 of each condition's generated
   reports (still labelled by our rules) against the CheXpert labels of the radiologist's report,
   restricted to the findings both systems define.
"Abnormal" = positive or uncertain in both systems (CheXpert 1 or -1).
Note: CheXpert labels were made from the impression section where present; our reference text is
the findings section, so some disagreement in check 1 is a difference of source text.

    python check_labeller_vs_chexpert.py            # run from the medical-compression folder
"""
import csv, gzip, json, random, sys
sys.path.insert(0, 'R2Gen/task_aware')
import clinical_spans as cs

MAP = {'atelectasis': 'Atelectasis', 'cardiomegaly': 'Cardiomegaly', 'consolidation': 'Consolidation',
       'edema': 'Edema', 'fracture': 'Fracture', 'nodule_mass': 'Lung Lesion', 'lung_opacity': 'Lung Opacity',
       'pleural_effusion': 'Pleural Effusion', 'pneumonia': 'Pneumonia', 'pneumothorax': 'Pneumothorax',
       'support_device': 'Support Devices'}
ABN = ('positive', 'uncertain')
chex = {}
with gzip.open('mimic-cxr-2.0.0-chexpert.csv.gz', 'rt') as f:
    for r in csv.DictReader(f):
        chex[int(r['study_id'])] = {c for c in MAP.values() if r[c] in ('1.0', '-1.0')}
ann = json.load(open('R2Gen/data/mimic_cxr/annotation.json'))
study = {e['id']: e['study_id'] for s in ('val', 'test') for e in ann[s]}


def ours(text):
    return {MAP[c] for c, v in cs.report_labels(text.split()).items() if v in ABN and c in MAP}


def prf(tp, fp, fn):
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), 2 * tp / max(2 * tp + fp + fn, 1)


out = {}
d = json.load(open('results/mimic_full/eval_test_r500/per_study.json'))
rows = [r for r in d['A'] if study[r['id']] in chex]
print(f'1. Labeller check on {len(rows)} radiologist reports (test split)')
print(f'   {"finding":22s} CheXpert+  ours+  precision recall  F1')
T = [0, 0, 0]
out['labeller'] = {}
for c in sorted(MAP.values()):
    tp = fp = fn = 0
    for r in rows:
        a, b = c in chex[study[r['id']]], c in ours(r['reference'])
        tp += a and b; fp += b and not a; fn += a and not b
    p, rc, f = prf(tp, fp, fn)
    T = [T[0] + tp, T[1] + fp, T[2] + fn]
    out['labeller'][c] = {'chexpert_pos': tp + fn, 'ours_pos': tp + fp, 'precision': p, 'recall': rc, 'f1': f}
    print(f'   {c:22s} {tp + fn:8d} {tp + fp:6d}   {p:6.2f}   {rc:5.2f}  {f:5.2f}')
p, rc, f = prf(*T)
out['labeller']['micro'] = {'precision': p, 'recall': rc, 'f1': f}
print(f'   {"all findings (micro)":22s} {T[0] + T[2]:8d} {T[0] + T[1]:6d}   {p:6.2f}   {rc:5.2f}  {f:5.2f}')

print('\n2. Abnormal-finding F1 of generated reports against CheXpert labels of the radiologist report')
random.seed(0)
out['codec'] = {}
for split in ('val', 'test'):
    for t in (500, 1000, 2000):
        d = json.load(open(f'results/mimic_full/eval_{split}_r{t}/per_study.json'))
        ids = sorted(r['id'] for r in d['A'] if study[r['id']] in chex)
        cnt = {}
        for c in ('A', 'Bft', 'F'):
            g = {r['id']: ours(r['generated']) for r in d[c]}
            cnt[c] = {i: (len(g[i] & chex[study[i]]), len(g[i] - chex[study[i]]), len(chex[study[i]] - g[i])) for i in ids}
        f1 = lambda c, ix: prf(*[sum(cnt[c][i][k] for i in ix) for k in range(3)])[2]
        n = len(ids)
        diffs = sorted(f1('F', ix) - f1('Bft', ix) for ix in ([ids[random.randrange(n)] for _ in range(n)] for _ in range(1000)))
        res = {c: f1(c, ids) for c in cnt}
        res['F_minus_Bft'] = [res['F'] - res['Bft'], diffs[25], diffs[974]]
        out['codec'][f'{split}_{t}'] = res
        print(f'   {split:4s} {t:5d} bits: uncompressed {res["A"]:.3f}  Bft {res["Bft"]:.3f}  F {res["F"]:.3f}  '
              f'F-Bft {res["F_minus_Bft"][0]:+.3f} [{diffs[25]:+.3f}, {diffs[974]:+.3f}]')
json.dump(out, open('results/mimic_full/chexpert_check.json', 'w'), indent=1)
print('\nsaved results/mimic_full/chexpert_check.json')
