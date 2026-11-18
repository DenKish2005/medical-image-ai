"""Per-finding comparison of MIMIC-CXR test reports (uncompressed, Bft, F) at 500 and 1,000 bits, with example cases.
Run from the medical-compression folder:  python analyze_mimic_reports.py
"""
import csv, gzip, json, sys, collections, random
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), 'R2Gen', 'task_aware'))
import clinical_spans as cs
import os
B = os.path.dirname(os.path.abspath(__file__)) + '/'
MAP = {'atelectasis':'Atelectasis','cardiomegaly':'Cardiomegaly','consolidation':'Consolidation','edema':'Edema','fracture':'Fracture',
       'nodule_mass':'Lung Lesion','lung_opacity':'Lung Opacity','pleural_effusion':'Pleural Effusion','pneumonia':'Pneumonia',
       'pneumothorax':'Pneumothorax','support_device':'Support Devices'}
chex = {}
with gzip.open(B+'mimic-cxr-2.0.0-chexpert.csv.gz','rt') as f:
    for r in csv.DictReader(f): chex[int(r['study_id'])] = {c for c in MAP.values() if r[c] in ('1.0','-1.0')}
ann = json.load(open(B+'R2Gen/data/mimic_cxr/annotation.json'))
study = {e['id']: e['study_id'] for e in ann['test']}
def ours(t): return {MAP[c] for c,v in cs.report_labels(t.split()).items() if v in ('positive','uncertain') and c in MAP}
out = {}
for T in (500, 1000):
    d = json.load(open(B+f'results/mimic_full/eval_test_r{T}/per_study.json'))
    R = {c:{r['id']:r for r in d[c]} for c in ('A','Bft','F')}
    ids = sorted(i for i in R['A'] if study.get(i) in chex)
    S = {c:{i:ours(R[c][i]['generated']) for i in ids} for c in R}
    ref = {i: chex[study[i]] for i in ids}
    stats = collections.Counter(); per = collections.defaultdict(collections.Counter)
    for i in ids:
        a = S['A'][i]
        for c in ('Bft','F'):
            g = S[c][i]
            stats[c+'_identical'] += R[c][i]['generated'] == R['A'][i]['generated']
            stats[c+'_same_abn_set'] += g == a
            stats[c+'_reworded_same_abn'] += (g == a) and R[c][i]['generated'] != R['A'][i]['generated']
            for k in a & ref[i]:            # correct finding of uncompressed model
                per[k][c+'_kept'] += k in g; per[k]['A_tp_'+c] += 1
            for k in g - a:                 # finding not in uncompressed report
                stats[c+'_added'] += 1; stats[c+'_added_correct'] += k in ref[i]; per[k][c+'_added'] += 1
            for k in a - g:
                stats[c+'_dropped'] += 1; stats[c+'_dropped_was_correct'] += k in ref[i]
    print(f'=== test, {T} bits, {len(ids)} images')
    for c in ('Bft','F'):
        print(c, {k[len(c)+1:]: v for k, v in stats.items() if k.startswith(c+'_')})
    print('finding | uncompressed correct | kept by Bft | kept by F | added by Bft | added by F')
    for k in sorted(per, key=lambda k: -per[k]['A_tp_F']):
        p = per[k]; print(f"{k:18s} {p['A_tp_F']:5d} {p['Bft_kept']:5d} {p['F_kept']:5d} {p['Bft_added']:5d} {p['F_added']:5d}")
    out[T] = (R, S, ref, ids)
# examples at 500 bits
R, S, ref, ids = out[500]
random.seed(11)
def show(i, why):
    print('\n---', why, i)
    print('  RADIOLOGIST:', R['A'][i]['reference'][:330]); print('    CheXpert abnormal:', sorted(ref[i]))
    for c in ('A','Bft','F'):
        print(f'  {c:3s} ({R[c][i]["bits"] if c!="A" else "uncompressed"}):', R[c][i]['generated'][:330]); print('    found:', sorted(S[c][i]))
groups = collections.defaultdict(list)
for i in ids:
    a, b, f, r = S['A'][i], S['Bft'][i], S['F'][i], ref[i]
    kb, kf = len(a & r & b), len(a & r & f)
    if kf > kb and b != a: groups['F keeps a correct finding Bft loses'].append(i)
    if kb > kf and f != a: groups['Bft keeps a correct finding F loses'].append(i)
    if (f - a) - r and not ((b - a) - r): groups['F invents a finding'].append(i)
    if (b - a) - r and not ((f - a) - r): groups['Bft invents a finding'].append(i)
for g, l in groups.items(): print(g, len(l))
for g, n in (('F keeps a correct finding Bft loses', 3), ('Bft keeps a correct finding F loses', 2), ('F invents a finding', 1), ('Bft invents a finding', 1)):
    for i in random.sample(groups[g], n): show(i, g)
