"""Rebuild R2Gen's MIMIC-CXR annotation.json from PhysioNet files, and check it against the checkpoint.

R2Gen builds its vocabulary from annotation['train'] at run time (words with count >= --threshold,
sorted, plus <unk>). The embedding of model_mimic_cxr.pth has 4336 rows = vocabulary + 1, so a
usable annotation must give exactly 4335 tokens with threshold 10. This script builds several
plausible variants (which report section, one entry per image or per study), reports each
variant's vocabulary size, and writes the variant(s) you choose.

Inputs (in the medical-compression folder): mimic-cxr-reports.zip (MIMIC-CXR 2.x),
mimic-cxr-2.0.0-split.csv (MIMIC-CXR-JPG). No torch needed.

    python build_mimic_annotation.py --scan                     # vocabulary size of every variant
    python build_mimic_annotation.py --write findings_fallback:image --out data/mimic_cxr/annotation.json
"""
import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modules.tokenizers import Tokenizer  # noqa: E402  (pure python: json, re, Counter)
sys.path.insert(0, str(HERE / 'mimic_txt'))
import section_parser as sp  # noqa: E402  (official MIT-LCP/mimic-cxr txt/section_parser.py)
CUSTOM_NAMES, CUSTOM_IDX = sp.custom_mimic_cxr_rules()


def official(study, text):
    """{normalized section name: text of the LAST section with that name}, as create_section_files.py."""
    secs, names, _ = sp.section_text(text)
    out = {}
    for n, t in zip(names, secs):
        out[n] = t.strip()
    key = 's' + study
    if key in CUSTOM_IDX:
        i, j = CUSTOM_IDX[key]
        out = {'impression': text[i:j].strip()}
    elif key in CUSTOM_NAMES:
        out[CUSTOM_NAMES[key]] = out.get(CUSTOM_NAMES[key], '')
    return out

TARGET_VOCAB = 4335          # model_mimic_cxr.pth: tgt_embed / logit have 4336 rows = vocab + 1
THRESHOLD = 10               # test_mimic_cxr.sh / train_mimic_cxr.sh
HEADER = re.compile(r'^\s*([A-Z][A-Z0-9 ,/&()\-]{1,60}):', re.M)


def sections(text):
    """Split a free-text report into {HEADER: body}, plus the preamble and the last paragraph."""
    out, marks = {}, list(HEADER.finditer(text))
    out['_PREAMBLE'] = text[:marks[0].start()] if marks else text
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        name = re.sub(r'\s+', ' ', m.group(1)).strip()
        if name not in out:
            out[name] = text[m.end():end]
    paras = [p for p in re.split(r'\n\s*\n', text) if p.strip()]
    out['_LAST'] = paras[-1] if paras else ''
    return out


def pick(sec, *names):
    for want in names:
        for k, v in sec.items():
            if not k.startswith('_') and want in k and v.strip():
                return v
    return ''


def strip_header(text):
    return HEADER.sub('', text, count=1) if HEADER.match(text) else text


def _o(sec, *names):
    for n in names:
        if sec.get(n):
            return sec[n]
    return ''


OFFICIAL = {
    'off_findings': lambda o: _o(o, 'findings'),
    'off_findings_fallback': lambda o: _o(o, 'findings', 'impression', 'last_paragraph', 'comparison'),
    'off_findings_impression': lambda o: ' '.join(x for x in (_o(o, 'findings'), _o(o, 'impression')) if x),
}

VARIANTS = {
    'findings': lambda s: pick(s, 'FINDINGS'),
    'findings_fallback': lambda s: pick(s, 'FINDINGS') or pick(s, 'IMPRESSION') or strip_header(s['_LAST']),
    'findings_impression': lambda s: ' '.join(x for x in (pick(s, 'FINDINGS'), pick(s, 'IMPRESSION')) if x),
    'impression_fallback': lambda s: pick(s, 'IMPRESSION') or pick(s, 'FINDINGS') or strip_header(s['_LAST']),
}


def load(base):
    z = zipfile.ZipFile(base / 'mimic-cxr-reports.zip')
    reports = {}
    for n in z.namelist():
        if n.endswith('.txt'):
            p = n.split('/')
            reports[p[-1][1:-4]] = (p[-2][1:], z.read(n).decode('utf-8', 'replace'))  # study -> (subject, text)
    split = defaultdict(list)
    with open(base / 'mimic-cxr-2.0.0-split.csv', newline='') as f:
        for r in csv.DictReader(f):
            split[r['study_id']].append((r['dicom_id'], r['split']))
    return reports, split


class _Args:
    def __init__(self, ann_path):
        self.ann_path, self.threshold, self.dataset_name = ann_path, THRESHOLD, 'mimic_cxr'


def build(reports, split, variant, unit):
    fn = VARIANTS.get(variant)
    ofn = OFFICIAL.get(variant)
    ann = {'train': [], 'val': [], 'test': []}
    empty = 0
    for study, images in sorted(split.items()):
        if study not in reports:
            continue
        subject, text = reports[study]
        rep = (ofn(official(study, text)) if ofn else fn(sections(text))).strip()
        if not rep:
            empty += 1
            continue
        s = images[0][1]
        s = 'val' if s == 'validate' else s
        folder = f'p{subject[:2]}/p{subject}/s{study}'
        dicoms = sorted(d for d, _ in images)
        if unit == 'image':
            for d in dicoms:
                ann[s].append({'id': d, 'study_id': int(study), 'subject_id': int(subject), 'report': rep,
                               'image_path': [f'{folder}/{d}.jpg'], 'split': s})
        else:
            ann[s].append({'id': dicoms[0], 'study_id': int(study), 'subject_id': int(subject), 'report': rep,
                           'image_path': [f'{folder}/{d}.jpg' for d in dicoms], 'split': s})
    return ann, empty


def vocab_size(ann, tmp):
    tmp.write_text(json.dumps({'train': ann['train']}))
    return len(Tokenizer(_Args(str(tmp))).idx2token)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=str(HERE.parent), help='folder with the zip and split csv')
    p.add_argument('--scan', action='store_true')
    p.add_argument('--write', default=None, help='variant:unit, e.g. findings_fallback:image')
    p.add_argument('--out', default='data/mimic_cxr/annotation.json')
    p.add_argument('--variants', nargs='*', default=None)
    p.add_argument('--tmp', default='/tmp/_mimic_vocab.json')
    a = p.parse_args()
    reports, split = load(Path(a.base))
    print(f'{len(reports)} reports, {len(split)} studies in split file')
    tmp = Path(a.tmp)
    if a.scan:
        for v in (a.variants or list(VARIANTS) + list(OFFICIAL)):
            for unit in ('image', 'study'):
                ann, empty = build(reports, split, v, unit)
                n = vocab_size(ann, tmp)
                print(f'{v:22s} {unit:5s}  train/val/test = {len(ann["train"])}/{len(ann["val"])}/{len(ann["test"])}'
                      f'  empty={empty:6d}  vocab={n}  {"MATCH" if n == TARGET_VOCAB else f"{n - TARGET_VOCAB:+d}"}',
                      flush=True)
    if a.write:
        v, unit = a.write.split(':')
        ann, empty = build(reports, split, v, unit)
        n = vocab_size(ann, tmp)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ann))
        print(f'wrote {out} ({v}, {unit}); vocab {n}; target {TARGET_VOCAB}; '
              f'{"MATCH" if n == TARGET_VOCAB else "MISMATCH - the checkpoint will not decode correctly"}')


if __name__ == '__main__':
    main()
