"""Choose the MIMIC-CXR images to download and write a wget URL list.

The full data/mimic_cxr/annotation.json must stay as it is (R2Gen builds its vocabulary from it).
This writes data/mimic_cxr/subset.json (ids per split) and data/mimic_cxr/download_urls.txt.
    python make_mimic_subset.py --train 12000 --val 1000        # test split is always complete
"""
import argparse
import json
import random
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--ann', default='data/mimic_cxr/annotation.json')
p.add_argument('--train', type=int, default=12000)
p.add_argument('--val', type=int, default=1000)
p.add_argument('--seed', type=int, default=2026)
a = p.parse_args()
ann = json.loads(Path(a.ann).read_text())
rng = random.Random(a.seed)
pick = {'train': rng.sample(ann['train'], min(a.train, len(ann['train']))),
        'val': rng.sample(ann['val'], min(a.val, len(ann['val']))), 'test': ann['test']}
base = 'https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
out = Path(a.ann).parent
(out / 'subset.json').write_text(json.dumps({k: [e['id'] for e in v] for k, v in pick.items()}))
urls = [base + e['image_path'][0] for k in ('test', 'val', 'train') for e in pick[k]]
(out / 'download_urls.txt').write_text('\n'.join(urls) + '\n')
print({k: len(v) for k, v in pick.items()}, 'total', len(urls), '-> ', out / 'download_urls.txt')
