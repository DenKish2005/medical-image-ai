"""Step 1. Cache frozen ResNet-101 layer4 maps for every IU X-Ray study and verify R2Gen reproduction.

Uses the deterministic evaluation transform (Resize(224,224) + ImageNet normalization) for all
splits, so train/val/test features match what R2Gen sees at test time.

    python task_aware/extract_features.py --out_dir ../results/ta_features --verify 8 -- \
        --image_dir data/iu_xray/images/ --ann_path data/iu_xray/annotation.json \
        --dataset_name iu_xray --load data/model_iu_xray.pth

Writes <out_dir>/{train,val,test}_features.npy (float32 [N,2,2048,7,7]), *_ids.json, manifest.json.
--verify N regenerates N val reports both from images (model(images,'sample')) and from the cached
features via the frozen wrapper and requires identical text (note Sec. 5, first bullet).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ta_common as tc  # noqa: E402


def main():
    mine, theirs = tc.split_argv()
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--splits', default='train,val,test')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--dtype', default='float32', choices=['float32', 'float16'])
    p.add_argument('--verify', type=int, default=8)
    p.add_argument('--subset', default=None, help='json {split: [ids]}; keep only these entries (mimic_cxr)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(mine)
    args = tc.r2gen_args(theirs)
    tc.fix_seed(args.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from torchvision import transforms
    from torch.utils.data import DataLoader
    from modules.datasets import IuxrayMultiImageDataset, MimiccxrSingleImageDataset
    import json
    import os
    subset = json.loads(Path(a.subset).read_text()) if a.subset else None

    def make_dataset(split):
        cls = IuxrayMultiImageDataset if args.dataset_name == 'iu_xray' else MimiccxrSingleImageDataset
        ds = cls(args, tokenizer, split, transform=tf)
        if subset is not None:
            keep = set(subset[split])
            ds.examples = [e for e in ds.examples if e['id'] in keep]
        have = [e for e in ds.examples if all(os.path.isfile(os.path.join(args.image_dir, q)) for q in e['image_path'])]
        if len(have) != len(ds.examples):
            print(f'{split}: {len(ds.examples) - len(have)} entries skipped (image file not on disk)')
            ds.examples = have
        return ds
    from modules.dataloaders import R2DataLoader

    r2, tokenizer = tc.load_frozen_r2gen(args, a.device)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    manifest = {'r2gen_checkpoint': str(args.load), 'r2gen_checkpoint_sha256': tc.sha256_file(args.load),
                'annotation_sha256': tc.sha256_file(args.ann_path), 'transform': 'Resize((224,224)) ToTensor '
                'Normalize(ImageNet)', 'layer': 'visual_extractor.model (ResNet-101 children[:-2]) = layer4 output',
                'dtype': a.dtype, 'splits': {}, 'torch_version': torch.__version__,
                'decoding': {k: getattr(args, k) for k in ('sample_method', 'beam_size', 'block_trigrams',
                                                           'max_seq_length', 'temperature', 'sample_n')}}
    for split in a.splits.split(','):
        ds = make_dataset(split)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=R2DataLoader.collate_fn)
        arr, ids, k = None, [], 0
        for sid, images, _, _ in dl:
            fs = r2.cnn_features(images.to(a.device))
            f = torch.stack(fs, 1).cpu().numpy().astype(a.dtype)
            if arr is None:
                shape = (len(ds),) + f.shape[1:]
                arr = np.lib.format.open_memmap(out / f'{split}_features.npy', mode='w+', dtype=a.dtype, shape=shape)
            arr[k:k + len(f)] = f
            k += len(f)
            ids += list(sid)
            print(f'\r{split}: {k}/{len(ds)}', end='', flush=True)
        arr.flush()
        print()
        tc.json_save(out / f'{split}_ids.json', ids)
        manifest['splits'][split] = {'n': len(ids), 'shape': list(arr.shape)}

    tc.json_save(out / 'manifest.json', manifest)
    print('feature shape per study:', manifest['splits'])

    if a.verify > 0:
        ds = make_dataset('val')
        store = tc.FeatureStore(out, 'val')
        mism = 0
        for i in range(min(a.verify, len(ds))):
            sid, image, *_ = ds[i]
            with torch.no_grad():
                seq = r2.model(image.unsqueeze(0).to(a.device), mode='sample')
            direct = tokenizer.decode_batch(seq.cpu().numpy())[0]
            f = store.batch([i], a.device)
            cached = r2.generate(*tc.views(f))[0]
            mism += direct != cached
            print(f'verify {sid}: {"OK" if direct == cached else "MISMATCH"}')
        manifest['verify'] = {'n': min(a.verify, len(ds)), 'mismatches': mism}
        tc.json_save(out / 'manifest.json', manifest)
        if mism:
            raise SystemExit('Cached-feature path does not reproduce R2Gen output; check dtype/transform.')
    print('done:', out.resolve())


if __name__ == '__main__':
    main()
