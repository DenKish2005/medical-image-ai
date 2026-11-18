"""Does the MIMIC-CXR R2Gen checkpoint use the image, on real MIMIC-CXR test images?

Same diagnostics as check_mimic_on_iu.py, but on the downloaded MIMIC-CXR-JPG test images and
against MIMIC reference reports (findings section). Uses the test entries whose image file is
already on disk, up to --max of them (seeded sample), once with the correct image and once with
images shuffled between entries.

Run from the R2Gen folder:
    python check_mimic_test.py --out ../results/mimic_test_check.json
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'task_aware'))


def mimic_args(load, ann, beam):
    from main_test import parse_agrs
    old = sys.argv
    sys.argv = [old[0], '--image_dir', 'unused', '--ann_path', ann, '--dataset_name', 'mimic_cxr',
                '--max_seq_length', '100', '--threshold', '10', '--batch_size', '16', '--load', load,
                '--beam_size', str(beam), '--sample_method', 'beam_search', '--block_trigrams', '1']
    try:
        args = parse_agrs()
    finally:
        sys.argv = old
    args.visual_extractor_pretrained = False
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--load', default='data/model_mimic_cxr.pth')
    p.add_argument('--mimic_ann', default='data/mimic_cxr/annotation.json')
    p.add_argument('--images', default='data/mimic_cxr/images/')
    p.add_argument('--split', default='test')
    p.add_argument('--max', type=int, default=1000)
    p.add_argument('--beam_size', type=int, default=3, help='test_mimic_cxr.sh uses 2; 3 matches the IU runs')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='../results/mimic_test_check.json')
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.Tensor.cuda = lambda self, *x, **k: self   # R2Gen beam search calls .cuda()

    from models.r2gen import R2GenModel
    from modules.tokenizers import Tokenizer
    import clinical_spans as cs
    from pycocoevalcap.bleu.bleu import Bleu

    args = mimic_args(a.load, a.mimic_ann, a.beam_size)
    tok = Tokenizer(args)
    model = R2GenModel(args, tok)
    ckpt = torch.load(a.load, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    state = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state.items()}
    rows = state['encoder_decoder.model.tgt_embed.0.lut.weight'].shape[0]
    print(f'vocabulary {len(tok.idx2token)} + 1 = {len(tok.idx2token) + 1}; checkpoint embedding rows {rows}')
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    allrows = json.loads(Path(a.mimic_ann).read_text())[a.split]
    iu = [e for e in allrows if (Path(a.images) / e['image_path'][0]).is_file()]
    print(f'{len(iu)} of {len(allrows)} {a.split} images are on disk')
    if len(iu) > a.max:
        iu = random.Random(a.seed).sample(iu, a.max)
    refs = [tok.clean_report(e['report']) for e in iu]
    keep = [i for i, r in enumerate(refs) if r.replace('.', '').strip()]
    iu, refs = [iu[i] for i in keep], [refs[i] for i in keep]
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    perm = list(range(len(iu)))
    random.Random(a.seed).shuffle(perm)
    donor = {perm[i]: perm[(i + 1) % len(perm)] for i in range(len(perm))}   # single cycle: nobody keeps own image

    def run(order):
        out = []
        for i in range(0, len(order), a.batch_size):
            imgs = [tf(Image.open(Path(a.images) / iu[j]['image_path'][0]).convert('RGB'))
                    for j in order[i:i + a.batch_size]]
            with torch.no_grad():
                seq = model(torch.stack(imgs).to(device), mode='sample')   # sample mode returns the token ids only
            out += tok.decode_batch(seq.cpu().numpy())
            print(f'\r{len(out)}/{len(order)}', end='', flush=True)
        print()
        return out

    correct = run(list(range(len(iu))))
    shuffled = run([donor[i] for i in range(len(iu))])

    def stats(gens):
        c = Counter(gens)
        tot = {'tp_abn': 0, 'fn_abn': 0, 'fp_abn': 0}
        pos = Counter()
        for r, g in zip(refs, gens):
            gl = cs.report_labels(g.split())
            res = cs.compare_labels(cs.report_labels(r.split()), gl)
            for k in tot:
                tot[k] += res[k]
            pos.update(k for k, v in gl.items() if v in ('positive', 'uncertain'))
        bleu, _ = Bleu(4).compute_score({i: [r] for i, r in enumerate(refs)}, {i: [g] for i, g in enumerate(gens)})
        ref_abn = sum(any(v in ('positive', 'uncertain') for v in cs.report_labels(r.split()).values()) for r in refs)
        gen_abn = sum(any(v in ('positive', 'uncertain') for v in cs.report_labels(g.split()).values()) for g in gens)
        f1 = 2 * tot['tp_abn'] / max(2 * tot['tp_abn'] + tot['fp_abn'] + tot['fn_abn'], 1)
        extra = {'abnormal_f1': f1, 'references_with_abnormal_finding': ref_abn, 'generated_with_abnormal_finding': gen_abn}
        return {**extra, 'distinct_reports': len(c), 'most_frequent_count': c.most_common(1)[0][1],
                'abnormal_recall': f"{tot['tp_abn']}/{tot['tp_abn'] + tot['fn_abn']}",
                'abnormal_false_positives': tot['fp_abn'], 'bleu4_vs_reference': bleu[3],
                'positive_findings_generated': dict(pos.most_common())}

    res = {'images': len(iu), 'beam_size': a.beam_size, 'correct': stats(correct),
           'shuffled': stats(shuffled),
           'reports_changed_by_shuffle': sum(x != y for x, y in zip(correct, shuffled)),
           'examples': [{'reference': refs[i], 'generated': correct[i]} for i in range(5)]}
    findings_same = sum(cs.report_labels(x.split()) == cs.report_labels(y.split()) for x, y in zip(correct, shuffled))
    res['same_findings_after_shuffle'] = findings_same
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({**res, 'generated_correct': correct, 'generated_shuffled': shuffled}, indent=1))
    print(json.dumps(res, indent=1))
    print('saved', Path(a.out).resolve())


if __name__ == '__main__':
    main()
