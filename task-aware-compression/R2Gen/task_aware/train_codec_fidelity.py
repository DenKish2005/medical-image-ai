"""Task-aware codec trained to PRESERVE the uncompressed R2Gen report.

Same trainer as train_codec_distill.py, with one change: the teacher-forced target of every
train/val study is the report R2Gen generates from the UNCOMPRESSED feature
(make_teacher_reports.py), not the radiologist's reference report. So

    L = mu*rate + lambda_f*feature_NMSE
        + lambda_t  * span-weighted NLL of the uncompressed report      (spans found in that report)
        + lambda_kd * KL(p_uncompressed || p_compressed) along that report

Every text term is zero-cost for a lossless codec, so the codec cannot "beat" the uncompressed
model; it can only get closer to it. The reference report never enters training.
Model selection (best.pt) uses the same quantities on the validation split.

    python task_aware/train_codec_fidelity.py --teacher_reports ../results/ta_full/teacher_reports.json \
        --stage task --init ../results/ta_full/B_c64_r500/best.pt --out_dir ../results/ta_full/F_c64_r500 \
        --feature_dir ../results/ta_features --rate_mode target --target_bits 500 --epochs 40 --lr 5e-5 \
        --lambda_f 1 --lambda_t 1 --lambda_kd 100 --alpha 2 -- <R2Gen args>
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pop_option(argv, name):
    """Remove '--name value' from the script part of argv (before '--') and return the value."""
    end = argv.index('--') if '--' in argv else len(argv)
    for i in range(end):
        if argv[i] == name:
            if i + 1 >= end:
                raise SystemExit(f'{name} needs a value')
            value = argv[i + 1]
            del argv[i:i + 2]
            return value
        if argv[i].startswith(name + '='):
            value = argv[i].split('=', 1)[1]
            del argv[i]
            return value
    return None


def peek_option(argv, name, default=None):
    end = argv.index('--') if '--' in argv else len(argv)
    for i in range(end):
        if argv[i] == name and i + 1 < end:
            return argv[i + 1]
        if argv[i].startswith(name + '='):
            return argv[i].split('=', 1)[1]
    return default


def substitute_reports(ann, teacher, splits=('train', 'val'), required=None):
    """Copy of the annotation whose train/val 'report' fields are the teacher reports.

    required: {split: ids that the trainer will use}; every one of them must have a teacher report.
    Without it, every annotation entry of the split must have one (IU X-Ray behaviour)."""
    ann = copy.deepcopy(ann)
    for split in splits:
        if split not in teacher:
            raise ValueError(f'teacher file has no "{split}" reports')
        need = required[split] if required is not None else [e['id'] for e in ann[split]]
        missing = [i for i in need if i not in teacher[split]]
        if missing:
            raise ValueError(f'{len(missing)} {split} studies have no teacher report, e.g. {missing[:3]}')
        for e in ann[split]:
            if e['id'] in teacher[split]:
                e['report'] = teacher[split][e['id']]
    return ann


def main():
    path = pop_option(sys.argv, '--teacher_reports')
    if not path:
        raise SystemExit('--teacher_reports is required (run make_teacher_reports.py first)')
    out_dir = None
    for i, v in enumerate(sys.argv):
        if v == '--out_dir' and i + 1 < len(sys.argv):
            out_dir = sys.argv[i + 1]
        elif v.startswith('--out_dir='):
            out_dir = v.split('=', 1)[1]

    import ta_common as tc
    import train_codec_distill as distill

    blob = json.loads(Path(path).read_text(encoding='utf-8'))
    teacher = blob['reports']
    original_load = tc.load_annotation

    def load_annotation(args):
        sha = blob.get('r2gen_checkpoint_sha256')
        if sha and sha != tc.sha256_file(args.load):
            raise ValueError('Teacher reports were generated with a different R2Gen checkpoint.')
        dec = blob.get('decoding', {})
        diff = {k: (v, getattr(args, k, None)) for k, v in dec.items() if getattr(args, k, None) != v}
        if diff:
            raise ValueError(f'Decoding arguments differ from the teacher reports: {diff}')
        required = None
        fdir = peek_option(sys.argv, '--feature_dir')
        if fdir:                                    # ids the trainer will actually use
            required = {}
            for split, opt in (('train', '--max_train'), ('val', '--max_val')):
                ids = json.loads((Path(fdir) / f'{split}_ids.json').read_text())
                n = int(peek_option(sys.argv, opt, 0) or 0)
                required[split] = ids[:n] if n else ids
        return substitute_reports(original_load(args), teacher, required=required)

    # The tokenizer vocabulary is still built from the original annotation file (args.ann_path);
    # only the targets handed to the trainer change.
    tc.load_annotation = load_annotation
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tc.json_save(Path(out_dir) / 'fidelity_targets.json', {
            'targets': 'report generated by frozen R2Gen from uncompressed features (not the reference report)',
            'teacher_reports': str(Path(path).resolve()), 'teacher_sha256': tc.sha256_file(path),
            'note': 'In config.json, "weighted_NLL", "KL" and the "reference prefix" all refer to these targets.'})
    distill.main()


if __name__ == '__main__':
    main()
