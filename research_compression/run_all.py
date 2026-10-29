"""One-command research pipeline with reproducibility metadata.

Example: python -m research_compression.run_all --config configs/cxr_research.example.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy
import torch

from .experiment import (audit, evaluate, load_cfg, probe, seed_everything,
                         sha256, train_baseline, train_codec, _json_safe)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args(argv)
    cfg=load_cfg(args.config)
    seed_everything(int(cfg.get('seed',42)))
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    info={'created_utc':datetime.now(timezone.utc).isoformat(),
          'config_sha256':sha256(args.config),
          'manifest_sha256':sha256(cfg['manifest']),
          'seed':int(cfg.get('seed',42)),
          'python':sys.version,'platform':platform.platform(),
          'torch':torch.__version__,'numpy':numpy.__version__,
          'cuda_available':torch.cuda.is_available(),
          'cuda_device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'experiment_config':cfg}
    info['audit']=audit(cfg)
    baseline=train_baseline(cfg)
    cfg['baseline_checkpoint']=baseline['baseline_checkpoint']
    info['baseline_checkpoint_sha256']=baseline['baseline_sha256']
    info['probe']=probe(cfg)
    codec=train_codec(cfg)
    cfg['codec_checkpoint']=codec['codec_checkpoint']
    info['codec_checkpoint_sha256']=codec['codec_sha256']
    info['metrics_file']=str(out/'metrics.json')
    evaluate(cfg)
    (out/'reproducibility.json').write_text(
        json.dumps(_json_safe(info),indent=2,allow_nan=False,default=str),encoding='utf-8')
    print('Complete:',out/'reproducibility.json')

if __name__=='__main__':main()
