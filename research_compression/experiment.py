"""Single-entry experiments: baseline training, codec training, real-byte probes, evaluation.

Run with `python -m research_compression.experiment <command> --config configs/cxr_research.json`.
Never use test images to select hyperparameters or set operating thresholds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .allocation import SpatialRiskHeads, allocate_bits, actual_rate, risk_head_loss, teacher_spatial_targets
from .codec import ConditionalFeatureCodec, encode_symbols, fp16_reference_bits, decode_symbols
from .data import CXRDataset, cxr_collate, boxes_to_mask, patient_manifest_sha, read_manifest, label_arrays
from .losses import compression_loss, masked_weighted_bce
from .metrics import (compute_metrics, patient_bootstrap_auc, paired_patient_bootstrap_delta,
                      validation_thresholds, sensitivity_at_thresholds)
from .models import DenseNet121Split


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load_cfg(path):
    with open(path,encoding='utf-8') as f:cfg=json.load(f)
    required=['manifest','image_root','diseases','uncertainty_policy','output_dir']
    for key in required:
        if not cfg.get(key):raise ValueError(f"Config requires {key}")
    if not isinstance(cfg['diseases'],list) or not all(isinstance(d,str) for d in cfg['diseases']):
        raise ValueError('Config diseases must be an array of label column names')
    if any(not 0<f<=1 for f in cfg.get('rate_fractions',[.5,.25,.1,.05])):
        raise ValueError('rate_fractions must be in (0,1]')
    read_manifest(cfg['manifest'],cfg['diseases'],cfg['uncertainty_policy'])
    return cfg


def batches(cfg, split, shuffle=False):
    ds=CXRDataset(cfg['manifest'],split=split,root=cfg['image_root'],
                  diseases=cfg['diseases'],uncertainty_policy=cfg['uncertainty_policy'],
                  image_size=int(cfg.get('image_size',224)),use_roi=True)
    return DataLoader(ds,batch_size=int(cfg.get('batch_size',4)),shuffle=shuffle,
                      num_workers=int(cfg.get('num_workers',0)),collate_fn=cxr_collate)


def device_for(cfg):
    return torch.device(cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu'))


def _sync(device):
    if device.type=='cuda':torch.cuda.synchronize()


def audit(cfg):
    df=read_manifest(cfg['manifest'],cfg['diseases'],cfg['uncertainty_policy'])
    payload={"manifest_sha256":patient_manifest_sha(cfg['manifest']),
             "policy":cfg['uncertainty_policy'], "counts":df.groupby('split').size().to_dict(),
             "patients":df.groupby('split')['patient_id'].nunique().to_dict()}
    for col in cfg['diseases']:
        payload.setdefault('labels',{})[col]={str(key):int(value) for key,value in df[col].value_counts(dropna=False).items()}
    return payload


def _class_weights(cfg, device):
    """Training-only prevalence weights; safe if a class has no positives."""
    if not cfg.get('use_class_weights',False):
        return None
    frame=read_manifest(cfg['manifest'],cfg['diseases'],cfg['uncertainty_policy'])
    frame=frame.loc[frame['split']=='train']
    y,valid=label_arrays(frame,cfg['diseases'],cfg['uncertainty_policy'])
    positives=((y==1)&valid).sum(axis=0)
    negatives=((y==0)&valid).sum(axis=0)
    weights=np.where(positives>0,negatives/np.maximum(positives,1),1.0)
    return torch.tensor(np.clip(weights,0.25,20.0),dtype=torch.float32,device=device)


def _baseline(cfg, device, checkpoint=True):
    model=DenseNet121Split(len(cfg['diseases']),split=cfg.get('split','denseblock2'),
                           pretrained=bool(cfg.get('pretrained_imagenet',False)) and not checkpoint)
    model.to(device)
    if checkpoint:
        path=Path(cfg.get('baseline_checkpoint',''))
        if not path.is_file():
            raise FileNotFoundError('Train baseline first, then set baseline_checkpoint explicitly')
        state=torch.load(path,map_location=device,weights_only=True)
        if state['diseases'] != cfg['diseases'] or state['split']!=model.split:
            raise ValueError('Baseline checkpoint labels or split layer differ from config')
        model.load_state_dict(state['model'],strict=True)
    return model


def train_baseline(cfg):
    device=device_for(cfg); model=_baseline(cfg,device,checkpoint=False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(cfg.get('baseline_lr',1e-4)))
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    best=float('inf'); path=output/'baseline.pt';history=[]
    pos_weights=_class_weights(cfg,device)
    for epoch in range(int(cfg.get('baseline_epochs',2))):
        for split in ('train','val'):
            model.train(split=='train'); total,count=0.,0
            for batch in batches(cfg,split,shuffle=split=='train'):
                x=batch['image'].to(device); y=batch['label'].to(device); valid=batch['valid'].to(device)
                with torch.set_grad_enabled(split=='train'):
                    loss=masked_weighted_bce(model(x),y,valid,positive_weights=pos_weights)
                    if split=='train':
                        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                total+=float(loss.detach())*len(x);count+=len(x)
            value=total/max(count,1)
            history.append({'epoch':epoch+1,'split':split,'loss':value})
            if split=='val' and value<best:
                best=value
                torch.save({'model':model.state_dict(),'diseases':cfg['diseases'],
                            'split':model.split, 'seed':cfg.get('seed',42)},path)
    (output/'baseline_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return {'baseline_checkpoint':str(path),'baseline_sha256':sha256(path),'history':history}


def _fit_modules(cfg, model, device):
    model.eval();
    for p in model.parameters():p.requires_grad_(False)
    with torch.no_grad():
        image=torch.zeros(1,3,int(cfg.get('image_size',224)),int(cfg.get('image_size',224)),device=device)
        shape=model.features(image).shape
    codec=ConditionalFeatureCodec(shape[1],latent_channels=int(cfg.get('latent_channels',64))).to(device)
    scorer=SpatialRiskHeads(shape[1]).to(device)
    return codec,scorer


def _rate_batch(cfg, model, codec, scorer, feature, fraction, mode, seed):
    target=max(1,round(fp16_reference_bits(feature[0])*fraction))
    with torch.no_grad():
        imp,unc=scorer(feature)
        depths=allocate_bits(imp,unc,channels=codec.latent_channels,target_bits=target,
                             mode=mode,alpha=float(cfg.get('importance_weight',.5)),seed=seed)
    return target,depths


def train_codec(cfg):
    device=device_for(cfg);model=_baseline(cfg,device)
    codec,scorer=_fit_modules(cfg,model,device)
    optimizer=torch.optim.AdamW(list(codec.parameters())+list(scorer.parameters()),
                                lr=float(cfg.get('codec_lr',1e-3)))
    rates=list(cfg.get('rate_fractions',[.5,.25,.1,.05]));epochs=int(cfg.get('codec_epochs',4))
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    path=output/'conditional_codec.pt';history=[]; best=float('inf')
    pos_weights=_class_weights(cfg,device)
    for epoch in range(epochs):
        for split in ('train','val'):
            codec.train(split=='train');scorer.train(split=='train')
            total,count=0.,0
            for iteration,batch in enumerate(batches(cfg,split,shuffle=split=='train')):
                image=batch['image'].to(device);targets=batch['label'].to(device);valid=batch['valid'].to(device)
                with torch.no_grad():
                    feature=model.features(image);teacher=model.predict_from_features(feature)
                fraction=float(rates[(epoch+iteration)%len(rates)] if split=='val' else random.choice(rates))
                target,depths=_rate_batch(cfg,model,codec,scorer,feature,fraction,'combined',seed=epoch)
                with torch.set_grad_enabled(split=='train'):
                    decoded=codec(feature,depths,rate_fraction=fraction)
                    student=model.predict_from_features(decoded.reconstructed)
                    roi=boxes_to_mask(batch['boxes'],feature.shape[2],feature.shape[3],device) if cfg.get('use_roi_loss',False) and split=='train' else None
                    loss,terms=compression_loss(student,teacher,targets,valid,feature,decoded.reconstructed,
                        decoded.estimated_bits,rate_target_bits=target,roi_mask=roi,
                        positive_weights=pos_weights,
                        reconstruction_weight=float(cfg.get('reconstruction_weight',.05)),
                        distill_weight=float(cfg.get('distill_weight',.5)),
                        lesion_weight=float(cfg.get('lesion_weight',0)),
                        rate_weight=float(cfg.get('rate_weight',1e-6)),
                        over_budget_weight=float(cfg.get('over_budget_weight',1e-5)))
                    imp,unc=scorer(feature.detach())
                    if split=='train':
                        target_imp,target_unc=teacher_spatial_targets(feature,model.predict_from_features)
                        loss=loss+float(cfg.get('score_supervision_weight',0.1))*risk_head_loss(
                            imp,unc,target_imp,target_unc)
                        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                total+=float(loss.detach())*len(image);count+=len(image)
            value=total/max(count,1);history.append({'epoch':epoch+1,'split':split,'loss':value})
            if split=='val' and value<best:
                best=value
                torch.save({'codec':codec.state_dict(),'scorer':scorer.state_dict(),
                            'channels':codec.encoder.in_channels,'latent_channels':codec.latent_channels,
                            'baseline_sha256':sha256(cfg['baseline_checkpoint']),
                            'diseases':cfg['diseases'],'split':model.split,
                            'rate_fractions':rates},path)
    (output/'codec_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return {'codec_checkpoint':str(path),'codec_sha256':sha256(path),'history':history}


def _trained(cfg,device):
    model=_baseline(cfg,device)
    codec,scorer=_fit_modules(cfg,model,device)
    path=Path(cfg.get('codec_checkpoint',''))
    if not path.is_file():raise FileNotFoundError('Set codec_checkpoint to trained conditional_codec.pt')
    state=torch.load(path,map_location=device,weights_only=True)
    if (state['split']!=model.split or state['diseases']!=cfg['diseases'] or
        state['channels']!=codec.encoder.in_channels or state['latent_channels']!=codec.latent_channels or
        state['baseline_sha256']!=sha256(cfg['baseline_checkpoint'])):
        raise ValueError('Codec and baseline checkpoint/config mismatch')
    codec.load_state_dict(state['codec']);scorer.load_state_dict(state['scorer'])
    codec.eval();scorer.eval();model.eval()
    return model,codec,scorer


def _evaluate_once(cfg,model,codec,scorer,device, *, split, mode, fraction):
    rows=[];ds=batches(cfg,split,shuffle=False)
    for iteration,batch in enumerate(ds):
        image=batch['image'].to(device)
        with torch.no_grad():
            feature=model.features(image)
            reference=model.predict_from_features(feature).sigmoid()
            target,depths=_rate_batch(cfg,model,codec,scorer,feature,fraction,mode,seed=iteration+int(cfg.get('seed',42)))
            _sync(device); t0=time.perf_counter()
            messages=codec.encode(feature,depths,fraction,split=model.split,rate_id=f'{fraction:.6f}')
            _sync(device);encode_ms=(time.perf_counter()-t0)*1000/len(image)
            _sync(device);t0=time.perf_counter()
            recovered=codec.decode(messages,expected_split=model.split,device=device)
            _sync(device);decode_ms=(time.perf_counter()-t0)*1000/len(image)
            student=model.predict_from_features(recovered).sigmoid()
        for i,message in enumerate(messages):
            rate=actual_rate(message,target)
            row={'patient_id':batch['patient_id'][i], 'path':batch['path'][i],
                 'split':split,'mode':mode,'rate_fraction':fraction,
                 'reference_fp16_payload_bits':fp16_reference_bits(feature[i]),
                 **rate,'encode_ms_batch_average':encode_ms,'decode_ms_batch_average':decode_ms}
            for j,name in enumerate(cfg['diseases']):
                row[f'gt_{name}']=float(batch['label'][i,j]);row[f'valid_{name}']=bool(batch['valid'][i,j])
                row[f'compressed_{name}']=float(student[i,j]);row[f'reference_{name}']=float(reference[i,j])
            rows.append(row)
    return rows


def _array_from_rows(rows,diseases,prefix):
    return np.array([[row[f'{prefix}_{name}'] for name in diseases] for row in rows])


def evaluate(cfg):
    device=device_for(cfg);model,codec,scorer=_trained(cfg,device)
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    summary={};diseases=cfg['diseases']
    for mode in cfg.get('eval_modes',['uniform','random','saliency','uncertainty','combined']):
        for fraction in cfg.get('rate_fractions',[.5,.25,.1,.05]):
            # Validation thresholds MUST be selected before reading test results.
            val=_evaluate_once(cfg,model,codec,scorer,device,split='val',mode=mode,fraction=fraction)
            valy=_array_from_rows(val,diseases,'gt')
            valp=_array_from_rows(val,diseases,'compressed')
            valv=_array_from_rows(val,diseases,'valid').astype(bool)
            thresholds=validation_thresholds(valy,valp,valv,specificity=float(cfg.get('specificity',.9)))
            reference_thresholds=validation_thresholds(
                valy,_array_from_rows(val,diseases,'reference'),valv,
                specificity=float(cfg.get('specificity',.9)))
            rows=_evaluate_once(cfg,model,codec,scorer,device,split='test',mode=mode,fraction=fraction)
            y=_array_from_rows(rows,diseases,'gt')
            p=_array_from_rows(rows,diseases,'compressed')
            valid=_array_from_rows(rows,diseases,'valid').astype(bool)
            ref_p=_array_from_rows(rows,diseases,'reference')
            metrics=compute_metrics(y,p,valid,diseases)
            metrics['reference']=compute_metrics(y,ref_p,valid,diseases)
            metrics['paired_macro_auroc_delta_vs_reference']=paired_patient_bootstrap_delta(
                y,ref_p,p,valid,[r['patient_id'] for r in rows],diseases,
                n_boot=int(cfg.get('bootstrap_samples',200)),seed=int(cfg.get('seed',42)))
            metrics['auroc_ci95']=patient_bootstrap_auc(y,p,valid,[r['patient_id'] for r in rows],diseases,
                                  n_boot=int(cfg.get('bootstrap_samples',200)),seed=int(cfg.get('seed',42)))
            metrics['sensitivity_at_validation_fixed_specificity']=sensitivity_at_thresholds(y,p,valid,diseases,thresholds)
            metrics['validation_thresholds']=dict(zip(diseases,thresholds))
            metrics['reference_sensitivity_at_validation_fixed_specificity']=sensitivity_at_thresholds(
                y,ref_p,valid,diseases,reference_thresholds)
            metrics['reference_validation_thresholds']=dict(zip(diseases,reference_thresholds))
            metrics['mean_actual_bits']=float(np.mean([r['actual_bits'] for r in rows]))
            metrics['mean_target_bits']=float(np.mean([r['target_bits'] for r in rows]))
            metrics['budget_met_fraction']=float(np.mean([r['budget_met'] for r in rows]))
            metrics['mean_encode_ms_batch_average']=float(np.mean([r['encode_ms_batch_average'] for r in rows]))
            metrics['mean_decode_ms_batch_average']=float(np.mean([r['decode_ms_batch_average'] for r in rows]))
            key=f'{mode}_{fraction:.4f}'
            summary[key]=metrics
            with open(output/f'predictions_{key}.csv','w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'metrics.json').write_text(json.dumps(_json_safe(summary),indent=2,allow_nan=False),encoding='utf-8')
    return summary


def probe(cfg):
    """Actual serializations for unlearned uniform 2/4/6/8-bit feature maps."""
    device=device_for(cfg);model=_baseline(cfg,device);model.eval()
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with torch.no_grad():
        for batch in batches(cfg,'val'):
            feature=model.features(batch['image'].to(device)).float().cpu().numpy()
            for i,f in enumerate(feature):
                for depth in (2,4,6,8):
                    message=encode_symbols(f,np.full(f.shape[1:],depth,dtype=np.uint8),split=model.split,rate_id=f'probe-{depth}')
                    recovered,_=decode_symbols(message,expected_split=model.split)
                    rows.append({'patient_id':batch['patient_id'][i],'split':model.split,
                                 'bits_per_symbol':depth, 'actual_bytes':len(message),
                                 'fp16_payload_bytes':f.size*2,
                                 'mse':float(np.mean((f-recovered)**2))})
    with open(output/'uniform_probe.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'probe_count':len(rows),'path':str(output/'uniform_probe.csv')}


def _json_safe(value):
    if isinstance(value, dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [_json_safe(v) for v in value]
    if isinstance(value, (float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('audit','train-baseline','train-codec','probe','evaluate'))
    parser.add_argument('--config',required=True)
    args=parser.parse_args(argv)
    cfg=load_cfg(args.config);seed_everything(int(cfg.get('seed',42)))
    funcs={'audit':audit,'train-baseline':train_baseline,'train-codec':train_codec,
           'probe':probe,'evaluate':evaluate}
    print(json.dumps(_json_safe(funcs[args.command](cfg)),indent=2,allow_nan=False,default=str))

if __name__ == '__main__':main()
