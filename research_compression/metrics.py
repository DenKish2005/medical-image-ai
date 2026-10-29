"""Patient-grouped AUROC confidence intervals and validation-fixed sensitivity."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _auc(y, p, valid):
    y,p = np.asarray(y)[valid],np.asarray(p)[valid]
    return float(roc_auc_score(y,p)) if len(set(y.tolist())) == 2 else float('nan')


def compute_metrics(target, probability, valid, diseases):
    y = np.asarray(target, dtype=float)
    p = np.asarray(probability, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if y.shape != p.shape or y.shape != mask.shape or y.ndim != 2 or y.shape[1] != len(diseases):
        raise ValueError("Metrics require aligned [N,num_diseases] arrays")
    scores = {name: _auc(y[:,i],p[:,i],mask[:,i]) for i,name in enumerate(diseases)}
    finite = [a for a in scores.values() if np.isfinite(a)]
    return {"per_disease_auroc": scores,
            "macro_auroc": float(np.mean(finite)) if finite else float('nan'),
            "prevalence": {name:float(y[mask[:,i],i].mean()) if mask[:,i].any() else float('nan')
                           for i,name in enumerate(diseases)}}


def validation_thresholds(y, probs, valid, specificity=0.9):
    """Choose disease-specific thresholds once on validation only."""
    if not 0 < specificity < 1:
        raise ValueError("specificity must be in (0,1)")
    y,p,v = np.asarray(y),np.asarray(probs),np.asarray(valid, dtype=bool)
    out=[]
    for col in range(y.shape[1]):
        yy,pp=y[v[:,col],col],p[v[:,col],col]
        if len(set(yy.tolist())) != 2:
            out.append(float('nan'))
            continue
        fpr,tpr,thresholds=roc_curve(yy,pp,drop_intermediate=False)
        choices=np.where((fpr <= 1-specificity+1e-12)&np.isfinite(thresholds))[0]
        out.append(float(thresholds[choices[np.argmax(tpr[choices])]]) if len(choices) else float('nan'))
    return out


def sensitivity_at_thresholds(y, p, valid, diseases, thresholds):
    y,p,v=np.asarray(y),np.asarray(p),np.asarray(valid,dtype=bool)
    result={}
    for i,name in enumerate(diseases):
        pos=v[:,i] & (y[:,i]==1)
        threshold=thresholds[i]
        result[name] = float((p[pos,i]>=threshold).mean()) if pos.any() and np.isfinite(threshold) else float('nan')
    return result


def patient_bootstrap_auc(y,p,valid,patients,diseases, *, n_boot=1000,seed=42):
    """Resample unique patients, retaining all of each sampled patient's images."""
    ids=np.asarray(patients,dtype=str)
    unique=np.unique(ids)
    rng=np.random.default_rng(seed)
    per={name:[] for name in diseases}
    per['macro']=[]
    for _ in range(n_boot):
        draw=rng.choice(unique,size=len(unique),replace=True)
        indices=np.concatenate([np.flatnonzero(ids==u) for u in draw])
        res=compute_metrics(np.asarray(y)[indices],np.asarray(p)[indices],np.asarray(valid)[indices],diseases)
        for name,score in res['per_disease_auroc'].items():
            if np.isfinite(score):per[name].append(score)
        if np.isfinite(res['macro_auroc']):per['macro'].append(res['macro_auroc'])
    return {key:[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))] if vals else [None,None]
            for key,vals in per.items()}


def paired_patient_bootstrap_delta(y, first, second, valid, patients, diseases, *, n_boot=1000, seed=42):
    """Paired macro-AUROC difference: second minus first, same sampled patients."""
    ids=np.asarray(patients,dtype=str); unique=np.unique(ids)
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(n_boot):
        selected=rng.choice(unique,size=len(unique),replace=True)
        indices=np.concatenate([np.flatnonzero(ids==i) for i in selected])
        a=compute_metrics(np.asarray(y)[indices],np.asarray(first)[indices],np.asarray(valid)[indices],diseases)['macro_auroc']
        b=compute_metrics(np.asarray(y)[indices],np.asarray(second)[indices],np.asarray(valid)[indices],diseases)['macro_auroc']
        if np.isfinite(a) and np.isfinite(b):diffs.append(b-a)
    return {"delta_macro_auroc":float(np.mean(diffs)) if diffs else None,
            "ci95":[float(np.quantile(diffs,.025)),float(np.quantile(diffs,.975))] if diffs else [None,None]}
