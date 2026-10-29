import json

import numpy as np
import pandas as pd
import pytest

from research_compression.data import read_manifest, label_arrays
from research_compression.metrics import (compute_metrics,patient_bootstrap_auc,
                                          validation_thresholds,sensitivity_at_thresholds,
                                          paired_patient_bootstrap_delta)
from research_compression.deployment import simulate_latency,latency_summary


def test_patient_leakage_detected(tmp_path):
    file=tmp_path/'manifest.csv'
    pd.DataFrame([{'patient_id':'a','path':'1.png','split':'train','P':1},
                  {'patient_id':'a','path':'2.png','split':'test','P':0}]).to_csv(file,index=False)
    with pytest.raises(ValueError,match='LEAKAGE'):
        read_manifest(file,['P'],'ignore')


def test_uncertainty_label_policies(tmp_path):
    file=tmp_path/'manifest.csv'
    df=pd.DataFrame([{'patient_id':'a','path':'a.png','split':'train','P':-1},
                     {'patient_id':'b','path':'b.png','split':'val','P':1},
                     {'patient_id':'c','path':'c.png','split':'test','P':0}])
    df.to_csv(file,index=False)
    got=read_manifest(file,['P'],'ignore')
    y,mask=label_arrays(got,['P'],'ignore')
    assert bool(mask[0,0]) is False
    y,mask=label_arrays(got,['P'],'one')
    assert y[0,0]==1 and bool(mask[0,0]) is True


def test_bootstrap_and_fixed_threshold():
    y=np.array([[0],[1],[0],[1],[0],[1]],float)
    p=np.array([[.1],[.9],[.2],[.8],[.15],[.75]],float)
    valid=np.ones_like(y,dtype=bool)
    patients=['p1','p1','p2','p2','p3','p3']
    scores=compute_metrics(y,p,valid,['D'])
    assert scores['macro_auroc']==1.0
    thresholds=validation_thresholds(y,p,valid,specificity=.9)
    assert sensitivity_at_thresholds(y,p,valid,['D'],thresholds)['D']==1.0
    ci=patient_bootstrap_auc(y,p,valid,patients,['D'],n_boot=32)
    assert ci['macro']==[1.,1.]
    delta=paired_patient_bootstrap_delta(y,p,p,valid,patients,['D'],n_boot=32)
    assert delta['delta_macro_auroc']==0.0


def test_deployment_formula():
    result=simulate_latency(edge_ms=10,encode_ms=2,transmitted_bits=8000,
                            bandwidth_bits_per_sec=8000,decode_ms=3,cloud_ms=5)
    assert result==1020
    assert latency_summary([10,20,30])['p95_ms']>20
