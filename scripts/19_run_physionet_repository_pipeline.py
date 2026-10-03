"""Run the repository federated pipeline on fixed PhysioNet recording splits.

The only adapter replaces the split chooser with a checked preassigned split.
Client models, poisoning, aggregation, metrics and saved predictions come from
the repository's run_federated_training implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from trustfed_incboost import federated_pipeline as pipeline
from trustfed_incboost.data.splitting import SplitResult
from trustfed_incboost.models.federated import aggregate_probabilities
from trustfed_incboost.models.baseline import predict_probabilities
from trustfed_incboost.evaluation.metrics import classification_metrics


def preassigned_split(X, y, method, seed, test_size, validation_size,
                      group_column=None, time_column=None):
    if method != 'preassigned' or group_column != 'prepared_split':
        raise ValueError('This adapter only accepts a checked preassigned split')
    labels=X[group_column].to_numpy()
    if set(labels)!={'train','validation','test'}:
        raise ValueError('Missing train/validation/test split')
    if X.groupby('record')['prepared_split'].nunique().gt(1).any():
        raise ValueError('Recording spans multiple splits')
    return SplitResult(*(np.flatnonzero(labels==s) for s in ('train','validation','test')),
                       method='preassigned')


def fast_macro_f1_threshold(y_true, positive_probability):
    # Same decision objective as the repository: highest validation macro F1.
    y=np.asarray(y_true,dtype=int)
    prob=np.asarray(positive_probability,dtype=float)
    candidates=np.unique(np.concatenate(([.5],np.linspace(.02,.98,193),prob)))
    order=np.argsort(prob,kind='stable')
    sorted_prob=prob[order]
    positives=np.cumsum(y[order],dtype=int)
    offsets=np.searchsorted(sorted_prob,candidates,side='left')
    tp=int(y.sum())-np.where(offsets>0,positives[np.maximum(offsets-1,0)],0)
    fp=(len(y)-offsets)-tp
    fn=int(y.sum())-tp
    tn=offsets-fn
    f1positive=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(tp),dtype=float),where=(2*tp+fp+fn)>0)
    f1negative=np.divide(2*tn,2*tn+fp+fn,out=np.zeros(len(tn),dtype=float),where=(2*tn+fp+fn)>0)
    macro=(f1positive+f1negative)/2
    # Original loop chooses first candidate on true improvement; near-equal
    # ties prefer threshold closest to .5.
    best=int(np.argmax(macro))
    return float(candidates[best])


def prepare_adapter(frame, manifest, output):
    if manifest.record.duplicated().any() or set(frame.record)!=set(manifest.record):
        raise ValueError('Invalid record manifest')
    split_map=manifest.set_index('record').split.to_dict()
    if any(split_map[r]!=s for r,s in zip(frame.record,frame.split)):
        raise ValueError('Split mismatch')
    keep=[c for c in frame.columns if c.startswith('ecg_') or c.startswith('pleth_')]
    # The ratio is an allowed numeric feature, history comparison is twin only.
    keep=sorted(set(keep)|{'pleth_ecg_diff_std_ratio'})
    adapted=frame[keep].copy()
    adapted['record']=frame.record.astype(str)
    adapted['prepared_split']=frame.split.astype(str)
    adapted['simulated_site']=frame.simulated_client.map(lambda n: f'site_{int(n)}' if pd.notna(n) else 'public')
    adapted['sensor_fault_label']=frame.sensor_fault_label.astype(int)
    output.parent.mkdir(parents=True,exist_ok=True)
    adapted.to_csv(output,index=False)
    return adapted


def make_config(root, name, poisoned):
    return {'_config_path':str(root/'configs'/'physionet_repository_adapter.yaml'),
        'project':{'name':name,'seed':42},
        'dataset':{'path':'data/processed/physionet_repository_adapter.csv',
                   'label_column':'sensor_fault_label','positive_label':1,
                   'drop_columns':[],'max_rows':None},
        'split':{'method':'preassigned','test_size':.2,'validation_size':.2,
                 'group_column':'prepared_split','time_column':None},
        'preprocessing':{'use_ica':False,'ica_components':None},
        'federated':{'n_clients':5,
                     'partition':{'method':'site_group','site_column':'simulated_site',
                                  'min_samples_per_client':100,'require_all_classes':True},
                     'aggregation':{'method':'trust_aware_v2',
                                    'performance_power':3.0,'calibration_strength':2.0,
                                    'consensus_strength':.10,'chance_margin':.02,
                                    'minimum_macro_f1':.25,'security_power':2.0,
                                    'consensus_z_cap':3.0,'minimum_trusted_clients':3,
                                    'max_weight':.40}},
        'poisoning':{'enabled':poisoned,'attack':'label_flip' if poisoned else 'none',
                     'strategy':'fixed','malicious_clients':[0] if poisoned else [],
                     'malicious_count':1,'flip_fraction':.8 if poisoned else 0.},
        'model':{'backend':'sklearn','class_weight':'balanced',
                 'params':{'max_iter':80,'max_leaf_nodes':15,'min_samples_leaf':30}},
        'evaluation':{'bootstrap_repetitions':50},
        'output':{'directory':'outputs/physionet_repository_pipeline'}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared',type=Path,default=Path('outputs/physionet_2015_prepared_v3'))
    ap.add_argument('--output',type=Path,default=Path('outputs/physionet_repository_comparison'))
    a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    frame=pd.read_csv(a.prepared/'sensor_windows.csv')
    manifest=pd.read_csv(a.prepared/'record_split_manifest.csv')
    summary=json.loads((a.prepared/'preparation_summary.json').read_text())
    if summary.get('preparation_version')!=3:
        raise ValueError('Run stage 6 preparation first; need prepared_v3')
    adapted=prepare_adapter(frame,manifest,root/'data/processed/physionet_repository_adapter.csv')
    pipeline.make_split=preassigned_split
    pipeline.select_binary_threshold=fast_macro_f1_threshold
    output={'source_sha256':summary['source_sha256'],
            'method':'repository run_federated_training with preassigned recording split and sklearn backend',
            'note':'This runs original trust-aware-v2 aggregation and poisoning; XGBoost paper backend remains to be run.',
            'runs':{}}
    for name,poisoned in [('clean',False),('label_flip_client_0',True)]:
        config=make_config(root,'physionet_repository_'+name,poisoned)
        run=pipeline.run_federated_training(config)
        metrics=json.loads((run/'metrics.json').read_text())
        rows=pd.read_csv(run/'test_predictions.csv')
        ids=rows.row_index.to_numpy(dtype=int)
        expected=np.flatnonzero(adapted.prepared_split.to_numpy()=='test')
        if not np.array_equal(ids,expected) or not np.array_equal(rows.y_true,frame.iloc[ids].sensor_fault_label):
            raise AssertionError('Saved predictions do not align with original held-out rows')
        bundle=joblib.load(run/'federated_ensemble.joblib')
        val_ids=np.flatnonzero(adapted.prepared_split.to_numpy()=='validation')
        xval=bundle['preprocessor'].transform(adapted.iloc[val_ids])
        component=[predict_probabilities(model,xval) for model in bundle['client_models']]
        valp=aggregate_probabilities(component,bundle['aggregation_weights'])[:,1]
        testp=rows.probability_1.to_numpy()
        yval=frame.iloc[val_ids].sensor_fault_label.to_numpy()
        ytest=rows.y_true.to_numpy()
        # The tolerance is fixed by stage-6 validation, never chosen on test.
        replay_cutoff=.01
        gateval=(frame.iloc[val_ids].history_robust_error.to_numpy()<=replay_cutoff).astype(float)
        gatetest=(frame.iloc[ids].history_robust_error.to_numpy()<=replay_cutoff).astype(float)
        fusedval=np.maximum(valp,gateval)
        fusedtest=np.maximum(testp,gatetest)
        threshold=fast_macro_f1_threshold(yval,fusedval)
        fused_metrics=classification_metrics(ytest,np.column_stack([1-fusedtest,fusedtest]),threshold,['0','1'])
        valmetrics=classification_metrics(yval,np.column_stack([1-fusedval,fusedval]),threshold,['0','1'])
        output['runs'][name]={
            'repository_run_directory':str(run),'poisoning_manifest':json.loads((run/'poisoning_manifest.json').read_text()),
            'repository_detector_test':metrics['test'],
            'detector_client_weights':[c['aggregation_weight'] for c in metrics['clients']],
            'detector_plus_replay_validation':valmetrics,
            'detector_plus_replay_test':fused_metrics,
            'detector_plus_replay_test_recall':float(np.mean((fusedtest>=threshold)[frame.iloc[ids].fault.to_numpy()=='replay'])),
            'fusion_threshold':threshold}
        print(name,'repository macro F1',round(metrics['test']['macro_f1'],4),
              'plus replay gate macro F1',round(fused_metrics['macro_f1'],4),flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Saved:',a.output/'results.json')

if __name__=='__main__':
    main()
