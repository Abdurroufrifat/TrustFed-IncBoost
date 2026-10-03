"""Evaluate replay matching and repository trust-aware aggregation on simulated clients.

Synthetic faults and simulated client poisoning; no hospital or cyberattack claims.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, roc_auc_score)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip', type=Path, default=Path('data/raw/physionet_2015/training.zip'))
    p.add_argument('--prepared', type=Path, default=Path('outputs/physionet_2015_prepared_v2'))
    p.add_argument('--output', type=Path, default=Path('outputs/physionet_2015_replay_poisoning'))
    a = p.parse_args()
    helpers = load_module(Path(__file__).with_name('15_run_physionet_federated_twin.py'), 'study_helpers')
    prep = load_module(Path(__file__).with_name('14_prepare_physionet_sensor_study.py'), 'prep_helpers')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    try:
        from trustfed_incboost.models.federated import aggregation_weights
    except ImportError as exc:
        raise RuntimeError('Repository src/trustfed_incboost missing; extract into project root') from exc
    summary = json.loads((a.prepared/'preparation_summary.json').read_text())
    if summary.get('preparation_version') != 2 or helpers.source_hash(a.zip) != summary['source_sha256']:
        raise ValueError('Expected stage-4 v2 prepared data from the same training.zip')
    frame = pd.read_csv(a.prepared/'sensor_windows.csv')
    manifest = pd.read_csv(a.prepared/'record_split_manifest.csv')
    if manifest.record.duplicated().any() or set(frame.record) != set(manifest.record):
        raise ValueError('Invalid recording manifest')
    source_splits = manifest.set_index('record').split.to_dict()
    if any(source_splits[rec] != split for rec,split in zip(frame.record,frame.split)):
        raise ValueError('Cross-split recording')
    forbidden = {'sensor_fault_label','fault','clinical_alarm_true_metadata','record','split',
                 'simulated_client','window_start_seconds','history_match_error'}
    features = sorted(set(frame.columns)-forbidden)
    if (frame[features].isna().any(axis=1) & (frame.fault != 'dropout')).any():
        raise ValueError('Unexpected missing signal features')
    frame[features] = frame[features].fillna(0.)
    train = frame.loc[frame.split=='train']
    val = frame.loc[frame.split=='validation']
    test = frame.loc[frame.split=='test']
    yv, yt = val.sensor_fault_label.to_numpy(), test.sensor_fault_label.to_numpy()
    with ZipFile(a.zip) as archive:
        refs = helpers.baseline_features(archive,manifest.record,prep)
    benign = train.loc[train.sensor_fault_label==0]
    values = benign[helpers.TWIN_COLS].to_numpy()
    reference = np.stack([refs[r] for r in benign.record])
    scale = np.maximum(np.quantile(np.abs(values-reference),.9,axis=0),.01)
    distribution = np.sort(helpers.twin_residuals(benign,refs,scale))
    # Train-clean waveforms fix a replay matching tolerance; this parameter
    # never uses held-out labels. Replay exemplars are exactly prior waveform
    # windows in the synthetic generator. Real-world approximate replay is untested.
    replay_tolerance = float(min(.02,np.quantile(benign.history_match_error,.001)))
    def twin_probability(part, with_replay):
        residual = helpers.twin_residuals(part,refs,scale)
        score = np.searchsorted(distribution,residual,side='right')/len(distribution)
        if with_replay:
            score = np.maximum(score,(part.history_match_error.to_numpy() <= replay_tolerance).astype(float))
        return score
    old_v,old_t = twin_probability(val,False),twin_probability(test,False)
    new_v,new_t = twin_probability(val,True),twin_probability(test,True)
    results = {'study':'synthetic fault detection with simulated clients and one malicious client',
               'source_sha256':summary['source_sha256'],'client_poisoning':{'client_id':0,'flip_fraction':.8},
               'replay_match_tolerance':replay_tolerance,'methods':{},'weights':{},'per_fault_recall':{},
               'limits':['simulated clients are not hospitals','injected faults are not observed cyberattacks',
                         'exact historical waveform replay strongly favors a history matching check',
                         'the earlier 30 seconds of each held-out recording are assumed clean',
                         'trust-aware weighting is reused from the repository, but this is not its full CAG-FE pipeline']}
    for scenario in ('clean','label_flip_client_0'):
        val_prob,test_prob,counts=[],[],[]
        for client in range(5):
            local = train.loc[train.simulated_client==client]
            y=local.sensor_fault_label.to_numpy().copy()
            if scenario != 'clean' and client == 0:
                rng=np.random.default_rng(2026)
                selected=rng.choice(len(y),size=int(.8*len(y)),replace=False)
                y[selected]=1-y[selected]
            model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,
                                                  min_samples_leaf=30,random_state=42+client)
            model.fit(local[features].to_numpy(),y)
            val_prob.append(model.predict_proba(val[features].to_numpy())[:,1])
            test_prob.append(model.predict_proba(test[features].to_numpy())[:,1])
            counts.append(len(local))
            print(f'{scenario}: fitted client {client+1}/5',flush=True)
        scores=[]
        vp=np.stack(val_prob)
        tp=np.stack(test_prob)
        median=np.median(vp,axis=0)
        for row in vp:
            hard=row >= .5
            scores.append({'f1':f1_score(yv,hard,average='macro',zero_division=0),
                           'pr':average_precision_score(yv,row),
                           'roc':roc_auc_score(yv,row),
                           'balanced':balanced_accuracy_score(yv,hard),
                           'brier':brier_score_loss(yv,row),
                           'consensus':float(np.mean(np.abs(row-median)))})
        s=lambda key: np.array([v[key] for v in scores])
        weights=aggregation_weights(np.array(counts),s('f1'),np.ones(5),method='trust_aware_v2',
                                    validation_pr_auc=s('pr'),validation_roc_auc=s('roc'),
                                    validation_balanced_accuracy=s('balanced'),
                                    validation_brier=s('brier'),consensus_deviation=s('consensus'))
        results['weights'][scenario]={'trust_aware_v2':weights.tolist(),'client_validation':scores}
        methods={
            'uniform_detector':(vp.mean(axis=0),tp.mean(axis=0)),
            'trust_aware_v2_detector':(np.average(vp,axis=0,weights=weights),np.average(tp,axis=0,weights=weights)),
            'old_signal_reference':(old_v,old_t),
            'replay_aware_signal_reference':(new_v,new_t)}
        dv,dt=methods['trust_aware_v2_detector']
        methods['trust_plus_old_reference']=(1-(1-dv)*(1-.25*old_v),1-(1-dt)*(1-.25*old_t))
        methods['trust_plus_replay_reference']=(1-(1-dv)*(1-.25*new_v),1-(1-dt)*(1-.25*new_t))
        methods['trust_plus_replay_gate']=(np.maximum(dv,(val.history_match_error.to_numpy() <= replay_tolerance).astype(float)),
                                           np.maximum(dt,(test.history_match_error.to_numpy() <= replay_tolerance).astype(float)))
        results['methods'][scenario]={}
        results['per_fault_recall'][scenario]={}
        for name,(v,t) in methods.items():
            threshold=helpers.select_threshold(yv,v)
            results['methods'][scenario][name]={'validation':helpers.scores(yv,v,threshold),
                                                  'test':helpers.scores(yt,t,threshold)}
            predicted=t>=threshold
            results['per_fault_recall'][scenario][name]={fault:float(predicted[test.fault.to_numpy()==fault].mean())
                for fault in ('flatline','dropout','gain','replay')}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'test_f1':{scenario:{name:metrics['test']['f1'] for name,metrics in methods.items()}
                                for scenario,methods in results['methods'].items()},
                      'test_replay_recall':{scenario:{name:metric['replay'] for name,metric in methods.items()}
                                            for scenario,methods in results['per_fault_recall'].items()}},indent=2))
    print('Saved:',a.output/'results.json')

if __name__=='__main__':
    main()
