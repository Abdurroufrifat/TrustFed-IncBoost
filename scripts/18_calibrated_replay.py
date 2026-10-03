"""Frozen-threshold stress test on held-out approximate replay waveforms.

Starts from stage-4 prepared v2 data; retrains clean simulated-client detector
only once, then changes test replay signals. Test labels never tune thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
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


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def perturbed_replay_frame(base, condition, archive, prep, feature_columns):
    observed = base.copy()
    replay_indices = np.flatnonzero(base.fault.to_numpy() == 'replay')
    records = base.iloc[replay_indices].groupby('record', sort=True).indices
    # GroupBy indices above are local positions into replay_indices.
    for record, local_positions in records.items():
        ecg, pleth, rate = prep.decode(archive, record)
        window = rate*5
        history = [pleth[k*window:(k+1)*window] for k in range(6)]
        reference_bank = prep.history_reference(history)
        for position in replay_indices[local_positions]:
            row = base.iloc[position]
            start = int(round(row.window_start_seconds*rate))
            index = (start-rate*30)//window
            current_ecg = ecg[start:start+window]
            replay = history[index % 6].copy()
            if condition.startswith('noise_'):
                fraction = float(condition.split('_')[1])
                digest = hashlib.sha256(f'{record}:{start}:{condition}'.encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
                replay += rng.normal(0, fraction*max(np.std(replay),.01),len(replay))
            elif condition.startswith('shift_'):
                replay = np.roll(replay,int(condition.split('_')[1]))
            elif condition != 'exact':
                raise ValueError(f'unknown replay condition: {condition}')
            stats = prep.stats(current_ecg,replay)
            stats['history_match_error'] = min(float(np.mean(np.abs(replay-prior)) /
                                                    max(float(np.std(prior)),.01)) for prior in history)
            stats['history_robust_error'] = prep.robust_history_error(replay,reference_bank)
            for key in (*feature_columns,'history_match_error','history_robust_error'):
                observed.at[observed.index[position],key] = stats[key]
    return observed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',type=Path,default=Path('data/raw/physionet_2015/training.zip'))
    p.add_argument('--prepared',type=Path,default=Path('outputs/physionet_2015_prepared_v3'))
    p.add_argument('--output',type=Path,default=Path('outputs/physionet_2015_robust_replay'))
    args=p.parse_args()
    scripts=Path(__file__).resolve().parent
    prep=import_file('physionet_prep',scripts/'14_prepare_physionet_sensor_study.py')
    helper=import_file('physionet_helper',scripts/'15_run_physionet_federated_twin.py')
    summary=json.loads((args.prepared/'preparation_summary.json').read_text())
    if summary.get('preparation_version') != 3 or helper.source_hash(args.zip) != summary['source_sha256']:
        raise ValueError('Need prepared_v3 generated from this exact training.zip')
    frame=pd.read_csv(args.prepared/'sensor_windows.csv')
    manifest=pd.read_csv(args.prepared/'record_split_manifest.csv')
    if manifest.record.duplicated().any() or set(frame.record) != set(manifest.record):
        raise ValueError('Invalid recording manifest')
    splits=manifest.set_index('record').split.to_dict()
    if any(splits[r]!=s for r,s in zip(frame.record,frame.split)):
        raise ValueError('Recording leaked across split')
    excluded={'sensor_fault_label','fault','clinical_alarm_true_metadata','record','split',
              'simulated_client','window_start_seconds','history_match_error','history_robust_error'}
    features=sorted(set(frame.columns)-excluded)
    missing=frame[features].isna().any(axis=1)
    if (missing & (frame.fault!='dropout')).any():
        raise ValueError('Unexpected missing features')
    frame[features]=frame[features].fillna(0.)
    train=frame.loc[frame.split=='train']
    validation=frame.loc[frame.split=='validation']
    test=frame.loc[frame.split=='test'].copy()
    yv=validation.sensor_fault_label.to_numpy()
    yt=test.sensor_fault_label.to_numpy()
    models=[]
    for client in range(5):
        local=train.loc[train.simulated_client==client]
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,
                                              min_samples_leaf=30,random_state=42+client)
        model.fit(local[features].to_numpy(),local.sensor_fault_label.to_numpy())
        models.append(model)
    val_prob=np.stack([m.predict_proba(validation[features].to_numpy())[:,1] for m in models])
    # The same validation diagnostics and repository aggregation function as stage 4.
    sys.path.insert(0,str(scripts.parent/"src"))
    from trustfed_incboost.models.federated import aggregation_weights
    med=np.median(val_prob,axis=0)
    diagnostics=[]
    for prob in val_prob:
        pred=prob>=.5
        diagnostics.append([f1_score(yv,pred,average='macro'),average_precision_score(yv,prob),
                            roc_auc_score(yv,prob),balanced_accuracy_score(yv,pred),
                            brier_score_loss(yv,prob),np.mean(np.abs(prob-med))])
    diag=np.array(diagnostics)
    weights=aggregation_weights(np.array([len(train.loc[train.simulated_client==i]) for i in range(5)]),
        diag[:,0],np.ones(5),method='trust_aware_v2',validation_pr_auc=diag[:,1],
        validation_roc_auc=diag[:,2],validation_balanced_accuracy=diag[:,3],
        validation_brier=diag[:,4],consensus_deviation=diag[:,5])
    benign=train.loc[train.sensor_fault_label==0]
    tolerance=float(min(.02,np.quantile(benign.history_match_error,.001)))
    val_detector=np.average(val_prob,axis=0,weights=weights)
    val_gate=validation.history_match_error.to_numpy()<=tolerance
    thresholds={'detector':helper.select_threshold(yv,val_detector),
                'detector_plus_replay_gate':helper.select_threshold(yv,np.maximum(val_detector,val_gate.astype(float)))}
    conditions=('exact','noise_0.001','noise_0.01','noise_0.05','noise_0.10',
                'shift_1','shift_5','shift_25')
    candidate_cutoffs=(0.,.001,.005,.01,.02,.05,.10,.20)
    calibration_conditions=('exact','noise_0.01','shift_5')
    with ZipFile(args.zip) as archive:
        validation_scenarios={c:perturbed_replay_frame(validation,c,archive,prep,features)
                              for c in calibration_conditions}
        def make_detector(part):
            probs=np.stack([m.predict_proba(part[features].fillna(0.).to_numpy())[:,1] for m in models])
            return np.average(probs,axis=0,weights=weights)
        validation_detector={c:make_detector(part) for c,part in validation_scenarios.items()}
        choices=[]
        for cutoff in candidate_cutoffs:
            predictions=[]
            for condition, part in validation_scenarios.items():
                gate=(part.history_robust_error.to_numpy()<=cutoff).astype(float)
                predictions.append(np.maximum(validation_detector[condition],gate))
            pooled=np.concatenate(predictions)
            pooled_labels=np.tile(yv,len(predictions))
            selected=helper.select_threshold(pooled_labels,pooled)
            f1s=[f1_score(yv,score>=selected) for score in predictions]
            clean=validation.fault.to_numpy()=='none'
            clean_fpr=float(np.mean((predictions[0]>=selected)[clean]))
            choices.append({'cutoff':cutoff,'threshold':selected,'calibration_mean_f1':float(np.mean(f1s)),
                            'calibration_clean_fpr':clean_fpr})
        # Clean validation false-alarm constraint is specified before test.
        eligible=[c for c in choices if c['calibration_clean_fpr']<=.15]
        if not eligible:
            raise ValueError('No robust tolerance meets validation clean FPR <= 15%')
        chosen=max(eligible,key=lambda x:(x['calibration_mean_f1'],-x['calibration_clean_fpr'],-x['cutoff']))
        output={'source_sha256':summary['source_sha256'],'conditions':{},
                'calibration_choices':choices,'chosen_calibration':chosen,
                'thresholds':thresholds,'client_weights':weights.tolist(),
                'calibration_conditions':calibration_conditions,'test_record_count':int(test.record.nunique()),
                'limits':['validation calibration includes synthetic approximate replay',
                          'timing alignment is limited to five samples',
                          'single recording split and synthetic clients',
                          'held-out recordings are not verified patient-disjoint']}
        for condition in conditions:
            changed=perturbed_replay_frame(test,condition,archive,prep,features)
            detector=make_detector(changed)
            exact_gate=changed.history_match_error.to_numpy()<=tolerance
            robust_gate=changed.history_robust_error.to_numpy()<=chosen['cutoff']
            methods={
                'detector':(detector,thresholds['detector']),
                'exact_replay_gate':(np.maximum(detector,exact_gate.astype(float)),thresholds['detector_plus_replay_gate']),
                'calibrated_aligned_gate':(np.maximum(detector,robust_gate.astype(float)),chosen['threshold'])}
            result={}
            for name,(predictions,threshold) in methods.items():
                positive=predictions>=threshold
                replay=changed.fault.to_numpy()=='replay'
                clean=changed.fault.to_numpy()=='none'
                result[name]={'overall':helper.scores(yt,predictions,threshold),
                    'replay_recall':float(positive[replay].mean()),
                    'clean_false_positive_rate':float(positive[clean].mean()),
                    'replay_count':int(replay.sum())}
            output['conditions'][condition]=result
            print(condition,'calibrated replay recall',round(result['calibrated_aligned_gate']['replay_recall'],4),
                  'F1',round(result['calibrated_aligned_gate']['overall']['f1'],4),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print('Saved:',args.output/'results.json')

if __name__=='__main__':
    main()
