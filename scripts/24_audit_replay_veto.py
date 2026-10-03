"""Audit causal state contamination during a mixed clean/synthetic-fault stream.

The attack schedule is fixed by hash, and only one observed row per time step
is given to each state policy. Paired clean rows supply evaluation targets only.
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
import joblib


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attack_schedule(record, start, seed, fraction):
    value=hashlib.sha256(f'{seed}:{record}:{start:.4f}'.encode()).digest()
    return int.from_bytes(value[:8],'big')/2**64<fraction


def safe_median(values):
    return float(np.median(values)) if len(values) else None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip',type=Path,default=Path('data/raw/physionet_2015/training.zip'))
    ap.add_argument('--prepared',type=Path,default=Path('outputs/physionet_2015_prepared_v3'))
    ap.add_argument('--state-audit',type=Path,default=Path('outputs/physionet_causal_signal_audit/summary.json'))
    ap.add_argument('--output',type=Path,default=Path('outputs/physionet_replay_veto_audit'))
    ap.add_argument('--replay-calibration',type=Path,default=Path('outputs/physionet_2015_robust_replay/results.json'))
    ap.add_argument('--federated-model',type=Path,default=None,help='Clean XGBoost federated_ensemble.joblib from script 20')
    ap.add_argument('--attack-fraction',type=float,default=.2)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--replay-condition',choices=('exact','noise_0.01','noise_0.05','shift_5','shift_25'),default='exact')
    args=ap.parse_args()
    if not 0<args.attack_fraction<1:raise ValueError('attack fraction must lie in (0,1)')
    prep=load('physionet_prep',Path(__file__).with_name('14_prepare_physionet_sensor_study.py'))
    prior=load('physionet_state',Path(__file__).with_name('22_audit_causal_signal_state.py'))
    config=json.loads((args.prepared/'preparation_summary.json').read_text())
    state_config=json.loads(args.state_audit.read_text())
    if config.get('preparation_version')!=3 or config['source_sha256']!=state_config['source_sha256']:
        raise ValueError('Inconsistent prepared_v3 and causal-state audit')
    if prior.source_sha(args.zip)!=config['source_sha256']:
        raise ValueError('training.zip source hash mismatch')
    replay_config=json.loads(args.replay_calibration.read_text())
    if replay_config['source_sha256']!=config['source_sha256']:
        raise ValueError('Replay calibration was computed on another source ZIP')
    replay_cutoff=float(replay_config['chosen_calibration']['cutoff'])
    frame=pd.read_csv(args.prepared/'sensor_windows.csv')
    manifest=pd.read_csv(args.prepared/'record_split_manifest.csv')
    if manifest.record.duplicated().any() or set(frame.record)!=set(manifest.record):
        raise ValueError('Invalid recording manifest')
    lookup=manifest.set_index('record').split.to_dict()
    if any(lookup[r]!=s for r,s in zip(frame.record,frame.split)):
        raise ValueError('Recording leaked across split')
    test=frame[frame.split=='test'].copy()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from trustfed_incboost.models.baseline import predict_probabilities
    from trustfed_incboost.models.federated import aggregate_probabilities
    expected=np.flatnonzero(frame.split.to_numpy()=='test')
    def matches_split(path):
        predictions=path.parent/'test_predictions.csv'
        if not predictions.is_file():return False
        saved=pd.read_csv(predictions)
        return (np.array_equal(saved.row_index.to_numpy(dtype=int),expected)
                and np.array_equal(saved.y_true.to_numpy(dtype=int),
                                   frame.iloc[expected].sensor_fault_label.to_numpy(dtype=int)))
    model_path=args.federated_model
    if model_path is None:
        candidates=sorted(Path('outputs/physionet_repository_pipeline').glob(
            'physionet_repository_xgboost_clean_*/federated_ensemble.joblib'),reverse=True)
        model_path=next((candidate for candidate in candidates if matches_split(candidate)),None)
        if model_path is None:
            raise FileNotFoundError('No compatible clean XGBoost model found. Run script 20 with prepared_v3 or supply --federated-model PATH')
    if not matches_split(model_path):
        raise ValueError('Federated model test split/labels do not match prepared_v3')
    bundle=joblib.load(model_path)
    if args.replay_condition!='exact':
        perturb=load('replay_perturbation',Path(__file__).with_name('18_calibrated_replay.py'))
        excluded={'sensor_fault_label','fault','clinical_alarm_true_metadata','record','split',
                  'simulated_client','window_start_seconds','history_match_error','history_robust_error'}
        feature_columns=sorted(set(test.columns)-excluded)
        with ZipFile(args.zip) as archive:
            test=perturb.perturbed_replay_frame(test,args.replay_condition,archive,prep,feature_columns)
    transformed=bundle['preprocessor'].transform(test)
    probabilities=aggregate_probabilities(
        [predict_probabilities(m,transformed) for m in bundle['client_models']],
        bundle['aggregation_weights'])[:,1]
    test['__detector_probability']=probabilities
    detector_gate=float(bundle['threshold'])
    features=list(prior.FEATURES)
    if features!=state_config['features']:raise ValueError('Feature mismatch')
    scales=np.array(state_config['feature_scale_from_train_clean'],dtype=float)
    alpha=float(state_config['selected_alpha'])
    gate=float(state_config['update_gate_from_train_clean'])
    with ZipFile(args.zip) as archive:
        states={record:prior.baseline(archive,record,prep) for record in test.record.unique()}
    results=[]
    fault_stats={}
    protected_fault_stats={}
    replay_veto_fault_stats={}
    for record,group in test.groupby('record',sort=True):
        static=states[record]
        models={name:static.copy() for name in ('frozen','always_update','deviation_gated','detector_and_deviation_gated','detector_deviation_replay_veto')}
        # The oracle uses clean targets for auditing contamination only.
        oracle=static.copy()
        error={name:[] for name in models}
        clean_error={name:[] for name in models}
        end_distance={}
        flagged=[]
        accepted_attack=[]
        selected_faults=[]
        protected_accepted_attack=[]
        protected_flags=[]
        replay_veto_flags=[]
        replay_veto_accepted_attack=[]
        time_groups=list(group.groupby('window_start_seconds',sort=True))
        for start,window_rows in time_groups:
            if len(window_rows)!=2 or (window_rows.fault=='none').sum()!=1:
                raise ValueError(f'Expected one clean and one fault row for {record} at {start}')
            clean=window_rows.loc[window_rows.fault=='none'].iloc[0]
            fault=window_rows.loc[window_rows.fault!='none'].iloc[0]
            attacked=attack_schedule(record,float(start),args.seed,args.attack_fraction)
            observed=fault if attacked else clean
            target=clean[features].to_numpy(dtype=float)
            observation=observed[features].to_numpy(dtype=float)
            # Dropout has nonfinite summary statistics; the sensor front-end
            # presents zero plus explicit missingness in the original CSV.
            observation=np.nan_to_num(observation,nan=0.0,posinf=0.0,neginf=0.0)
            if not np.isfinite(target).all():raise ValueError('Nonfinite clean target')
            for name,state in models.items():
                predictive=float(np.sqrt(np.mean(((target-state)/scales)**2)))
                error[name].append(predictive)
                if not attacked:clean_error[name].append(predictive)
            observed_deviation=float(np.sqrt(np.mean(((observation-models['deviation_gated'])/scales)**2)))
            accepted=observed_deviation<=gate
            protected_deviation=float(np.sqrt(np.mean(((observation-models['detector_and_deviation_gated'])/scales)**2)))
            protected_accept=protected_deviation<=gate and float(observed['__detector_probability'])<detector_gate
            replay_deviation=float(np.sqrt(np.mean(((observation-models['detector_deviation_replay_veto'])/scales)**2)))
            replay_accept=(replay_deviation<=gate and float(observed['__detector_probability'])<detector_gate
                           and float(observed['history_robust_error'])>replay_cutoff)
            replay_veto_flags.append((attacked,not replay_accept))
            protected_flags.append((attacked,not protected_accept))
            flagged.append((attacked,not accepted))
            if attacked:
                accepted_attack.append(bool(accepted))
                protected_accepted_attack.append(bool(protected_accept))
                replay_veto_accepted_attack.append(bool(replay_accept))
                selected_faults.append(str(fault.fault))
                current=fault_stats.setdefault(str(fault.fault),{'windows':0,'accepted_for_update':0})
                current['windows']+=1
                current['accepted_for_update']+=int(accepted)
                protected=protected_fault_stats.setdefault(str(fault.fault),{'windows':0,'accepted_for_update':0})
                protected['windows']+=1
                protected['accepted_for_update']+=int(protected_accept)
                veto=replay_veto_fault_stats.setdefault(str(fault.fault),{'windows':0,'accepted_for_update':0})
                veto['windows']+=1
                veto['accepted_for_update']+=int(replay_accept)
            models['always_update']=(1-alpha)*models['always_update']+alpha*observation
            if accepted:
                models['deviation_gated']=(1-alpha)*models['deviation_gated']+alpha*observation
            if protected_accept:
                models['detector_and_deviation_gated']=(1-alpha)*models['detector_and_deviation_gated']+alpha*observation
            if replay_accept:
                models['detector_deviation_replay_veto']=(1-alpha)*models['detector_deviation_replay_veto']+alpha*observation
            oracle=(1-alpha)*oracle+alpha*target
        end_distance={name:float(np.sqrt(np.mean(((model-oracle)/scales)**2)))
                      for name,model in models.items()}
        pred=np.array(flagged)
        protected_pred=np.array(protected_flags)
        replay_veto_pred=np.array(replay_veto_flags)
        results.append({'record':record,'windows':len(time_groups),'attacked_windows':len(accepted_attack),
                        'accepted_attacked_fraction':float(np.mean(accepted_attack)) if accepted_attack else None,
                        'protected_accepted_attacked_fraction':float(np.mean(protected_accepted_attack)) if protected_accepted_attack else None,
                        'replay_veto_accepted_attacked_fraction':float(np.mean(replay_veto_accepted_attack)) if replay_veto_accepted_attack else None,
                        'fault_counts':{name:selected_faults.count(name) for name in sorted(set(selected_faults))},
                        'attack_recall':float(np.mean(pred[pred[:,0],1])) if pred[:,0].any() else None,
                        'protected_attack_recall':float(np.mean(protected_pred[protected_pred[:,0],1])) if protected_pred[:,0].any() else None,
                        'replay_veto_attack_recall':float(np.mean(replay_veto_pred[replay_veto_pred[:,0],1])) if replay_veto_pred[:,0].any() else None,
                        'replay_veto_clean_false_positive_rate':float(np.mean(replay_veto_pred[~replay_veto_pred[:,0],1])) if (~replay_veto_pred[:,0]).any() else None,
                        'protected_clean_false_positive_rate':float(np.mean(protected_pred[~protected_pred[:,0],1])) if (~protected_pred[:,0]).any() else None,
                        'clean_false_positive_rate':float(np.mean(pred[~pred[:,0],1])) if (~pred[:,0]).any() else None,
                        **{f'{name}_clean_error':float(np.mean(clean_error[name])) for name in models},
                        **{f'{name}_all_target_error':float(np.mean(error[name])) for name in models},
                        **{f'{name}_end_distance_to_clean_oracle':value for name,value in end_distance.items()}})
    table=pd.DataFrame(results)
    def pooled(column,weight):
        mask=table[column].notna()
        return float(np.average(table.loc[mask,column],weights=table.loc[mask,weight]))
    report={'source_sha256':config['source_sha256'],'attack_fraction_requested':args.attack_fraction,
            'replay_condition':args.replay_condition,
            'attacked_windows_observed':int(table.attacked_windows.sum()),
            'total_windows':int(table.windows.sum()),'test_record_count':len(table),
            'seed':args.seed,'selected_alpha':alpha,'train_calibrated_update_gate':gate,
            'federated_model_run':str(model_path.parent),'federated_validation_threshold':detector_gate,
            'replay_cutoff_selected_on_validation':replay_cutoff,
            'per_fault':{k:{**v,'accepted_fraction':v['accepted_for_update']/v['windows']}
                         for k,v in fault_stats.items()},
            'protected_per_fault':{k:{**v,'accepted_fraction':v['accepted_for_update']/v['windows']}
                         for k,v in protected_fault_stats.items()},
            'replay_veto_per_fault':{k:{**v,'accepted_fraction':v['accepted_for_update']/v['windows']}
                         for k,v in replay_veto_fault_stats.items()},
            'mixed_stream':{'attack_recall':pooled('attack_recall','attacked_windows'),
                            'attacked_windows_accepted_for_update':pooled('accepted_attacked_fraction','attacked_windows'),
                            'protected_attacked_windows_accepted_for_update':pooled('protected_accepted_attacked_fraction','attacked_windows'),
                            'protected_attack_recall':pooled('protected_attack_recall','attacked_windows'),
                            'replay_veto_attacked_windows_accepted_for_update':pooled('replay_veto_accepted_attacked_fraction','attacked_windows'),
                            'replay_veto_attack_recall':pooled('replay_veto_attack_recall','attacked_windows'),
                            'replay_veto_clean_false_positive_rate':float(np.mean(table.replay_veto_clean_false_positive_rate)),
                            'protected_clean_false_positive_rate':float(np.mean(table.protected_clean_false_positive_rate)),
                            'clean_false_positive_rate':float(np.mean(table.clean_false_positive_rate)),
                            'median_clean_prediction_error_by_policy':{name:safe_median(table[f'{name}_clean_error']) for name in models},
                            'median_final_distance_to_oracle_by_policy':{name:safe_median(table[f'{name}_end_distance_to_clean_oracle']) for name in models},
                            'fraction_records_gated_beats_always_on_clean':float(np.mean(table.deviation_gated_clean_error<table.always_update_clean_error)),
                            'fraction_records_gated_beats_frozen_on_clean':float(np.mean(table.deviation_gated_clean_error<table.frozen_clean_error)),
                            'fraction_records_protected_beats_frozen_on_clean':float(np.mean(table.detector_and_deviation_gated_clean_error<table.frozen_clean_error))},
            'limitations':['Synthetic fault schedule, not observed attacks.',
                           'The oracle clean stream is used only after inference for audit; it cannot run online.',
                           'A deviation gate may absorb subtle attacks even when prediction error improves.',
                           'No verified patient or hospital identities; no validated patient digital twin.']}
    args.output.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output/'per_record.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['mixed_stream'],indent=2))
    print('Saved:',args.output/'summary.json')

if __name__=='__main__':
    main()
