"""Check whether a causal per-record signal state tracks CLEAN PhysioNet windows.

This is a validation of a simple state surrogate, not a certified patient twin.
The state uses no future windows or clinical alarm labels for prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

FEATURES=('ecg_std','pleth_std','pleth_diff_std','pleth_ecg_diff_std_ratio')


def load_prep():
    source=Path(__file__).with_name('14_prepare_physionet_sensor_study.py')
    spec=importlib.util.spec_from_file_location('physionet_preparation',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_sha(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
    return sha.hexdigest()


def baseline(archive, record, prep):
    ecg,pleth,rate=prep.decode(archive,record)
    size=rate*5
    features=[prep.stats(ecg[i*size:(i+1)*size],pleth[i*size:(i+1)*size])
              for i in range(6)]
    return np.median([[f[k] for k in FEATURES] for f in features],axis=0)


def run_record(group,state,scales,alpha,gate):
    current=state.copy()
    static_error=[]
    dynamic_error=[]
    updated=0
    values=group.loc[:,list(FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite values in clean waveform features')
    for observed in values:
        fixed=float(np.sqrt(np.mean(((observed-state)/scales)**2)))
        causal=float(np.sqrt(np.mean(((observed-current)/scales)**2)))
        static_error.append(fixed)
        dynamic_error.append(causal)
        if causal<=gate:
            current=(1-alpha)*current+alpha*observed
            updated+=1
    return {'static_mean':float(np.mean(static_error)),
            'causal_mean':float(np.mean(dynamic_error)),
            'updated_fraction':updated/len(values),
            'n_windows':len(values)}


def record_table(frame,states,scales,alpha,gate):
    rows=[]
    for record,part in frame.groupby('record',sort=True):
        group=part.sort_values('window_start_seconds')
        result=run_record(group,states[record],scales,alpha,gate)
        rows.append({'record':record,'split':str(group.split.iloc[0]),
                     'clinical_alarm_true_metadata':int(group.clinical_alarm_true_metadata.iloc[0]),
                     **result})
    return pd.DataFrame(rows)


def summarize(table):
    result={}
    for split,part in table.groupby('split'):
        result[split]={'records':len(part),
                       'mean_static_error':float(part.static_mean.mean()),
                       'mean_causal_error':float(part.causal_mean.mean()),
                       'median_static_error':float(part.static_mean.median()),
                       'median_causal_error':float(part.causal_mean.median()),
                       'records_static_error_over_10':int((part.static_mean>10).sum()),
                       'records_causal_error_over_10':int((part.causal_mean>10).sum()),
                       'median_relative_reduction':float(np.median((part.static_mean-part.causal_mean)/
                                                                 np.maximum(part.static_mean,1e-9))),
                       'fraction_records_improved':float(np.mean(part.causal_mean<part.static_mean)),
                       'mean_update_fraction':float(part.updated_fraction.mean()),
                       'by_clinical_verdict':{str(k):{'records':len(g),
                            'mean_static_error':float(g.static_mean.mean()),
                            'mean_causal_error':float(g.causal_mean.mean())}
                            for k,g in part.groupby('clinical_alarm_true_metadata')}}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip',type=Path,default=Path('data/raw/physionet_2015/training.zip'))
    parser.add_argument('--prepared',type=Path,default=Path('outputs/physionet_2015_prepared_v3'))
    parser.add_argument('--output',type=Path,default=Path('outputs/physionet_causal_signal_audit'))
    args=parser.parse_args()
    summary=json.loads((args.prepared/'preparation_summary.json').read_text())
    if summary.get('preparation_version')!=3 or source_sha(args.zip)!=summary['source_sha256']:
        raise ValueError('Use prepared_v3 created from this exact training.zip')
    prep=load_prep()
    frame=pd.read_csv(args.prepared/'sensor_windows.csv')
    manifest=pd.read_csv(args.prepared/'record_split_manifest.csv')
    if manifest.record.duplicated().any() or set(frame.record)!=set(manifest.record):
        raise ValueError('Invalid recording manifest')
    splits=manifest.set_index('record').split.to_dict()
    if any(splits[r]!=s for r,s in zip(frame.record,frame.split)):
        raise ValueError('Recording crossed a split')
    clean=frame.loc[frame.fault=='none'].copy()
    if clean.duplicated(['record','window_start_seconds']).any():
        raise ValueError('Repeated clean window')
    with ZipFile(args.zip) as archive:
        states={record:baseline(archive,record,prep) for record in manifest.record}
    train=clean.loc[clean.split=='train']
    reference=np.stack([states[r] for r in train.record])
    changes=np.abs(train[list(FEATURES)].to_numpy(dtype=float)-reference)
    scale=np.maximum(np.quantile(changes,.90,axis=0),.01)
    normalized=np.sqrt(np.mean(((train[list(FEATURES)].to_numpy()-reference)/scale)**2,axis=1))
    gate=float(np.quantile(normalized,.95))
    choices=[]
    validation=clean.loc[clean.split=='validation']
    val_states={r:states[r] for r in validation.record.unique()}
    for alpha in (.01,.05,.10,.20,.50):
        table=record_table(validation,val_states,scale,alpha,gate)
        score=float(np.mean(table.causal_mean))
        choices.append({'alpha':alpha,'validation_record_mean_error':score,
                        'validation_records_improved':float(np.mean(table.causal_mean<table.static_mean))})
    chosen=min(choices,key=lambda x:(x['validation_record_mean_error'],x['alpha']))
    table=record_table(clean,states,scale,chosen['alpha'],gate)
    report={'source_sha256':summary['source_sha256'],'features':list(FEATURES),
            'feature_scale_from_train_clean':scale.tolist(),
            'update_gate_from_train_clean':gate,'validation_choices':choices,
            'selected_alpha':chosen['alpha'],'summary':summarize(table),
            'interpretation_limitations':[
                'All evaluated windows in this audit are known-clean controlled copies, not a mixed live stream.',
                'The initial 30 seconds of each recording are assumed clean and available at deployment.',
                'Better one-step prediction error would support state tracking, not a validated patient digital twin.',
                'Large waveform outliers make mean error unstable; inspect the reported medians and per-record CSV.',
                'Clinical alarm verdicts are metadata only and never train or tune the state.',
                'The dataset lacks verified patient/hospital identities and observed cyberattacks.']}
    args.output.mkdir(parents=True,exist_ok=True)
    table.to_csv(args.output/'per_record.csv',index=False)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Chosen update rate:',chosen['alpha'],'training-only update gate:',round(gate,4))
    print('Test result:',json.dumps(report['summary']['test'],indent=2))
    print('Saved:',args.output/'summary.json')

if __name__=='__main__':
    main()
