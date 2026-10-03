"""Repeat repository XGBoost/twin experiment on three recording-level splits.

All copies of any recording stay together; attack labels are generated before
this step and clinical alarm metadata are used only to balance record splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEEDS=(42,43,44)


def assign(manifest, seed):
    # Split recordings using original clinical verdict as a balancing variable,
    # never as the sensor-fault target.
    items=manifest[['record','clinical_alarm_true']].drop_duplicates().reset_index(drop=True)
    train,rest=train_test_split(items,test_size=.4,random_state=seed,
                                stratify=items.clinical_alarm_true)
    val,test=train_test_split(rest,test_size=.5,random_state=seed+1,
                              stratify=rest.clinical_alarm_true)
    assigned={}
    sites={}
    for split,part in [('train',train),('validation',val),('test',test)]:
        for record in part.record:
            if record in assigned:raise AssertionError('Repeated record')
            assigned[record]=split
        if split=='train':
            order=sorted(part.record,key=lambda r:hashlib.sha256(f'{seed}:{r}'.encode()).hexdigest())
            sites.update({record:i%5 for i,record in enumerate(order)})
    if len(assigned)!=len(items):raise AssertionError('Missing recording')
    return assigned,sites


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared',type=Path,default=Path('outputs/physionet_2015_prepared_v3'))
    ap.add_argument('--output',type=Path,default=Path('outputs/physionet_xgboost_three_splits'))
    args=ap.parse_args()
    script=Path(__file__).with_name('20_run_physionet_repository_xgboost.py')
    source=json.loads((args.prepared/'preparation_summary.json').read_text())
    if source.get('preparation_version')!=3:raise ValueError('Requires prepared_v3')
    frame=pd.read_csv(args.prepared/'sensor_windows.csv')
    original_manifest=pd.read_csv(args.prepared/'record_split_manifest.csv')
    if original_manifest.record.duplicated().any() or set(frame.record)!=set(original_manifest.record):
        raise ValueError('Invalid source recording manifest')
    all_results={'source_sha256':source['source_sha256'],'seeds':{},
                 'split_unit':'recording; patient identity not verified',
                 'comparison':'same five-client repository XGBoost configuration; split and simulated client allocation vary',
                 'notes':['synthetic sensor faults','simulated clients','overlapping datasets between splits: descriptive stability, not independent replications',
                          'replay tolerance 0.01 inherited from initial validation, not retuned on each test split']}
    for seed in SEEDS:
        assigned,sites=assign(original_manifest,seed)
        prepared=args.output/f'prepared_seed_{seed}'
        prepared.mkdir(parents=True,exist_ok=True)
        updated=frame.copy()
        updated['split']=updated.record.map(assigned)
        updated['simulated_client']=updated.record.map(sites)
        if updated.split.isna().any() or updated.groupby('record').split.nunique().gt(1).any():
            raise AssertionError('Recording crossed split')
        updated.to_csv(prepared/'sensor_windows.csv',index=False)
        split_manifest=original_manifest.copy()
        split_manifest['split']=split_manifest.record.map(assigned)
        split_manifest['simulated_client']=split_manifest.record.map(sites)
        split_manifest.to_csv(prepared/'record_split_manifest.csv',index=False)
        report=dict(source)
        report['seed']=seed
        report['split_record_counts']={split:int(sum(split_manifest.split==split)) for split in ('train','validation','test')}
        (prepared/'preparation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        destination=args.output/f'run_seed_{seed}'
        subprocess.run([sys.executable,str(script),'--prepared',str(prepared),'--output',str(destination)],check=True)
        result=json.loads((destination/'results.json').read_text())
        entry={'record_counts':report['split_record_counts'],'conditions':{}}
        for condition,r in result['runs'].items():
            entry['conditions'][condition]={
                'detector_macro_f1':r['repository_detector_test']['macro_f1'],
                'fused_macro_f1':r['detector_plus_replay_test']['macro_f1'],
                'replay_recall':r['detector_plus_replay_test_recall'],
                'client_weights':r['detector_client_weights'],
                'flipped_labels_client_0':r['poisoning_manifest']['clients']['0']['flipped_labels']}
        all_results['seeds'][str(seed)]=entry
        print('Finished split',seed,flush=True)
    all_results['summary']={}
    for condition in ('clean','label_flip_client_0'):
        values=np.array([[all_results['seeds'][str(seed)]['conditions'][condition][key]
                          for seed in SEEDS] for key in ('detector_macro_f1','fused_macro_f1','replay_recall')])
        all_results['summary'][condition]={key:{'mean':float(np.mean(v)),'min':float(np.min(v)),
                                                  'max':float(np.max(v)),
                                                  'std_sample':float(np.std(v,ddof=1))}
                                            for key,v in zip(('detector_macro_f1','fused_macro_f1','replay_recall'),values)}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
    print('Saved:',args.output/'summary.json')

if __name__=='__main__':
    main()
