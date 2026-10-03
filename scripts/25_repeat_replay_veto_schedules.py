"""Repeat the frozen replay-veto policy across independent fault schedules.

Recording split, fitted model and thresholds stay fixed. The resulting seeds
are sensitivity checks on the same held-out recordings, not new cohorts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',type=Path,default=Path('data/raw/physionet_2015/training.zip'))
    p.add_argument('--prepared',type=Path,default=Path('outputs/physionet_2015_prepared_v3'))
    p.add_argument('--state-audit',type=Path,default=Path('outputs/physionet_causal_signal_audit/summary.json'))
    p.add_argument('--replay-calibration',type=Path,default=Path('outputs/physionet_2015_robust_replay/results.json'))
    p.add_argument('--federated-model',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/physionet_replay_veto_schedule_check'))
    p.add_argument('--seeds',type=int,nargs='+',default=[2026,3031,4042,5053])
    args=p.parse_args()
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('Repeated schedule seed')
    args.output.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed in args.seeds:
        destination=args.output/f'seed_{seed}'
        command=[sys.executable,str(Path(__file__).with_name('24_audit_replay_veto.py')),
                 '--zip',str(args.zip),'--prepared',str(args.prepared),
                 '--state-audit',str(args.state_audit),
                 '--replay-calibration',str(args.replay_calibration),
                 '--seed',str(seed),'--output',str(destination)]
        if args.federated_model:command+=['--federated-model',str(args.federated_model)]
        subprocess.run(command,check=True)
        report=json.loads((destination/'summary.json').read_text())
        if report['seed']!=seed or report['test_record_count']!=126:
            raise ValueError('Inconsistent seed or test count')
        if runs and (report['source_sha256']!=runs[0]['source_sha256'] or
                     report['federated_model_run']!=runs[0]['federated_model_run'] or
                     report['replay_cutoff_selected_on_validation']!=runs[0]['replay_cutoff_selected_on_validation']):
            raise ValueError('Model, threshold, or source changed between schedules')
        runs.append(report)
    metrics=('replay_veto_attack_recall','replay_veto_attacked_windows_accepted_for_update',
             'replay_veto_clean_false_positive_rate')
    output={'source_sha256':runs[0]['source_sha256'],'federated_model_run':runs[0]['federated_model_run'],
            'validation_replay_cutoff':runs[0]['replay_cutoff_selected_on_validation'],
            'seeds':args.seeds,'recordings_reused_per_schedule':126,
            'per_seed':[{'seed':r['seed'],'attacked_windows':r['attacked_windows_observed'],
                         'replay_windows_accepted':r['replay_veto_per_fault']['replay']['accepted_for_update'],
                         'gain_windows_accepted':r['replay_veto_per_fault']['gain']['accepted_for_update'],
                         **{k:r['mixed_stream'][k] for k in metrics}} for r in runs],
            'ranges':{k:{'min':float(np.min([r['mixed_stream'][k] for r in runs])),
                         'max':float(np.max([r['mixed_stream'][k] for r in runs]))} for k in metrics},
            'limits':['All schedules reuse the same test recordings and the same synthetic fault types.',
                      'The fixed thresholds were selected on earlier validation data.',
                      'These are sensitivity checks, not independent external validation or a clinical digital twin.']}
    (args.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output['per_seed'],indent=2))
    print('Saved:',args.output/'summary.json')

if __name__=='__main__':main()
