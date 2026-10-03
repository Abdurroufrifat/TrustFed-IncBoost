"""Hold all training and thresholds fixed while perturbing held-out replay signals."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',type=Path,default=Path('data/raw/physionet_2015/training.zip'))
    p.add_argument('--prepared',type=Path,default=Path('outputs/physionet_2015_prepared_v3'))
    p.add_argument('--state-audit',type=Path,default=Path('outputs/physionet_causal_signal_audit/summary.json'))
    p.add_argument('--replay-calibration',type=Path,default=Path('outputs/physionet_2015_robust_replay/results.json'))
    p.add_argument('--federated-model',type=Path)
    p.add_argument('--output',type=Path,default=Path('outputs/physionet_replay_state_stress'))
    p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    conditions=('exact','noise_0.01','noise_0.05','shift_5','shift_25')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for condition in conditions:
        destination=args.output/condition
        cmd=[sys.executable,str(Path(__file__).with_name('24_audit_replay_veto.py')),
             '--zip',str(args.zip),'--prepared',str(args.prepared),'--state-audit',str(args.state_audit),
             '--replay-calibration',str(args.replay_calibration),'--seed',str(args.seed),
             '--replay-condition',condition,'--output',str(destination)]
        if args.federated_model:cmd+=['--federated-model',str(args.federated_model)]
        subprocess.run(cmd,check=True)
        report=json.loads((destination/'summary.json').read_text())
        if report['replay_condition']!=condition:raise ValueError('Inconsistent replay condition')
        if rows and (report['source_sha256']!=rows[0]['source_sha256'] or
                     report['federated_model_run']!=rows[0]['model'] or
                     report['attacked_windows_observed']!=rows[0]['attacked_windows']):
            raise ValueError('Changed data, model, or attack schedule')
        rows.append({'condition':condition,'source_sha256':report['source_sha256'],
                     'model':report['federated_model_run'],
                     'attacked_windows':report['attacked_windows_observed'],
                     'replay_windows':report['replay_veto_per_fault']['replay']['windows'],
                     'replay_accepted_for_update':report['replay_veto_per_fault']['replay']['accepted_for_update'],
                     'gain_accepted_for_update':report['replay_veto_per_fault']['gain']['accepted_for_update'],
                     'all_fault_recall':report['mixed_stream']['replay_veto_attack_recall'],
                     'clean_rejection_rate':report['mixed_stream']['replay_veto_clean_false_positive_rate']})
    summary={'seed':args.seed,'validation_replay_cutoff':json.loads(args.replay_calibration.read_text())['chosen_calibration']['cutoff'],
             'per_condition':rows,
             'limits':['Replay noise and shifts are controlled synthetic transformations, not observed attacks.',
                       'All conditions reuse the same held-out recordings and fault schedule.',
                       'Only previously selected validation thresholds are used; no test-condition tuning.',
                       'Performance does not validate a clinical digital twin or hospital deployment.']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
    print('Saved:',args.output/'summary.json')

if __name__=='__main__':main()
