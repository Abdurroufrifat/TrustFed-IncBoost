"""Run, resume, and verify the full PhysioNet feasibility experiment.

Run from repository root. Outputs are written under outputs/; raw data stays local.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

STEPS=(
    ('source_audit','13_audit_physionet_training.py','outputs/physionet_2015_audit.json'),
    ('preparation','14_prepare_physionet_sensor_study.py','outputs/physionet_2015_prepared_v3/preparation_summary.json'),
    ('replay_calibration','18_calibrated_replay.py','outputs/physionet_2015_robust_replay/results.json'),
    ('federated_xgboost','20_run_physionet_repository_xgboost.py','outputs/physionet_repository_xgboost/results.json'),
    ('record_splits','21_repeat_physionet_record_splits.py','outputs/physionet_xgboost_three_splits/summary.json'),
    ('causal_state','22_audit_causal_signal_state.py','outputs/physionet_causal_signal_audit/summary.json'),
    ('fault_schedules','25_repeat_replay_veto_schedules.py','outputs/physionet_replay_veto_schedule_check/summary.json'),
    ('replay_stress','26_stress_replay_state_updates.py','outputs/physionet_replay_state_stress/summary.json'),
)


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
    return sha.hexdigest()


def verify(name,path,source):
    if not path.is_file():raise FileNotFoundError(path)
    data=json.loads(path.read_text())
    if name=='source_audit':
        if data.get('source_sha256')!=source:raise ValueError('Source audit hash mismatch')
        return {'path':str(path),'file_exists':True}
    if name=='replay_stress':
        rows=data['per_condition']
        if {r['condition'] for r in rows}!={'exact','noise_0.01','noise_0.05','shift_5','shift_25'}:
            raise ValueError('Incomplete replay stress conditions')
        hashes={r['source_sha256'] for r in rows}
        if hashes!={source}:raise ValueError('Replay stress source hash mismatch')
        if not all((path.parent/r['condition']/'per_record.csv').is_file() for r in rows):
            raise FileNotFoundError('Missing replay stress per-record outputs')
        return {'path':str(path),'conditions':len(rows)}
    key='source_sha256'
    if data.get(key)!=source:raise ValueError(f'{name} source hash mismatch: {path}')
    if name=='preparation' and data.get('preparation_version')!=3:
        raise ValueError('Prepared data version must be 3')
    if name=='preparation' and not all((path.parent/f).is_file()
                                      for f in ('sensor_windows.csv','record_split_manifest.csv')):
        raise FileNotFoundError('Missing prepared sensor windows or split manifest')
    if name=='federated_xgboost' and set(data.get('runs',{}))!={'clean','label_flip_client_0'}:
        raise ValueError('Federated runs incomplete')
    if name=='federated_xgboost' and not list(Path('outputs/physionet_repository_pipeline').glob(
            'physionet_repository_xgboost_clean_*/federated_ensemble.joblib')):
        raise FileNotFoundError('Missing fitted clean federated model')
    if name=='record_splits' and len(data.get('seeds',{}))!=3:
        raise ValueError('Expected three recording split seeds')
    if name=='fault_schedules' and len(data.get('per_seed',[]))!=4:
        raise ValueError('Expected four fault schedules')
    return {'path':str(path),'file_exists':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip',type=Path,default=Path('data/raw/physionet_2015/training.zip'))
    ap.add_argument('--force',action='store_true',help='Rerun stages even when verified output exists')
    ap.add_argument('--verify-only',action='store_true',help='Check results without running experiments')
    ap.add_argument('--dry-run',action='store_true',help='List stages without executing')
    ap.add_argument('--manifest',type=Path,default=Path('outputs/physionet_project_final/manifest.json'))
    args=ap.parse_args()
    if not args.zip.is_file():raise FileNotFoundError(f'Place source ZIP at {args.zip}')
    source=digest(args.zip)
    root=Path(__file__).resolve().parent
    checks={}
    for name,script,relative in STEPS:
        path=Path(relative)
        reason=None
        if path.is_file():
            try:checks[name]=verify(name,path,source)
            except (ValueError,KeyError,TypeError,FileNotFoundError) as e:reason=str(e)
        else:reason='missing result'
        if args.dry_run:
            print(name,'RUN' if args.force or reason else 'REUSE',reason or '')
            continue
        if args.verify_only:
            if reason:raise ValueError(f'{name}: {reason}')
            print('Verified',name,flush=True)
            continue
        if args.force or reason:
            command=[sys.executable,str(root/script)]
            if name in ('source_audit','preparation','replay_calibration','causal_state','fault_schedules','replay_stress'):
                command+=['--zip',str(args.zip)]
            print('Running',name,'because',reason or '--force',flush=True)
            subprocess.run(command,check=True)
            checks[name]=verify(name,path,source)
        else:print('Reusing',name,flush=True)
    if args.dry_run:return
    manifest={'status':'feasibility_study_complete','source_sha256':source,
              'steps':checks,'scope':'simulated clients and synthetic ECG/PLETH faults on 2015 recordings',
              'limits':['No verified patient or hospital identities.',
                        'Clinical alarm verdicts are not cybersecurity ground truth.',
                        'No independently validated patient digital twin or observed attack cohort.',
                        'Modified replay at 5% noise and 25-sample shift can bypass the fixed veto.']}
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved:',args.manifest)

if __name__=='__main__':main()
