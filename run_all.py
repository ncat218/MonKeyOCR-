#!/usr/bin/env python3
"""One command: auto-submit one Slurm job, bootstrap, then run all Phase 3 experiments."""
from pathlib import Path
import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

# THE ONE DATA LOCATION LINE. Set this later to the parent containing all three datasets.
DATASET_PARENT = ""

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'src'))


def workspace_for(config, explicit=''):
    if explicit or config.get('workspace_root'):
        return Path(explicit or config['workspace_root']).expanduser().resolve()
    datastore = Path('/datastore') / getpass.getuser()
    parent = datastore if datastore.is_dir() else REPO.parent
    return parent / '_workspace_monkeyocr_text_detection_phase3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(REPO / 'configs/phase3_h200_normal.json'))
    parser.add_argument('--workspace', default='')
    parser.add_argument('--dataset-parent', default='')
    parser.add_argument('--in-allocation', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--plan-only', action='store_true', help='Read config only; no setup, dataset access or GPU job.')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    from monkey_textdet.pipeline import validate_config
    validate_config(config)
    if not re.fullmatch('[A-Za-z0-9_-]+', config['run_name']):
        raise ValueError('run_name must contain only letters, digits, underscores and hyphens')
    workspace = workspace_for(config, args.workspace)
    dataset_parent = (args.dataset_parent or os.environ.get('TEXTDET_DATA_PARENT', '') or
                      DATASET_PARENT or config.get('dataset_parent', ''))
    if args.plan_only:
        print(json.dumps(dict(workspace=str(workspace), dataset_parent=dataset_parent or 'AUTO /datastore/<user> and /datastore/shared',
                              configuration=config), ensure_ascii=False, indent=2))
        return
    scheduler = config['scheduler']
    in_slurm = bool(os.environ.get('SLURM_JOB_ID'))
    if scheduler['auto_submit'] and not in_slurm and not args.in_allocation and shutil.which('sbatch'):
        # Only lightweight submission happens on the login node. All setup/audit/GPU work is inside the job.
        from monkey_textdet.common import utc_stamp
        log_dir = workspace / config['run_name'] / 'logs' / 'slurm_all_experiments'
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_stamp()
        script = log_dir / f'{stamp}__submit_all_phase3_experiments_normal.sh'
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--in-allocation',
                   '--config', str(config_path), '--workspace', str(workspace)]
        if dataset_parent:
            command += ['--dataset-parent', dataset_parent]
        script.write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(REPO)) +
                          '\nexec ' + shlex.join(command) + '\n', encoding='utf-8')
        cmd = ['sbatch', '--parsable', '--job-name=monkey_phase3_all_E0_tiles_E1_E2',
               '--ntasks=1',
               '--account=' + scheduler['account'], '--partition=' + scheduler['partition'],
               '--gres=' + scheduler['gres'], '--cpus-per-task=' + str(scheduler['cpus']),
               '--mem=' + str(scheduler['ram_gb']) + 'G', '--time=' + scheduler['walltime'],
               '--signal=B:USR1@180',
               '--output=' + str(log_dir / (stamp + '__all_phase3_experiments__job_%j.stdout.log')),
               '--error=' + str(log_dir / (stamp + '__all_phase3_experiments__job_%j.stderr.log')), str(script)]
        job = subprocess.check_output(cmd, text=True).strip()
        print('Submitted ONE job for the complete experiment suite:', job)
        print('Slurm logs:', log_dir)
        print('If interrupted, run the same python run_all.py command to resume; do not change run_name.')
        return
    if sys.platform != 'linux':
        raise RuntimeError('Actual execution requires a Linux GPU allocation. Use --plan-only on Windows.')
    if args.in_allocation and not in_slurm:
        raise RuntimeError('--in-allocation requires SLURM_JOB_ID; do not run compute on the HPC login node.')
    if shutil.which('sbatch') and not in_slurm:
        raise RuntimeError('Slurm detected without an allocation; enable scheduler.auto_submit.')
    from monkey_textdet.bootstrap import launch
    launch(REPO, config_path, workspace, dataset_parent)


if __name__ == '__main__':
    main()
