"""Collect existing FULL_R1 evidence with the Python standard library only.

No torch, environment changes, model execution, Slurm submission, or deletion.
The only output is a new ZIP; existing experiment files are opened read-only.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RUN_NAME = ('phase4_FULL_R1_official_train_GT_intersection_DBNet_vs_ASF_'
            'mixed_10000updates_final_weights_only_seed42_v2')
VARIANTS = ('A_DBNet_noASF__mixed_clean_degraded',
            'B_DBNet_plus_ASF__mixed_clean_degraded')
METRICS = ('clean_hmean', 'degraded_hmean', 'balanced_clean_degraded_hmean')
MAX_FILE = 8 * 1024**2
MAX_TOTAL = 64 * 1024**2
REPO = Path(__file__).resolve().parents[1]


def inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validation_rows(text, variant, source):
    """Keep the raw logger step, without pretending it is an optimizer update.

    JSON scalar steps and textual last-train-iteration context are deliberately
    separate: LoggerHook validation output may not print a train iteration.
    """
    rows = []
    last_train = None
    for number, line in enumerate(text.splitlines(), 1):
        match = re.search(r'Iter\(train\)\s*\[\s*(\d+)\s*/', line)
        if match:
            last_train = int(match[1])
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            obj = None
        values = {}
        raw_step = None
        if isinstance(obj, dict):
            values = {str(k).split('/')[-1]: v for k, v in obj.items()
                      if str(k).split('/')[-1] in METRICS}
            raw_step = obj.get('step', obj.get('iter'))
        else:
            for key in METRICS:
                match = re.search(r'(?<![\w/])(?:shared/)?' + key
                                  + r':\s*([-+\d.eE]+)', line)
                if match:
                    try:
                        values[key] = float(match[1])
                    except ValueError:
                        pass
        if not all(k in values for k in METRICS[:2]):
            continue
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               or not math.isfinite(v) or not 0 <= v <= 1
               for v in values.values()):
            continue
        rows.append(dict(variant=variant, source=source, line=number,
                         raw_logger_step=raw_step,
                         last_train_microiteration_in_this_text=last_train,
                         **{k: values.get(k) for k in METRICS}))
    return rows


def selected_files(run):
    """Explicit evidence areas only: never datasets, manifests or weights."""
    configs = run / 'inputs/05_model_configs__only_FPNC_ASF_differs'
    for path in sorted(configs.glob('*')):
        if path.is_file() and path.suffix in ('.py', '.json'):
            yield path, 'model_config_or_initialization_receipt', None
    for variant in VARIANTS:
        training = run / 'outputs/training' / variant
        for path in sorted(training.rglob('*')):
            if path.is_file() and path.suffix in ('.log', '.json', '.py'):
                yield path, 'training_log_or_metadata', variant
    for path in sorted((run / 'logs').rglob('*.log')):
        yield path, 'worker_or_slurm_log', None
    for name in ('READ_ME_FIRST__FULL_R1_DBNet_ASF_run_outputs_and_logs.json',
                 'COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json'):
        yield run / name, 'run_metadata', None


def collect(workspace, run_name=RUN_NAME, job_id='9276', output=None):
    workspace = Path(workspace).resolve()
    run = workspace / run_name
    if not inside(run, workspace) or run.resolve() == workspace or not run.is_dir():
        raise ValueError('Expected an existing experiment directory inside workspace')
    if not re.fullmatch(r'[0-9]+', job_id):
        raise ValueError('job-id must contain digits only')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output).resolve() if output else workspace / (
        'DIAGNOSTICS__FULL_R1__DBNet_ASF_initialization_training_validation_'
        'and_source_snapshot__job' + job_id + '__' + stamp + '.zip')
    if not inside(output, workspace) or inside(output, run) or output.suffix.lower() != '.zip':
        raise ValueError('Output must be a new ZIP in workspace, outside the experiment directory')
    if output.exists():
        raise FileExistsError('Refusing to overwrite: ' + str(output))
    entries, warnings, curves, receipts, probes = [], [], [], {}, {}
    payload = []
    total = 0

    def add(path, archive_name, category, variant=None, root=run):
        nonlocal total
        if not path.is_file():
            warnings.append('Missing evidence: ' + str(path))
            return
        if not inside(path, root):
            warnings.append('Skipped redirected path outside declared root: ' + str(path))
            return
        size = path.stat().st_size
        truncated = size > MAX_FILE
        if truncated and path.suffix != '.log':
            warnings.append('Skipped oversized non-log: ' + str(path))
            return
        read_size = min(size, MAX_FILE)
        if total + read_size > MAX_TOTAL:
            warnings.append('Collection size limit reached; skipped: ' + str(path))
            return
        with path.open('rb') as f:
            if truncated:
                f.seek(-MAX_FILE, 2)
            data = f.read(MAX_FILE)
        if truncated:
            warnings.append('Only last 8 MiB included: ' + str(path))
        total += len(data)
        payload.append((archive_name, data))
        entries.append(dict(source=str(path), archive_name=archive_name, category=category,
                            original_bytes=size, included_bytes=len(data), truncated=truncated,
                            included_bytes_sha256=hashlib.sha256(data).hexdigest()))
        text = data.decode('utf-8', errors='replace')
        if variant and category == 'training_log_or_metadata':
            curves.extend(validation_rows(text, variant, archive_name))
            if path.name.startswith('COMPATIBILITY__'):
                try:
                    probes[variant] = json.loads(text)
                except ValueError:
                    warnings.append('Unreadable compatibility JSON: ' + str(path))
        if path.name.startswith('INITIALIZATION_RECEIPT__'):
            try:
                receipt = json.loads(text)
                receipts[receipt['variant']] = receipt
            except (ValueError, KeyError, TypeError):
                warnings.append('Unreadable initialization receipt: ' + str(path))

    for path, category, variant in selected_files(run):
        add(path, 'experiment/' + path.relative_to(run).as_posix(), category, variant)
    for name in ('run_all.py', 'configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json',
                 'src/monkey_textdet/variants.py', 'src/monkey_textdet/training.py',
                 'src/monkey_textdet/asf_suite.py', 'src/monkey_textdet/mask_sweep.py',
                 'src/monkey_textdet/mmocr_plugins.py'):
        add(REPO / name, 'repository_at_collection/' + name, 'current_repository_snapshot', root=REPO)
    upstream = workspace / ('environment_and_models__phase1_verified_stack/'
                            'upstream_MonkeyOCRv2/detection/mmocr')
    for name in ('mmocr/models/textdet/necks/fpn_cat.py',
                 'mmocr/models/textdet/heads/db_head.py',
                 'mmocr/models/textdet/postprocessors/db_postprocessor.py',
                 'mmocr/models/common/backbones/monkeyocr_v2_vitae.py'):
        add(upstream / name, 'installed_upstream_snapshot/' + name, 'upstream_source_snapshot', root=workspace)
    site = workspace / ('environment_and_models__phase1_verified_stack/'
                        'python311_venv/lib/python3.11/site-packages')
    for name in ('mmengine/model/weight_init.py',
                 'mmengine/optim/optimizer/default_constructor.py',
                 'mmengine/optim/optimizer/optimizer_wrapper.py'):
        add(site / name, 'installed_optimizer_and_initialization/' + name,
            'installed_optimizer_source_snapshot', root=workspace)

    evidence = {}
    for variant in VARIANTS:
        receipt, probe = receipts.get(variant), probes.get(variant)
        records = [r for r in curves if r['variant'] == variant]
        if not receipt:
            warnings.append('No initialization receipt for ' + variant)
        if not probe:
            warnings.append('No compatibility probe receipt for ' + variant)
        if not records:
            warnings.append('No validation curve parsed for ' + variant + '; inspect original logs')
        evidence[variant] = dict(initialization_receipt_available=bool(receipt),
            strict_load_recorded=receipt.get('strict_full_state_load_passed') if receipt else None,
            shared_weights_equal_recorded=receipt.get('shared_weight_equality_passed') if receipt else None,
            new_parameter_tensor_count=len(receipt.get('newly_initialized_keys', [])) if receipt else None,
            ASF_tensors_with_finite_probe_gradients=probe.get('ASF_parameter_tensors_with_finite_gradients') if probe else None,
            validation_records_parsed_including_duplicate_log_formats=len(records))
    fields = ['variant', 'source', 'line', 'raw_logger_step',
              'last_train_microiteration_in_this_text', *METRICS]
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=fields)
    writer.writeheader()
    writer.writerows(curves)
    try:
        git = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, timeout=10)
        head = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        head = None
    manifest = dict(run_name=run_name, job_id_supplied_by_user=job_id, collected_utc=stamp,
        repository_HEAD_at_collection=head, included_uncompressed_bytes=total,
        limits=dict(max_bytes_per_file=MAX_FILE, max_total_source_bytes=MAX_TOTAL),
        entries=entries, warnings=warnings, recorded_evidence=evidence,
        limitations=[
            'No GPU work or checkpoint tensor inspection; receipts are recorded evidence, not fresh verification.',
            'No initial ASF forward evaluation or attention measurements were performed.',
            'No checkpoint parameter-change comparison; use_bf16 is not proof that backbone updates vanished.',
            'Repository snapshot is current at collection, not proof it equals the job source fingerprint.',
            'CSV may contain duplicate validation points from scalar JSON and text logs; rates are fractions.',
            'Raw logger step and preceding train microiteration are separate; confirm axes before fitting trends.',
            'Logs over 8 MiB are tails; missing/truncated files are explicitly reported.'])
    readme = (
        '# DBNet / ASF training diagnostics\n\n'
        'Read COLLECTION_MANIFEST.json first, especially warnings and limitations.\n'
        'VALIDATION_CURVES__raw_steps_and_source_lines.csv contains parsed Hmean history.\n'
        'Compare duplicate points using their source and raw step, not their row count.\n'
        'experiment/ contains existing model configs, initialization receipts, compatibility probes, '
        'training scalars/text logs and worker logs. No checkpoints, images or dataset manifests included.\n'
        'repository_at_collection/ and installed_upstream_snapshot/ help inspect implementation; '
        'they are labelled as snapshots at collection time.\n'
        'This collector does not retrain, infer, change an environment, submit a job, or delete anything.\n')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for name, content in payload:
            z.writestr(name, content)
        z.writestr('READ_ME_FIRST__what_was_collected_and_what_is_not_verified.md', readme)
        z.writestr('COLLECTION_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        z.writestr('VALIDATION_CURVES__raw_steps_and_source_lines.csv', table.getvalue())
    return output, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-name', default=RUN_NAME)
    parser.add_argument('--job-id', default='9276')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output, manifest = collect(args.workspace, args.run_name, args.job_id, args.output)
    print('ZIP_TO_SEND:', output)
    print('ZIP_MiB:', round(output.stat().st_size / 1024**2, 2))
    print('VALIDATION_EVIDENCE:', json.dumps(manifest['recorded_evidence'], ensure_ascii=False))
    for warning in manifest['warnings']:
        print('COLLECTION_WARNING:', warning)
    print('No GPU/model execution; existing experiment files were read only.')


if __name__ == '__main__':
    main()
