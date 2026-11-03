"""Compare saved initial/final tensors on CPU; never construct or run a model.

Use the existing experiment Python (torch and numpy already installed). Only
checkpoints whose SHA256 matches the recorded run metadata may be deserialized.
The sole output is a new small ZIP, with no checkpoint tensors included.
"""
import argparse
import ast
import csv
import gc
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RUN_NAME = ('phase4_FULL_R1_official_train_GT_intersection_DBNet_vs_ASF_'
            'mixed_10000updates_final_weights_only_seed42_v2')
VARIANTS = ('A_DBNet_noASF__mixed_clean_degraded', 'B_DBNet_plus_ASF__mixed_clean_degraded')


def initial_path(config):
    """Read a generated literal assignment, without executing a Python config."""
    tree = ast.parse(config.read_text(encoding='utf-8'))
    values = [ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'load_from' for t in node.targets)]
    if len(values) != 1 or not isinstance(values[0], str):
        raise ValueError('Expected exactly one literal load_from in ' + str(config))
    return Path(values[0])


def verified_file(path, root, expected):
    path = Path(path).resolve()
    path.relative_to(root.resolve())
    if not path.is_file() or path.stat().st_size > 512 * 1024**2:
        raise ValueError('Checkpoint missing or exceeds 512 MiB limit: ' + str(path))
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError('Missing recorded SHA256')
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('Checkpoint SHA256 does not match run record: ' + str(path))
    return path


def delta_stats(source, cast_baseline, final):
    """Separate a loading dtype conversion from changes accumulated in training."""
    source, baseline, final = [np.asarray(x, dtype=np.float32) for x in (source, cast_baseline, final)]
    if source.shape != baseline.shape or source.shape != final.shape or not final.size:
        raise ValueError('Nonempty tensors must have matching shapes')
    if any(not np.isfinite(x).all() for x in (source, baseline, final)):
        raise ValueError('Non-finite saved tensor')
    diff = final.astype(np.float64) - baseline.astype(np.float64)
    return dict(elements=int(final.size),
        changed_elements_after_initial_cast=int(np.count_nonzero(final != baseline)),
        changed_elements_vs_raw_source=int(np.count_nonzero(final != source)),
        initial_cast_changed_elements=int(np.count_nonzero(baseline != source)),
        sum_abs_delta_after_initial_cast=float(np.abs(diff).sum()),
        max_abs_delta_after_initial_cast=float(np.abs(diff).max()))


def group_for(key):
    for prefix, group in [('backbone.vitae.', 'backbone_vitae'), ('backbone.', 'backbone_other'),
                          ('neck.asf_', 'ASF'), ('neck.', 'neck_without_ASF'),
                          ('det_head.', 'det_head')]:
        if key.startswith(prefix):
            return group
    return 'other'


def compare(initial, final, torch):
    before = torch.load(initial, map_location='cpu', weights_only=False)
    after = torch.load(final, map_location='cpu', weights_only=False)
    before, after = before.get('state_dict', before), after.get('state_dict', after)
    if set(before) != set(after):
        raise ValueError('Initial/final state keys differ; do not silently drop unmatched keys')
    rows, summary = [], {}
    for key in sorted(before):
        a, b = before[key], after[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError('Unexpected non-tensor state: ' + key)
        if a.shape != b.shape:
            raise ValueError('State shape changed: ' + key)
        if not a.is_floating_point() or not b.is_floating_point() or not b.numel():
            continue
        known_buffer = key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))
        group = group_for(key) + ('__known_BN_buffers' if known_buffer else '')
        # Checkpoint loading casts into the live model dtype. Compare against
        # that baseline as well as raw source to avoid counting cast rounding
        # as learning. Saved tensors alone do not identify all parameter/buffer roles.
        baseline = a.to(dtype=b.dtype)
        stats = delta_stats(a.float().numpy(), baseline.float().numpy(), b.float().numpy())
        row = dict(key=key, group=group, initial_dtype=str(a.dtype), final_dtype=str(b.dtype),
                   shape=str(list(b.shape)), **stats)
        rows.append(row)
        entry = summary.setdefault(group, dict(elements=0, changed_elements_after_initial_cast=0,
            changed_elements_vs_raw_source=0, initial_cast_changed_elements=0,
            sum_abs_delta_after_initial_cast=0., max_abs_delta_after_initial_cast=0., final_dtype_elements={}))
        for name in ('elements', 'changed_elements_after_initial_cast', 'changed_elements_vs_raw_source',
                     'initial_cast_changed_elements', 'sum_abs_delta_after_initial_cast'):
            entry[name] += stats[name]
        entry['max_abs_delta_after_initial_cast'] = max(entry['max_abs_delta_after_initial_cast'], stats['max_abs_delta_after_initial_cast'])
        entry['final_dtype_elements'][str(b.dtype)] = entry['final_dtype_elements'].get(str(b.dtype), 0) + stats['elements']
    for entry in summary.values():
        entry['changed_fraction_after_initial_cast'] = entry['changed_elements_after_initial_cast'] / entry['elements']
        entry['mean_abs_delta_after_initial_cast'] = entry['sum_abs_delta_after_initial_cast'] / entry['elements']
    key = 'neck.asf_attn.attention_wise.conv.weight'
    projection = None
    if key in before:
        projection = {label: dict(shape=list(state[key].shape),
                                  std_population=float(state[key].float().std(unbiased=False)),
                                  max_abs=float(state[key].float().abs().max()))
                      for label, state in [('initial', before), ('final', after)]}
    del before, after
    gc.collect()
    return dict(groups=summary, attention_projection_weight_statistics=projection), rows


def run(workspace, run_name):
    import torch
    torch.set_num_threads(1)
    workspace = workspace.resolve()
    experiment = (workspace / run_name).resolve()
    experiment.relative_to(workspace)
    if experiment == workspace or not experiment.is_dir():
        raise ValueError('Expected the completed experiment inside workspace')
    configs = experiment / 'inputs/05_model_configs__only_FPNC_ASF_differs'
    results, rows, sources = {}, [], {}
    original = None
    for variant in VARIANTS:
        cfg = configs / ('training__' + variant + '__same_mixed_entries_and_optimizer_budget.py')
        metadata = experiment / 'outputs/training' / variant / ('TRAINING_COMPLETE__' + variant + '__final_weights_only.json')
        record = json.loads(metadata.read_text())
        start = verified_file(initial_path(cfg), workspace, record['initial_checkpoint_sha256'])
        finish = verified_file(Path(record['evaluation_checkpoint']), workspace, record['evaluation_checkpoint_sha256'])
        if variant == VARIANTS[0]:
            original = start
        print('CPU_COMPARE_START:', variant, flush=True)
        result, detail = compare(start, finish, torch)
        result.update(initial_checkpoint=str(start), final_checkpoint=str(finish),
                      initial_sha256=record['initial_checkpoint_sha256'],
                      final_sha256=record['evaluation_checkpoint_sha256'])
        results[variant] = result
        rows.extend(dict(variant=variant, **row) for row in detail)
        for group, value in result['groups'].items():
            print(variant, group, 'changed_after_initial_cast_pct=',
                  round(100 * value['changed_fraction_after_initial_cast'], 4),
                  'dtype_elements=', value['final_dtype_elements'], flush=True)
    # Derive the actual MMOCR root from the frozen A config rather than guessing
    # the nested checkout layout that caused the previous collector warnings.
    mmocr = original.parent.parent
    for relative in ('mmocr/models/common/backbones/monkeyocr_v2_vitae.py',
                     'mmocr/models/textdet/necks/fpn_cat.py',
                     'mmocr/models/textdet/heads/db_head.py',
                     'mmocr/models/textdet/postprocessors/db_postprocessor.py'):
        path = mmocr / relative
        if path.is_file() and path.stat().st_size < 1024**2:
            path.resolve().relative_to(workspace)
            sources['installed_source/' + relative] = path.read_bytes()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = workspace / ('AUDIT__FULL_R1_job9276__CPU_saved_weight_changes_BF16_and_ASF__' + stamp + '.zip')
    report = dict(created_utc=stamp, torch_version=torch.__version__, run_name=run_name,
        comparison='initial checkpoint versus final checkpoint, separately for A and B', results=results,
        limitations=[
            'CPU tensor inspection only: no model construction, forward, training or GPU calls.',
            'Floating state is not identical to trainable parameters. Known BatchNorm buffers are split out; other buffers may remain.',
            'Small final displacement does not prove no intermediate updates, and large displacement does not prove useful learning.',
            'Weight projection statistics do not measure attention activation saturation.',
            'All checkpoints were SHA256-verified against the run records before loading; no weights are included in the ZIP.'])
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('CHECKPOINT_UPDATE_AUDIT__A_B__initial_cast_and_final_weights.json', json.dumps(report, indent=2))
        z.writestr('PER_TENSOR__saved_state_displacements_and_dtypes.csv', table.getvalue())
        for name, data in sources.items():
            z.writestr(name, data)
    print('ZIP_TO_SEND:', output)
    print('ZIP_MiB:', round(output.stat().st_size / 1024**2, 3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-name', default=RUN_NAME)
    args = parser.parse_args()
    run(args.workspace, args.run_name)
