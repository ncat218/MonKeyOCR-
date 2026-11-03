"""FP32 control: preserve frozen job9276 inputs and exact ASF initialization."""
from pathlib import Path

from .common import identity, jsonl, read_json, save_json, sha256


def apply_precision(cfg, precision):
    if precision not in ('native', 'fp32'):
        raise ValueError('parameter_precision must be native or fp32')
    if precision == 'fp32':
        cfg.model.backbone['use_bf16'] = False
        cfg['textdet_parameter_precision'] = 'fp32'


def enforce_fp32(model):
    """Cast before checkpoint loading/optimizer construction, then verify storage."""
    import torch
    model.float()
    verify_fp32(model)


def verify_fp32(model):
    import torch
    bad = [k for k, p in model.named_parameters()
           if p.is_floating_point() and p.dtype != torch.float32]
    if bad:
        raise RuntimeError('FP32 parameter-storage guard failed: ' + str(bad[:10]))


def runtime_math_flags():
    import torch
    return dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
                note='FP32 parameter storage is verified separately; backend TF32 flags are recorded, not silently equated to parameter dtype.')


def validate_control(config):
    import re
    control = config['precision_control']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', control['reference_run']):
        raise ValueError('Reference run must be a workspace directory name')
    if control['reference_run'] == config['run_name']:
        raise ValueError('Use a new run_name for the FP32 control')
    hashes = [control[k] for k in ('reference_source_fingerprint','master_sha256','protocol_sha256','ASF_initial_sha256')]
    if set(control['split_sha256']) != {'train','validation','test','tiling_development'}:
        raise ValueError('Pin all four reference split manifests')
    hashes += list(control['split_sha256'].values())
    if control.get('reference_comparison_sha256'):
        hashes.append(control['reference_comparison_sha256'])
    if any(not isinstance(h, str) or not re.fullmatch('[a-f0-9]{64}', h) for h in hashes):
        raise ValueError('Reference hashes must be full SHA256 hex digests')
    initial = Path(control['ASF_initial_relative_path'])
    if initial.is_absolute() or '..' in initial.parts:
        raise ValueError('Reference ASF initial must be a relative path inside the old run')
    count = control['diagnostic_validation_sources']
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError('Choose a positive fixed number of validation diagnostic sources')


def reference_paths(workspace, config):
    ref = config['precision_control']
    root = (Path(workspace)/ref['reference_run']).resolve()
    root.relative_to(Path(workspace).resolve())
    if root == (Path(workspace)/config['run_name']).resolve():
        raise ValueError('Precision control must use a separate run_name')
    return root


def verify_probe_pair(probes):
    from .variants import VARIANTS
    for variant in VARIANTS:
        probe = probes[variant]
        if (probe.get('parameter_storage') != 'float32' or
                probe.get('actual_OptimWrapper_with_declared_accumulation_clipping_and_warmup') is not True or
                probe.get('observed_optimizer_updates') != 1):
            raise RuntimeError('Expected a real FP32 first-update probe: '+variant)
    a, b = [probes[v] for v in VARIANTS]
    if (a['sample_ids_by_microiteration'] != b['sample_ids_by_microiteration'] or
            a['optimizer_lr_before_each_microiteration'] != b['optimizer_lr_before_each_microiteration']):
        raise RuntimeError('A/B probes did not use the same sample order and first-update LR')


def initial_validation_rows(manifest, count):
    """Fixed validation-only diagnostics, including the largest resized canvas.

    This anticipates FP32 inference memory demand before a long training, without
    consulting test images, model outputs or GT difficulty to choose parameters.
    """
    import math
    first, largest, largest_area = [], None, -1
    for row in jsonl(manifest):
        if len(first) < count:
            first.append(row)
        width, height = row['clean_size']
        factor = min(4068/max(width,height), 1024/min(width,height))
        dims = [int(v*factor+.5) for v in (width,height)]
        area = math.ceil(dims[0]/32)*32*math.ceil(dims[1]/32)*32
        if area > largest_area:
            largest, largest_area = row, area
    if largest is None:
        raise ValueError('No validation source for initial inference diagnostics')
    if largest['source_image_id'] not in {r['source_image_id'] for r in first}:
        first[-1] = largest
    return first


def reference_stages(root, verify_hashes):
    results = {}
    for key, stage in (
        ('shared', '01_audit_full_MAIN_and_freeze_official_train_GT_intersection'),
        ('splits', '02_freeze_same_source_clean_SHA_grouped_train_validation_test'),
        ('annotations', '03_export_same_mixed_targets_for_DBNet_and_ASF')):
        marker = root/'state'/(stage+'__stage_status.json')
        if not marker.is_file():
            raise RuntimeError('Keep job9276 inputs and state; missing '+str(marker))
        record = read_json(marker)
        if record.get('status') != 'completed':
            raise RuntimeError('Reference input stage was not completed: '+stage)
        for name, digest in record['artifacts'].items():
            path = Path(name).resolve()
            path.relative_to(root)
            if not path.is_file() or (verify_hashes and sha256(path) != digest):
                raise RuntimeError('Reference input missing/modified: '+str(path))
        results[key] = record['result']
    return results


def pre_submit_reference(workspace, config):
    """Small JSON/existence checks on login; bulk hashing stays inside the job."""
    ref = config['precision_control']
    root = reference_paths(workspace, config)
    completion = read_json(root/'COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json')
    if completion['source_fingerprint'] != ref['reference_source_fingerprint']:
        raise ValueError('Unexpected reference run source fingerprint')
    previous = read_json(root/'run_configuration__FULL_R1_two_architectures_same_budget.json')
    current_training = dict(config['training'])
    current_training.pop('parameter_precision', None)
    if (current_training != previous['training'] or config['seed'] != previous['seed'] or
            config['split'] != previous['split'] or
            config['mask_thresholds_validation_only'] != previous['mask_thresholds_validation_only']):
        raise ValueError('Precision control must preserve the reference optimizer schedule, seed, splits and threshold candidates')
    reference_stages(root, verify_hashes=False)
    initial = root/ref['ASF_initial_relative_path']
    initial.resolve().relative_to(root)
    if not initial.is_file():
        raise RuntimeError('Keep the exact job9276 ASF initialization: '+str(initial))
    return root


def bind_reference(workspace, config, paths, shared_dir, split_dir, annotations_dir):
    """Revalidate frozen bytes; borrow large GT files read-only, copy tiny receipts."""
    root = pre_submit_reference(workspace, config)
    ref = config['precision_control']
    records = reference_stages(root, verify_hashes=True)
    shared, splits, annotations = records['shared'], records['splits'], records['annotations']
    if shared['manifest_sha256'] != ref['master_sha256'] or sha256(shared['manifest']) != ref['master_sha256']:
        raise ValueError('Reference master does not match the agreed job9276 manifest')
    for fold, expected in ref['split_sha256'].items():
        if sha256(splits[fold]) != expected:
            raise ValueError('Reference split identity changed: '+fold)
    old_shared = Path(shared['manifest']).parent
    old_audit = read_json(old_shared/'data_audit__FULL_R1_counts_GT_intersection_and_exclusions.json')
    from .full_data import read_metadata
    _, current_metadata = read_metadata(paths)
    source_files = dict(current_metadata['metadata_sha256'], main=sha256(paths['main_image_manifest']),
                        clean=sha256(paths['clean_json']), textocr=sha256(paths['textocr_json']))
    original_stage = read_json(root/'state/01_audit_full_MAIN_and_freeze_official_train_GT_intersection__stage_status.json')
    if original_stage['input_fingerprint'] != identity(dict(files=source_files, source=ref['reference_source_fingerprint'])):
        raise ValueError('Source metadata files changed since the reference audit')
    if (sha256(paths['clean_json']) != old_audit['clean_annotation_sha256'] or
            sha256(paths['textocr_json']) != old_audit['textocr_sha256']):
        raise ValueError('Source annotation bytes changed since job9276')
    protocol = old_shared/'evaluation_protocol.json'
    if sha256(protocol) != ref['protocol_sha256']:
        raise ValueError('Reference evaluation protocol changed')
    locations = dict(clean_root=paths['clean_images_root'], degraded_root=paths['degraded_root'])
    previous = read_json(old_shared/'dataset_locations.json')
    if {k:str(Path(v).resolve()) for k,v in locations.items()} != {k:str(Path(v).resolve()) for k,v in previous.items()}:
        raise ValueError('Reference training annotations contain absolute image paths; dataset roots changed')
    # Existing frozen geometry is reused, but every image used for training or
    # evaluation is checked again. No decoder/GT regeneration or large-file copy.
    count = 0
    for row in jsonl(shared['manifest']):
        for condition in ('clean', 'degraded'):
            path = (Path(locations[condition+'_root'])/row[condition+'_relpath']).resolve()
            path.relative_to(Path(locations[condition+'_root']).resolve())
            if not path.is_file() or sha256(path) != row[condition+'_sha256']:
                raise ValueError('Frozen source image missing/changed: '+str(path))
        count += 1
        if count % 500 == 0:
            print('FP32 REFERENCE IMAGE HASH AUDIT', count, '/', shared['sources'], flush=True)
    if count != shared['sources']:
        raise ValueError('Reference source count mismatch')
    initial = root/ref['ASF_initial_relative_path']
    if sha256(initial) != ref['ASF_initial_sha256']:
        raise ValueError('ASF initial bytes changed; do not draw a fresh initialization')
    for folder in (shared_dir, split_dir, annotations_dir):
        folder.mkdir(parents=True, exist_ok=True)
    for name in ('evaluation_protocol.json', 'data_audit__FULL_R1_counts_GT_intersection_and_exclusions.json'):
        (shared_dir/name).write_bytes((old_shared/name).read_bytes())
    policy = Path(splits['train']).parent/'split_policy_and_counts.json'
    (split_dir/policy.name).write_bytes(policy.read_bytes())
    receipt = dict(reference_run=str(root), shared=shared, splits=splits, annotations=annotations,
        ASF_initial=str(initial), ASF_initial_sha256=sha256(initial), sources_rehashed=count,
        policy='Large frozen inputs are referenced read-only, never copied or regenerated. Keep reference inputs/state and ASF initialization.',
        factor='FP32 parameter storage and computation in training/inference; ASF initialization, optimizer schedule, targets and folds unchanged.')
    save_json(shared_dir/'REFERENCE__job9276_frozen_inputs_and_image_hashes.json', receipt)
    save_json(annotations_dir/'REFERENCE__read_only_job9276_training_targets.json', annotations)
    historical = root/'outputs/FINAL_COMPARISON__FULL_R1_DBNet_vs_ASF__native_and_validation_selected_masks/COMPARISON__DBNet_ASF_native_and_selected_masks__all_metrics_and_word_errors.json'
    if historical.is_file() and ref.get('reference_comparison_sha256'):
        if sha256(historical) != ref['reference_comparison_sha256']:
            raise ValueError('Historical comparison differs from the report supplied for job9276')
        (shared_dir/'REFERENCE_RESULTS__job9276_BF16_regime.json').write_bytes(historical.read_bytes())
    return dict(shared=shared, splits=splits, annotations=annotations, ASF_initial=str(initial))
