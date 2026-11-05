"""Read-only binding to completed job9793; never recreate its data or weights."""
from pathlib import Path

from .common import identity, jsonl, read_json, save_json, sha256
from .mask_sweep import prediction_file

AP = 'A_DBNet_noASF__mixed_clean_degraded'
COMPLETE = 'COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json'
RECEIPT = 'inputs/02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT/REFERENCE__job9276_frozen_inputs_and_image_hashes.json'
INFER_CONFIG = 'inputs/05_model_configs__only_FPNC_ASF_differs/inference__'+AP+'__native_global_resolution.py'
SELECTION = 'outputs/validation_selection/SELECTED_MASK__'+AP+'__validation_only.json'
COMPARISON = ('outputs/FINAL_COMPARISON__FULL_R1_DBNet_vs_ASF__native_and_validation_selected_masks/'
              'COMPARISON__FP32_DBNet_ASF_native_and_selected_masks__all_metrics_and_word_errors.json')


def inside(path, root):
    path = Path(path).resolve()
    path.relative_to(Path(root).resolve())
    return path


def reference_files(workspace, config):
    """Small login-node check only: no model, large manifest, or image hashing."""
    ref = config['reference_job9793']
    root = inside(Path(workspace)/ref['run_name'], workspace)
    if root == (Path(workspace)/config['run_name']).resolve():
        raise ValueError('Use a NEW run_name; reference job9793 is read-only')
    for relative, digest in ref['small_file_sha256'].items():
        path = inside(root/relative, root)
        if not path.is_file():
            raise FileNotFoundError('Keep job9793 inputs, state and A predictions: '+str(path))
        if sha256(path) != digest:
            raise ValueError('job9793 evidence changed: '+str(path))
    completion = read_json(root/COMPLETE)
    if completion['source_fingerprint'] != ref['source_fingerprint']:
        raise ValueError('Wrong completed reference source version')
    receipt = read_json(root/RECEIPT)
    for fold in ('train', 'validation', 'test'):
        path = inside(receipt['splits'][fold], workspace)
        if not path.is_file():
            raise FileNotFoundError('Keep frozen job9276 inputs: '+str(path))
    for fold in ('validation', 'test'):
        masks = config['mask_thresholds_validation_only'] if fold == 'validation' else [.3, .2]
        for mask in masks:
            path = prediction_file(root/'outputs/global_predictions'/AP/fold, mask)
            if not path.is_file():
                raise FileNotFoundError('Missing original A′ raw predictions (not in the small ZIP): '+str(path))
    return root


def verify_recorded_stage(root, stage, required):
    marker = read_json(root/'state'/(stage+'__stage_status.json'))
    if marker['status'] != 'completed':
        raise ValueError('Reference stage is not completed: '+stage)
    artifacts = marker['artifacts']
    for filename in required:
        path = inside(filename, root)
        digest = artifacts.get(str(path))
        if not digest or not path.is_file() or sha256(path) != digest:
            raise ValueError('Missing/changed recorded reference artifact: '+str(path))
    return {str(Path(p).resolve()):artifacts[str(Path(p).resolve())] for p in required}


def verify_prediction_records(path, manifest, locations, checkpoint_sha, config_sha, sweep_identity):
    """Validate every cached record, not just the existence of a predictions file."""
    from .inference import tasks_for
    tasks = {t['sample_id']: t for t in tasks_for(manifest, locations)}
    seen = set()
    manifest_sha = sha256(manifest)
    for row in jsonl(path):
        key = row['sample_id']
        if key in seen or key not in tasks:
            raise ValueError('Duplicate/unexpected cached prediction: '+key)
        seen.add(key)
        task = tasks[key]
        for field in ('source_image_id', 'condition', 'image_sha256'):
            if row[field] != task[field]:
                raise ValueError('Cached prediction task mismatch: '+key+' '+field)
        if (row['manifest_sha256'], row['checkpoint_sha256'], row['inference_config_sha256'],
                row['inference_identity']) != (manifest_sha, checkpoint_sha, config_sha, identity(sweep_identity)):
            raise ValueError('Cached prediction identity mismatch: '+key)
        if row['mask_threshold'] not in sweep_identity['masks']:
            raise ValueError('Cached mask is outside recorded sweep')
    if seen != set(tasks):
        raise ValueError('Incomplete cached predictions: '+str(path))
    return len(seen)


def bind_reference(workspace, config, locations, output):
    """Allocation-only: hash frozen folds, relevant image bytes and cached predictions."""
    from .inference import tasks_for
    root = reference_files(workspace, config)
    ref = config['reference_job9793']
    receipt = read_json(root/RECEIPT)
    splits = {fold: Path(receipt['splits'][fold]) for fold in ('train','validation','test')}
    for fold, path in splits.items():
        if sha256(path) != ref['split_sha256'][fold]:
            raise ValueError('Changed frozen '+fold+' manifest')
    master = inside(receipt['shared']['manifest'], workspace)
    if sha256(master) != ref['master_sha256']:
        raise ValueError('Changed frozen master manifest')
    protocol = master.parent/'evaluation_protocol.json'
    if sha256(protocol) != ref['protocol_sha256']:
        raise ValueError('Changed evaluation protocol')
    groups = {}
    for fold, path in splits.items():
        rows = list(jsonl(path))
        sources = {r['source_image_id'] for r in rows}
        if len(rows) != ref['source_counts'][fold] or len(sources) != len(rows):
            raise ValueError('Expected one MAIN pair per frozen source: '+fold)
        groups[fold] = sources
    if any(groups[a] & groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Source leakage between frozen folds')
    # Byte-identical source images also must not cross folds.
    sha_groups = {f:{r['clean_sha256'] for r in jsonl(p)} for f,p in splits.items()}
    if any(sha_groups[a] & sha_groups[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Clean SHA leakage between frozen folds')
    checked = 0
    for fold in ('validation','test'):
        for task in tasks_for(splits[fold], locations):
            if not Path(task['path']).is_file() or sha256(task['path']) != task['image_sha256']:
                raise ValueError('Current dataset differs from job9793: '+task['path'])
            checked += 1
            if checked % 500 == 0:
                print('E0/A′ FROZEN VALIDATION/TEST IMAGE AUDIT', checked, '/',
                      2*(ref['source_counts']['validation']+ref['source_counts']['test']), flush=True)
    config_sha = sha256(root/INFER_CONFIG)
    predictions, verified_artifacts, sweep_identities = {}, {}, {}
    for fold in ('validation','test'):
        folder = root/'outputs/global_predictions'/AP/fold
        masks = config['mask_thresholds_validation_only'] if fold == 'validation' else [.3,.2]
        paths = [prediction_file(folder, mask) for mask in masks]
        stage = ('07_validation_masks_' if fold == 'validation' else '10_test_native_and_locked_mask_')+AP
        verified_artifacts.update(verify_recorded_stage(root, stage, paths+[folder/'immutable_map_sweep_identity.json',
                                                folder/'COMPLETE__all_requested_mask_predictions.json']))
        signature = read_json(folder/'immutable_map_sweep_identity.json')
        sweep_identities[fold] = signature
        if signature['checkpoint_sha256'] != ref['checkpoint_sha256'] or signature['inference_config_sha256'] != config_sha:
            raise ValueError('Wrong A′ checkpoint/config in saved sweep')
        if signature.get('runtime_math_flags') != ref['runtime_math_flags']:
            raise ValueError('Wrong A′ runtime math flags in saved sweep')
        predictions[fold] = {}
        for mask, path in zip(masks, paths):
            verify_prediction_records(path, splits[fold], locations, ref['checkpoint_sha256'], config_sha, signature)
            # In addition to sweep membership, each file must contain precisely its own mask.
            if any(r['mask_threshold'] != mask for r in jsonl(path)):
                raise ValueError('Mixed masks in cached file: '+str(path))
            predictions[fold][str(mask)] = str(path)
    result = dict(reference_run=str(root), splits={k:str(v) for k,v in splits.items()},
                  protocol=str(protocol), model_config=str(root/INFER_CONFIG), predictions=predictions,
                  comparison=str(root/COMPARISON), images_rehashed=checked,
                  verified_Aprime_artifact_sha256=verified_artifacts, Aprime_sweep_identities=sweep_identities,
                  policy='Read-only frozen folds and A′ predictions; no model/data regeneration; E0 uses exact same inference configuration.')
    save_json(output, result)
    return result
