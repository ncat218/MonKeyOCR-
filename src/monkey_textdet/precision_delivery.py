"""Small automatic delivery packet: reports, real diagnostics, no data/weights."""
import json
import zipfile
from pathlib import Path

import numpy as np

from .common import jsonl, read_json, save_json, sha256
from .variants import VARIANTS


def attention_summary(root):
    folder = root/'outputs/global_predictions'/VARIANTS[1]
    name = 'DIAGNOSTICS__real_ASF_attention_and_feature_RMS__per_image.jsonl'
    initial = list(jsonl(folder/'initial_validation_diagnostics'/name))
    source_ids = {r['source_image_id'] for r in initial}
    final = [r for r in jsonl(folder/'validation'/name) if r['source_image_id'] in source_ids]
    if {r['sample_id'] for r in initial} != {r['sample_id'] for r in final}:
        raise ValueError('Initial/final attention diagnostics must compare identical validation images')
    summary = {}
    keys = ('attention_mean','fraction_attention_below_0p05','fraction_attention_above_0p95',
            'feature_rms_before_ASF','feature_rms_after_ASF')
    for stage, records in [('initial',initial),('final',final)]:
        summary[stage] = {}
        for condition in ('clean','degraded'):
            selected = [r for r in records if r['condition'] == condition]
            summary[stage][condition] = dict(images=len(selected), **{
                key:np.asarray([r[key] for r in selected]).mean(axis=0).tolist() for key in keys})
    result = root/'outputs/INITIAL_VALIDATION_DIAGNOSTICS__untrained_A_B__not_test'/(
        'ATTENTION__same_validation_sources_initial_vs_final_FP32_ASF.json')
    save_json(result, dict(summary=summary, initial_per_image=initial, final_per_image=final,
        scope='Same validation images, native global padded FPN grid including background/padding; '
              'not GT-conditioned or a causal corruption study. Saturation alone does not prove missed-word cause.'))
    return result


def package_results(root):
    root = Path(root)
    attention_summary(root)
    marker = root/'COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json'
    output = root/'RESULTS_AND_DIAGNOSTICS__FULL_R1_FP32_DBNet_ASF__final_weights_only.zip'
    identity = dict(completion_marker_sha256=sha256(marker))
    if output.exists():
        with zipfile.ZipFile(output) as z:
            if z.testzip() is not None or json.loads(z.read('DELIVERY_IDENTITY.json')) != identity:
                raise RuntimeError('Existing delivery ZIP differs; do not overwrite evidence')
        return output
    selected = {marker, root/'run_configuration__FULL_R1_two_architectures_same_budget.json',
                root/'READ_ME_FIRST__FULL_R1_DBNet_ASF_run_outputs_and_logs.json'}
    for relative in ('inputs/02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT',
        'inputs/03_frozen_detection_source_splits__official_train_intersection__70_15_15',
        'inputs/04_shared_mixed_training_and_exact_validation_annotations',
        'inputs/05_model_configs__only_FPNC_ASF_differs',
        'outputs/FINAL_COMPARISON__FULL_R1_DBNet_vs_ASF__native_and_validation_selected_masks',
        'outputs/validation_selection', 'outputs/INITIAL_VALIDATION_DIAGNOSTICS__untrained_A_B__not_test'):
        selected.update(p for p in (root/relative).rglob('*')
                        if p.is_file() and p.suffix in ('.json','.md','.csv','.py'))
    for variant in VARIANTS:
        training = root/'outputs/training'/variant
        selected.update(p for p in training.rglob('*') if p.is_file() and
            (p.name == 'scalars.json' or p.suffix == '.log' or
             (p.parent == training and p.suffix == '.json')))
    files = []
    for path in sorted(selected):
        if not path.is_file():
            raise RuntimeError('Required delivery artifact missing: '+str(path))
        path.resolve().relative_to(root.resolve())
        if path.stat().st_size > 8*1024**2:
            raise RuntimeError('Unexpected oversized delivery metadata: '+str(path))
        files.append(path)
    if sum(p.stat().st_size for p in files) > 64*1024**2:
        raise RuntimeError('Delivery metadata exceeds declared 64 MiB uncompressed limit')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('DELIVERY_IDENTITY.json', json.dumps(identity))
        for path in files:
            z.write(path, path.relative_to(root).as_posix())
    save_json(root/'DELIVERY__where_to_find_FP32_reports_and_diagnostics.json',
              dict(zip=str(output), zip_sha256=sha256(output), files=len(files)))
    return output
