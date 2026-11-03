"""One global network forward per image, several native mask-threshold decodes.

No T-map inference, unclip/loss changes or disk-sized probability-map archive.
Partial prediction files are independently resumable per threshold.
"""
import argparse
import copy
import math
import signal
import time
from contextlib import ExitStack
from pathlib import Path

from .common import canonical, identity, locked_json, read_json, resume_jsonl, save_json, sha256, stop_signals


def threshold_tag(value):
    return 'mask_'+format(value, '.2f').replace('.', 'p')


def prediction_file(output, value):
    return Path(output)/threshold_tag(value)/'predictions__global__clean_and_degraded.jsonl'


def choose_mask(candidates):
    """Predeclared native-first stable tie break; same threshold for clean/degraded."""
    if not candidates:
        raise ValueError('No validation candidates')
    for c in candidates:
        if not math.isfinite(c['balanced_validation_hmean']):
            raise ValueError('Non-finite validation selection metric')
    return max(candidates, key=lambda c: c['balanced_validation_hmean'])


def run(a):
    from .inference import tasks_for
    import torch
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.apis import TextDetInferencer
    from mmocr.utils import register_all_modules
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    masks = read_json(a.thresholds)['mask_thresholds']
    if not masks or len(set(masks)) != len(masks) or any(not 0 < t < 1 for t in masks):
        raise ValueError('Invalid mask thresholds')
    tasks = list(tasks_for(a.manifest, read_json(a.locations)))
    expected = {t['sample_id'] for t in tasks}
    if len(expected) != len(tasks):
        raise ValueError('Duplicate inference tasks')
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(a.model_config))
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    register_all_modules(init_default_scope=True)
    record_attention = bool(getattr(a, 'record_attention', False))
    attention_path = a.output/'DIAGNOSTICS__real_ASF_attention_and_feature_RMS__per_image.jsonl'
    fingerprint = dict(manifest_sha256=sha256(a.manifest), checkpoint_sha256=sha256(a.checkpoint),
        inference_config_sha256=sha256(a.model_config), worker_sha256=sha256(__file__),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(0), masks=masks,
        label=a.model_label, mode='global_shared_P_map__native_DBPostprocessor',
        real_attention_diagnostics=record_attention)
    if cfg.get('textdet_parameter_precision') == 'fp32':
        from .precision_control import runtime_math_flags
        fingerprint['runtime_math_flags'] = runtime_math_flags()
    locked_json(a.output/'immutable_map_sweep_identity.json', fingerprint)
    signature = identity(fingerprint)
    seen, files = {}, {}
    for threshold in masks:
        path = prediction_file(a.output, threshold)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = resume_jsonl(path)
        ids = {r['sample_id'] for r in existing}
        if len(ids) != len(existing) or not ids <= expected:
            raise ValueError('Invalid resumed threshold prediction coverage')
        if any(r['inference_identity'] != signature or r['mask_threshold'] != threshold for r in existing):
            raise ValueError('Mixed resumed prediction identities')
        seen[threshold], files[threshold] = ids, path
    attention_seen = set()
    if record_attention:
        previous = resume_jsonl(attention_path)
        attention_seen = {r['sample_id'] for r in previous}
        if len(attention_seen) != len(previous) or not attention_seen <= expected or any(r['inference_identity'] != signature for r in previous):
            raise ValueError('Mixed/incomplete identity in resumed ASF attention diagnostics')
    if all(s == expected for s in seen.values()) and (not record_attention or attention_seen == expected):
        save_json(a.output/'COMPLETE__all_requested_mask_predictions.json',
                  dict(samples_per_threshold=len(expected), masks=masks, resumed_all=True))
        return
    detector = TextDetInferencer(model=str(a.model_config), weights=str(a.checkpoint), device='cuda:0')
    detector.model.eval()
    if cfg.get('textdet_parameter_precision') == 'fp32':
        from .precision_control import enforce_fp32
        # Reload below after casting, so an inferencer's initial loading cannot
        # silently quantize FP32 final weights into a BF16 construction.
        enforce_fp32(detector.model)
    load_checkpoint(detector.model, str(a.checkpoint), map_location='cpu', strict=True)
    attention_latest, attention_handles = {}, []
    if record_attention:
        from .fp32_checks import attach_attention_recorder
        attention_latest, attention_handles = attach_attention_recorder(detector.model)
    pp = detector.model.det_head.postprocessor
    if (pp.mask_thr, pp.min_text_score, pp.min_text_width, pp.unclip_ratio, pp.text_repr_type) != (.3, .3, 5, 1.5, 'quad'):
        raise ValueError('Native DB postprocessor differs from the pinned experiment defaults')
    stop = [False]
    for sig in stop_signals():
        signal.signal(sig, lambda signum, frame: stop.__setitem__(0, True))

    def unpack(sample):
        pred = sample.pred_instances
        polys = [[float(v) for v in p.reshape(-1)] for p in pred.polygons]
        scores = [float(v) for v in pred.scores.detach().cpu().tolist()]
        if len(polys) != len(scores) or any(not math.isfinite(v) for p in polys for v in p):
            raise ValueError('Invalid polygon result')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
            raise ValueError('Invalid polygon confidence')
        return dict(polygons=polys, scores=scores)

    checked_equivalence = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with ExitStack() as stack:
        sinks = {t: stack.enter_context(p.open('ab')) for t, p in files.items()}
        attention_sink = stack.enter_context(attention_path.open('ab')) if record_attention else None
        for index, task in enumerate(tasks):
            needed = [t for t in masks if task['sample_id'] not in seen[t]]
            if not needed and (not record_attention or task['sample_id'] in attention_seen):
                continue
            if stop[0]:
                raise InterruptedError('Mask sweep interrupted; same command resumes partial predictions')
            if sha256(task['path']) != task['image_sha256']:
                raise ValueError('Input image bytes changed: '+task['sample_id'])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, data = next(iter(detector.preprocess([task['path']], batch_size=1)))
            with torch.inference_mode():
                processed = detector.model.data_preprocessor(data, training=False)
                probability = detector.model(**processed, mode='tensor')
            attention_stats = copy.deepcopy(attention_latest) if record_attention else None
            torch.cuda.synchronize()
            forward_ms = (time.perf_counter()-t0)*1000
            if not isinstance(probability, torch.Tensor) or probability.ndim != 3 or probability.shape[0] != 1:
                raise ValueError('Expected native DBHead N,H,W probability tensor')
            if probability.shape[-2:] != processed['inputs'].shape[-2:]:
                raise ValueError('Unexpected P-map stride; do not guess map coordinates')
            if not torch.isfinite(probability).all():
                raise ValueError('Non-finite probability map')
            # Once per launch, verify the optimized path against ordinary native
            # predict on the same preprocessed image, including polygon rescaling.
            if not checked_equivalence:
                pp.mask_thr = .3
                with torch.inference_mode():
                    native = detector.model.predict(processed['inputs'], copy.deepcopy(processed['data_samples']))
                decoded = pp(probability, copy.deepcopy(processed['data_samples']))
                if unpack(native[0]) != unpack(decoded[0]):
                    raise ValueError('Shared-map path differs from native DBNet prediction')
                save_json(a.output/'VERIFICATION__shared_P_map_native_predict_equivalence.json',
                          dict(sample_id=task['sample_id'], exact_polygons_and_scores_equal=True,
                               P_shape=list(probability.shape), padded_input_shape=list(processed['inputs'].shape)))
                checked_equivalence = True
            if record_attention and task['sample_id'] not in attention_seen:
                if not attention_stats:
                    raise RuntimeError('ASF hook did not execute on the actual inference path')
                attention_sink.write(canonical(dict(sample_id=task['sample_id'],
                    source_image_id=task['source_image_id'], condition=task['condition'],
                    inference_identity=signature, checkpoint_sha256=fingerprint['checkpoint_sha256'],
                    **attention_stats)))
                attention_sink.flush()
                attention_seen.add(task['sample_id'])
            for threshold in needed:
                pp.mask_thr = threshold
                tick = time.perf_counter()
                decoded = pp(probability, copy.deepcopy(processed['data_samples']))
                torch.cuda.synchronize()
                post_ms = (time.perf_counter()-tick)*1000
                row = {k: v for k, v in task.items() if k not in ('path', 'size')}
                row.update(unpack(decoded[0]), model=a.model_label+'__'+threshold_tag(threshold),
                    manifest_sha256=fingerprint['manifest_sha256'], checkpoint_sha256=fingerprint['checkpoint_sha256'],
                    inference_config_sha256=fingerprint['inference_config_sha256'], inference_identity=signature,
                    mask_threshold=threshold, min_text_score=.3, min_text_width=5, unclip_ratio=1.5,
                    inference_ms_with_decode_and_postprocess=forward_ms+post_ms,
                    timing_note='Shared forward/decode plus this threshold decode; excludes hash/equivalence probe; includes attention hooks when enabled')
                sinks[threshold].write(canonical(row))
                sinks[threshold].flush()
                seen[threshold].add(task['sample_id'])
            if (index+1) % 25 == 0:
                print('GLOBAL P-MAP', a.model_label, index+1, '/', len(tasks),
                      'thresholds', len(masks), 'elapsed_s', round(time.perf_counter()-started), flush=True)
    if not all(s == expected for s in seen.values()):
        raise ValueError('Incomplete threshold predictions; no scoring')
    if record_attention and attention_seen != expected:
        raise ValueError('Incomplete real attention diagnostics')
    for handle in attention_handles:
        handle.remove()
    save_json(a.output/'COMPLETE__all_requested_mask_predictions.json', dict(samples_per_threshold=len(expected),
        masks=masks, peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
        seconds=time.perf_counter()-started, files={threshold_tag(t): str(p) for t,p in files.items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('manifest', 'locations', 'model-config', 'checkpoint', 'thresholds', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--model-label', required=True)
    p.add_argument('--record-attention', action='store_true')
    run(p.parse_args())
