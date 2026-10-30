"""One external TextOCR polygon evaluator for BOTH detectors; no score sweep/test tuning."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def maximum_matching(adjacency, num_predictions):
    """Maximum cardinality bipartite matching, not order-dependent greedy matching."""
    prediction_to_gt = [-1] * num_predictions
    gt_to_prediction = [-1] * len(adjacency)
    for start in range(len(adjacency)):
        queue, seen_gt, previous_prediction = deque([start]), {start}, {}
        free = -1
        while queue and free < 0:
            gt_index = queue.popleft()
            for pred_index in adjacency[gt_index]:
                if pred_index in previous_prediction:
                    continue
                previous_prediction[pred_index] = gt_index
                owner = prediction_to_gt[pred_index]
                if owner < 0:
                    free = pred_index
                    break
                if owner not in seen_gt:
                    seen_gt.add(owner); queue.append(owner)
        while free >= 0:
            gt_index = previous_prediction[free]
            old_prediction = gt_to_prediction[gt_index]
            prediction_to_gt[free] = gt_index
            gt_to_prediction[gt_index] = free
            free = old_prediction
    return [(g, p) for p, g in enumerate(prediction_to_gt) if g >= 0]


def rates(counts):
    tp, gt, pred, covered = [float(x) for x in counts]
    precision = tp / pred if pred else 0.
    recall = tp / gt if gt else 0.
    return dict(precision=precision, recall=recall,
                hmean=2*precision*recall/(precision+recall) if precision+recall else 0.,
                word_coverage_recall=covered/gt if gt else 0.)


def evaluate_image(gt_entries, roi_mapping, prediction, protocol):
    valid_gt = gt_entries['valid']
    ignored_gt = gt_entries['ignored']
    gt_shapes = [shape(x['geometry']) for x in valid_gt]
    roi = shape(roi_mapping)
    ignore_union = unary_union([shape(x['geometry']) for x in ignored_gt])
    polygons, scores = prediction['polygons'], prediction['scores']
    if len(polygons) != len(scores):
        raise ValueError('Mismatched polygon/score counts')
    kept, invalid_count, ignored_roi, ignored_annotations = [], 0, 0, 0
    for original_index, (coords, score) in enumerate(zip(polygons, scores)):
        if not math.isfinite(float(score)) or not 0 <= score <= 1:
            raise ValueError('Non-finite/out-of-range model confidence')
        array = np.asarray(coords, dtype=float)
        if array.size < 6 or array.size % 2 or not np.isfinite(array).all():
            invalid_count += 1
            continue  # Invalid output is counted as an FP below, not silently discarded.
        geom = Polygon(array.reshape(-1, 2))
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or geom.area <= 0:
            invalid_count += 1
            continue
        clipped = geom.intersection(roi)
        if clipped.is_empty or clipped.area / geom.area < protocol['min_prediction_roi_fraction']:
            ignored_roi += 1
            continue
        if not ignore_union.is_empty and clipped.intersection(ignore_union).area / clipped.area > protocol['ignore_prediction_overlap']:
            ignored_annotations += 1
            continue
        kept.append((original_index, clipped))
    adjacency = []
    for g in gt_shapes:
        overlaps = []
        for j, (_, p) in enumerate(kept):
            inter = g.intersection(p).area
            iou = inter / (g.area + p.area - inter) if inter else 0.
            if iou >= protocol['match_iou']:
                overlaps.append((j, iou))
        adjacency.append([j for j, _ in sorted(overlaps, key=lambda x: (-x[1], x[0]))])
    pairs = maximum_matching(adjacency, len(kept))
    matched_gt = {g for g, _ in pairs}
    matched_predictions = {p for _, p in pairs}
    union = unary_union([g for _, g in kept])
    covered = {i for i, g in enumerate(gt_shapes)
               if not union.is_empty and union.intersection(g).area / g.area >= protocol['word_coverage']}
    counts = [len(pairs), len(gt_shapes), len(kept) + invalid_count, len(covered)]
    size_bins = {}
    for label, lo, hi in [('small_lt16px', 0, 16), ('medium_16_32px', 16, 32), ('large_ge32px', 32, float('inf'))]:
        indices = {i for i, g in enumerate(valid_gt) if lo <= g['original_height'] < hi}
        size_bins[label] = dict(gt=len(indices), matched=len(indices & matched_gt), covered=len(indices & covered))
    return dict(counts=counts, **rates(counts),
                missed_ann_ids=[valid_gt[i]['ann_id'] for i in range(len(gt_shapes)) if i not in matched_gt],
                matched_ann_ids=[valid_gt[i]['ann_id'] for i in sorted(matched_gt)],
                fp_prediction_indices=[kept[i][0] for i in range(len(kept)) if i not in matched_predictions],
                invalid_prediction_count=invalid_count, ignored_outside_roi=ignored_roi,
                ignored_on_annotations=ignored_annotations,
                predicted_roi_area_fraction=float(union.area / roi.area) if not union.is_empty else 0.,
                size_bins=size_bins)


def bootstrap_paired(source_vectors, repetitions=1000, seed=42):
    matrix = np.asarray(source_vectors, dtype=float)
    if len(matrix) < 2 or repetitions <= 0:
        return dict(method='cluster bootstrap by source_image_id', available=False)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(repetitions):
        aggregate = matrix[rng.integers(0, len(matrix), len(matrix))].sum(axis=0)
        clean, degraded = rates(aggregate[:4]), rates(aggregate[4:])
        for key in clean:
            samples[key].append(100*(degraded[key]-clean[key]))
    return dict(method='cluster bootstrap by source_image_id', available=True,
                repetitions=repetitions, seed=seed,
                delta_pp_95ci={key: [float(x) for x in np.quantile(values, [.025, .975])]
                               for key, values in samples.items()})


def evaluate(manifest, predictions, protocol_path, output_dir, bootstrap=1000):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, preds = read_jsonl(manifest), read_jsonl(predictions)
    protocol = json.loads(Path(protocol_path).read_text())
    manifest_hash = sha256(manifest)
    pred_map = {p['sample_id']: p for p in preds}
    if len(pred_map) != len(preds):
        raise ValueError('Duplicate prediction sample_id')
    clean_by_id = {}
    for row in rows:
        source = row['source_image_id']
        if source in clean_by_id and row['gt_clean'] != clean_by_id[source]['gt_clean']:
            raise ValueError('Clean GT identity differs across variants; common-source GT required')
        clean_by_id[source] = row
    expected = {f'clean::{s}' for s in clean_by_id} | {f'deg::{r["variant_id"]}' for r in rows}
    if expected != set(pred_map):
        raise ValueError(f'Incomplete prediction set: missing={len(expected-set(pred_map))}, extra={len(set(pred_map)-expected)}')
    if any(p.get('manifest_sha256') != manifest_hash for p in preds):
        raise ValueError('Predictions belong to a different frozen manifest')
    models = {p['model'] for p in preds}
    if len(models) != 1:
        raise ValueError('Do not mix model predictions')
    clean_results, degraded_results = {}, []
    size_totals = {c: defaultdict(lambda: np.zeros(3)) for c in ('clean', 'degraded')}
    for source, row in sorted(clean_by_id.items()):
        key = f'clean::{source}'
        result = evaluate_image(row['gt_clean'], row['roi_clean'], pred_map[key], protocol)
        clean_results[source] = dict(sample_id=key, source_image_id=source, condition='clean', **result)
        for bucket, vals in result['size_bins'].items():
            size_totals['clean'][bucket] += [vals['gt'], vals['matched'], vals['covered']]
    for i, row in enumerate(rows):
        key = f'deg::{row["variant_id"]}'
        result = evaluate_image(row['gt_degraded'], row['roi_degraded'], pred_map[key], protocol)
        degraded_results.append(dict(sample_id=key, source_image_id=row['source_image_id'],
                                     variant_id=row['variant_id'], condition='degraded',
                                     family=row['condition'], severity=row['severity'],
                                     has_perspective=row['has_perspective'], **result))
        for bucket, vals in result['size_bins'].items():
            size_totals['degraded'][bucket] += [vals['gt'], vals['matched'], vals['covered']]
        if (i+1) % 250 == 0:
            print('EVALUATED:', i+1, '/', len(rows), flush=True)
    def summarize(items):
        counts = np.asarray([r['counts'] for r in items], dtype=float).sum(axis=0)
        return dict(instances=len(items), counts=dict(zip(('tp','gt','pred','covered'), counts.tolist())), **rates(counts))
    clean_summary = summarize(list(clean_results.values()))
    degraded_summary = summarize(degraded_results)
    by_source = defaultdict(list)
    for r in degraded_results:
        by_source[r['source_image_id']].append(r)
    vectors, pairs = [], []
    for source, variants in sorted(by_source.items()):
        c = clean_results[source]
        d_counts = np.mean([r['counts'] for r in variants], axis=0)
        vectors.append(list(c['counts']) + d_counts.tolist())
        d = rates(d_counts)
        pair = dict(source_image_id=source, variants=len(variants),
                    clean_counts=c['counts'], degraded_mean_counts=d_counts.tolist())
        for metric in ('precision', 'recall', 'hmean', 'word_coverage_recall'):
            pair[metric + '_clean'] = c[metric]
            pair[metric + '_degraded'] = d[metric]
            pair[metric + '_delta_pp'] = 100*(d[metric]-c[metric])
        pairs.append(pair)
    matrix = np.asarray(vectors)
    paired_clean, paired_degraded = rates(matrix[:, :4].sum(axis=0)), rates(matrix[:, 4:].sum(axis=0))
    stratified = {}
    for group in sorted({(r['family'], r['severity'], r['has_perspective']) for r in degraded_results}):
        subset = [r for r in degraded_results if (r['family'], r['severity'], r['has_perspective']) == group]
        paired_baseline = [clean_results[r['source_image_id']] for r in subset]
        c, d = summarize(paired_baseline), summarize(subset)
        stratified[str(group)] = dict(clean_paired=c, degraded=d,
                                      delta_pp={k:100*(d[k]-c[k]) for k in paired_clean})
    latency = {}
    for condition in ('clean', 'degraded'):
        values = [float(p['inference_ms_with_decode_and_postprocess']) for p in preds
                  if p.get('condition') == condition and 'inference_ms_with_decode_and_postprocess' in p]
        latency[condition] = (dict(available=True, samples=len(values), mean_ms=float(np.mean(values)),
                                   p50_ms=float(np.quantile(values, .5)), p95_ms=float(np.quantile(values, .95)))
                              if values else dict(available=False))
    result = dict(evaluation_type='external_TextVQA_TextOCR_word_level_polygon_evaluation',
                  model=next(iter(models)), manifest_sha256=manifest_hash,
                  evaluator_sha256=sha256(__file__),
                  protocol_sha256=sha256(protocol_path), unique_sources=len(clean_by_id),
                  degraded_variants=len(rows), score_policy=protocol['score_policy'],
                  latency_with_decode_and_postprocess=latency,
                  clean_unique=clean_summary, degraded_instance=degraded_summary,
                  paired_source_weighted_micro=dict(clean=paired_clean, degraded=paired_degraded,
                      delta_pp={k:100*(paired_degraded[k]-paired_clean[k]) for k in paired_clean}),
                  macro_source_mean={k:dict(clean=float(np.mean([p[k+'_clean'] for p in pairs])),
                                           degraded=float(np.mean([p[k+'_degraded'] for p in pairs])))
                                     for k in paired_clean},
                  bootstrap=bootstrap_paired(vectors, bootstrap), strata=stratified,
                  size_bins={cond:{key:dict(gt=int(v[0]),matched=int(v[1]),covered=int(v[2]),
                                           recall=float(v[1]/v[0]) if v[0] else None)
                                   for key,v in buckets.items()} for cond,buckets in size_totals.items()},
                  limitations=['Synthetic degradation, not captured degraded images',
                               'TextOCR train is included in MonkeyOCRv2 pretraining sources; no unseen-data claim',
                               'Word GT versus possibly line predictions; coverage is diagnostic and can be gamed by large polygons',
                               'Full-set baseline audit; do not later fine-tune and claim this entire set is an untouched test'])
    (output/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (output/'per_image.jsonl').open('w',encoding='utf-8') as f:
        for row in list(clean_results.values())+degraded_results:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with (output/'paired_source_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    worst=sorted(pairs,key=lambda r:r['recall_delta_pp'])[:20]
    (output/'worst_recall_sources.json').write_text(json.dumps(worst,indent=2)+'\n')
    table=['# Clean versus degraded — checkpoint baseline','',
           'External TextVQA images + TextOCR word GT; not paper-official Hmean. No fine-tuning.', '',
           '| Metric | Clean paired (%) | Degraded paired (%) | Delta (pp) | 95% cluster-bootstrap CI (pp) |',
           '|---|---:|---:|---:|---|']
    for key in paired_clean:
        interval=result['bootstrap'].get('delta_pp_95ci',{}).get(key)
        table.append(f'| {key} | {100*paired_clean[key]:.3f} | {100*paired_degraded[key]:.3f} | '
                     f'{100*(paired_degraded[key]-paired_clean[key]):+.3f} | {interval} |')
    table += ['', f'Sources: {len(clean_by_id)}; degraded variants: {len(rows)}.',
              f'Manifest SHA256: `{manifest_hash}`.', '',
              'Delta = degraded - clean. CI excluding zero indicates a directional difference under this protocol;',
              'effect size in percentage points determines practical magnitude. No preset conclusion of a large drop.',
              'Review word-vs-line granularity, giant boxes, FP/FN and geometric masks alongside these scores.']
    (output/'clean_vs_degraded.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--predictions',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args()
    results=evaluate(a.manifest,a.predictions,a.protocol,a.output_dir,a.bootstrap)
    print(json.dumps(results['paired_source_weighted_micro'],indent=2))
