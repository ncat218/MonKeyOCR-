"""Native-mask word diagnostics: descriptive geometry, paired GT transitions, bounded review queue."""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union

from .common import jsonl, read_json, save_json


def size_bin(height):
    return next(label for lo,hi,label in [(0,4,'0_4'),(4,8,'4_8'),(8,12,'8_12'),
        (12,16,'12_16'),(16,32,'16_32'),(32,float('inf'),'ge32')] if lo <= height < hi)


def kept_shapes(gt, roi, prediction, protocol):
    """Mirror vendor ROI/ignore eligibility exactly; matching stays in the vendor evaluator."""
    roi = shape(roi)
    ignore = unary_union([shape(g['geometry']) for g in gt['ignored']])
    kept = []
    for index, coords in enumerate(prediction['polygons']):
        array = np.asarray(coords, dtype=float)
        if array.size < 6 or array.size % 2 or not np.isfinite(array).all():
            continue
        polygon = Polygon(array.reshape(-1,2))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            continue
        clipped = polygon.intersection(roi)
        if clipped.is_empty or clipped.area/polygon.area < protocol['min_prediction_roi_fraction']:
            continue
        if not ignore.is_empty and clipped.intersection(ignore).area/clipped.area > protocol['ignore_prediction_overlap']:
            continue
        kept.append((index, clipped))
    return kept


def word_geometry(gt, roi, prediction, evaluated, protocol):
    """Exclusive FN categories + overlapping merge/fragment-compatible signatures, no causal labels."""
    kept = kept_shapes(gt, roi, prediction, protocol)
    valid = gt['valid']
    shapes = [shape(g['geometry']) for g in valid]
    intersections = np.zeros((len(valid), len(kept)))
    for i,g in enumerate(shapes):
        for j,(_,p) in enumerate(kept):
            if g.intersects(p):
                intersections[i,j] = g.intersection(p).area
    half_counts = (intersections >= np.asarray([g.area for g in shapes])[:,None]*.5).sum(axis=0)
    matched = set(evaluated['matched_ann_ids'])
    rows = []
    for i,(ann,g) in enumerate(zip(valid,shapes)):
        overlaps = [intersections[i,j]/(g.area+p.area-intersections[i,j]) for j,(_,p) in enumerate(kept)]
        best = int(np.argmax(overlaps)) if overlaps else None
        best_iou = overlaps[best] if best is not None else 0.
        is_tp = ann['ann_id'] in matched
        category = ('TP' if is_tp else 'FN_no_kept_prediction_IoU_ge0p1' if best_iou < .1 else
                    'FN_overlap_below_match_IoU' if best_iou < protocol['match_iou'] else
                    'FN_assignment_conflict_candidate')
        coverage = intersections[i]/g.area
        fragments = [p for j,(_,p) in enumerate(kept) if 0 < coverage[j] < .5]
        rows.append(dict(ann_id=ann['ann_id'], original_height=ann['original_height'],
            size_bin=size_bin(ann['original_height']), matched=is_tp, category=category,
            best_kept_IoU=round(best_iou,6),
            best_overlap_prediction_score=(prediction['scores'][kept[best][0]] if best is not None and best_iou > 0 else None),
            merge_compatible_FN=bool(not is_tp and any(coverage[j] >= .5 and half_counts[j] >= 2 for j in range(len(kept)))),
            fragmentation_compatible_FN=bool(not is_tp and len(fragments) >= 2 and
                unary_union(fragments).intersection(g).area/g.area >= .5),
            half_covered_but_unmatched_FN=bool(not is_tp and any(coverage >= .5))))
    fp = dict(FP_partial_overlap=0, FP_no_valid_GT_IoU_ge0p1=0,
              FP_invalid_geometry=evaluated['invalid_prediction_count'])
    by_original = {index:j for j,(index,_) in enumerate(kept)}
    score_groups = dict(TP=[], FP=[])
    false = set(evaluated['fp_prediction_indices'])
    for original,j in by_original.items():
        score_groups['FP' if original in false else 'TP'].append(float(prediction['scores'][original]))
        if original not in false:
            continue
        p = kept[j][1]
        maximum = max((intersections[i,j]/(g.area+p.area-intersections[i,j]) for i,g in enumerate(shapes)), default=0.)
        fp['FP_partial_overlap' if maximum >= .1 else 'FP_no_valid_GT_IoU_ge0p1'] += 1
    if sum(fp.values()) != evaluated['counts'][2]-evaluated['counts'][0]:
        raise ValueError('Geometry FP partition does not reproduce vendor counts')
    if sum(not r['matched'] for r in rows) != evaluated['counts'][1]-evaluated['counts'][0]:
        raise ValueError('Geometry FN partition does not reproduce vendor counts')
    return rows, fp, score_groups


def transition(clean, degraded):
    return 'both_TP' if clean and degraded else 'clean_TP_degraded_FN' if clean else 'clean_FN_degraded_TP' if degraded else 'both_FN'


def select_review(candidates, maximum, seed):
    """Round-robin deterministic strata; purposive review, never an error prevalence estimator."""
    groups = defaultdict(list)
    for row in candidates:
        bucket = row['review_stratum']
        digest = hashlib.sha256(f'{seed}:{row["source_image_id"]}:{row["ann_id"]}'.encode()).hexdigest()
        groups[bucket].append((digest,row))
    for bucket in groups:
        groups[bucket].sort(key=lambda pair: pair[0])
    result, sources = [], Counter()
    while len(result) < maximum:
        added = False
        for bucket in sorted(groups):
            while groups[bucket]:
                _,row = groups[bucket].pop(0)
                if sources[row['source_image_id']] >= 2:
                    continue
                result.append(row)
                sources[row['source_image_id']] += 1
                added = True
                break
            if len(result) == maximum:
                break
        if not added:
            break
    return result


def analyze_native(manifest, protocol_path, predictions, evaluations, output, config):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    protocol = read_json(protocol_path)
    predicted = {m:{r['sample_id']:r for r in jsonl(p)} for m,p in predictions.items()}
    evaluated = {m:{r['sample_id']:r for r in jsonl(Path(p)/'per_image.jsonl')} for m,p in evaluations.items()}
    geometry, paired, adaptation, candidates = {}, {}, {}, []
    word_path = output/'WORDS__native_mask__E0_vs_Aprime__same_GT_ID_clean_degraded.csv'
    writer = None
    with word_path.open('w',encoding='utf-8',newline='') as sink:
        for number,row in enumerate(jsonl(manifest),1):
            source = row['source_image_id']
            clean_ids = {g['ann_id'] for g in row['gt_clean']['valid']}
            degraded_ids = {g['ann_id'] for g in row['gt_degraded']['valid']}
            if clean_ids != degraded_ids:
                raise ValueError('Paired analysis requires same eligible GT IDs in both conditions')
            details = {}
            for model in ('E0','Aprime'):
                details[model] = {}
                for condition,key in [('clean','clean::'+source),('degraded','deg::'+row['variant_id'])]:
                    words,fp,scores = word_geometry(row['gt_'+condition],row['roi_'+condition],
                        predicted[model][key],evaluated[model][key],protocol)
                    details[model][condition] = {w['ann_id']:w for w in words}
                    aggregate = geometry.setdefault(model+'__'+condition,dict(FN_categories=Counter(),
                        FP_categories=Counter(), overlapping_signatures=Counter(), score_histograms={k:[0]*10 for k in ('TP','FP')}))
                    aggregate['FN_categories'].update(w['category'] for w in words if not w['matched'])
                    aggregate['FP_categories'].update(fp)
                    for name in ('merge_compatible_FN','fragmentation_compatible_FN','half_covered_but_unmatched_FN'):
                        aggregate['overlapping_signatures'][name] += sum(w[name] for w in words)
                    for group, values in scores.items():
                        for value in values:
                            aggregate['score_histograms'][group][min(9,int(value*10))] += 1
            for ann in row['gt_clean']['valid']:
                key = ann['ann_id']
                word = dict(source_image_id=source,ann_id=key,original_height=ann['original_height'],
                            size_bin=size_bin(ann['original_height']), operations=';'.join(row['operations']))
                for model in ('E0','Aprime'):
                    c,d = [details[model][condition][key] for condition in ('clean','degraded')]
                    state = transition(c['matched'],d['matched'])
                    paired.setdefault(model,{}).setdefault(word['size_bin'],Counter())[state] += 1
                    for condition,w in [('clean',c),('degraded',d)]:
                        for field in ('matched','category','best_kept_IoU','best_overlap_prediction_score',
                                      'merge_compatible_FN','fragmentation_compatible_FN','half_covered_but_unmatched_FN'):
                            word[model+'__'+condition+'__'+field] = w[field]
                    word[model+'__clean_degraded_transition'] = state
                for condition in ('clean','degraded'):
                    e,a = [details[m][condition][key]['matched'] for m in ('E0','Aprime')]
                    state = 'both_TP' if e and a else 'lost_by_finetuning' if e else 'repaired_by_finetuning' if a else 'both_FN'
                    word[condition+'__E0_to_Aprime'] = state
                    adaptation.setdefault(condition,{}).setdefault(word['size_bin'],Counter())[state] += 1
                if writer is None:
                    writer = csv.DictWriter(sink,fieldnames=list(word));writer.writeheader()
                writer.writerow(word)
                e0c,e0d = [details['E0'][c][key] for c in ('clean','degraded')]
                if word['degraded__E0_to_Aprime'] in ('lost_by_finetuning','repaired_by_finetuning'):
                    stratum = word['degraded__E0_to_Aprime']
                elif e0c['matched'] and not e0d['matched']:
                    stratum = 'E0_clean_TP_lost_under_degradation'
                elif e0c['merge_compatible_FN'] or e0d['merge_compatible_FN']:
                    stratum = 'E0_merge_compatible_geometry'
                elif e0c['fragmentation_compatible_FN'] or e0d['fragmentation_compatible_FN']:
                    stratum = 'E0_fragment_compatible_geometry'
                elif not e0c['matched'] and not e0d['matched']:
                    stratum = 'E0_both_FN'
                else:
                    continue
                candidates.append(dict(source_image_id=source, ann_id=key,
                    review_stratum=stratum+'__'+('small' if ann['original_height'] < 16 else 'ge16'),
                    original_height=ann['original_height'], operations=row['operations']))
            if number % 100 == 0:
                print('NATIVE E0/A′ WORD GEOMETRY AND TRANSITIONS',number,flush=True)
    review = select_review(candidates,config['eda']['review_words_max'],config['seed'])
    save_json(output/'REVIEW_QUEUE__stratified_word_crops__manual_labels_pending.json', dict(
        selection='Deterministic purposive strata; at most two words/source. Not representative prevalence.',words=review))
    summary = dict(geometry=geometry, paired_clean_degraded_by_size=paired,
        E0_to_Aprime_by_size=adaptation, review_words=len(review),
        scope='Native mask 0.30, unchanged vendor matching. FN categories are exclusive; merge/fragment signatures overlap. '
              'No-kept-prediction does not establish feature collapse. Mixed-operation strata are associations, not causal effects. '
              'Score histograms cover valid kept polygons, fixed bins [0,.1,...,1]; invalid FP excluded from score histogram only.')
    save_json(output/'DIAGNOSIS__native_E0_Aprime__word_size_geometry_and_paired_transitions.json',summary)
    return summary
