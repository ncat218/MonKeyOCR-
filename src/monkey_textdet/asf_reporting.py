"""Auditable ASF effects, threshold effects, word sizes and corruption groups."""
import csv
from pathlib import Path

from .common import jsonl, read_json, save_json
from .reporting import method_difference_ci
from .variants import VARIANTS
from .vendor.evaluate_shared import rates


def summarize_words(manifest_rows, per_image):
    indexed = {(r['source_image_id'], r['condition']): r for r in per_image}
    buckets = {}
    survival = dict(small_words=0, detected_both=0, clean_only=0, degraded_only=0, missed_both=0)
    groups = {}
    for row in manifest_rows:
        source = row['source_image_id']
        c, d = indexed[(source, 'clean')], indexed[(source, 'degraded')]
        cm, dm = set(c['matched_ann_ids']), set(d['matched_ann_ids'])
        for ann in row['gt_clean']['valid']:
            height, key = ann['original_height'], ann['ann_id']
            name = next(label for lo, hi, label in [(0,4,'0_4'), (4,8,'4_8'), (8,12,'8_12'),
                         (12,16,'12_16'), (16,32,'16_32'), (32,float('inf'),'ge32')] if lo <= height < hi)
            b = buckets.setdefault(name, dict(gt=0, clean_matched=0, degraded_matched=0))
            b['gt'] += 1
            b['clean_matched'] += int(key in cm)
            b['degraded_matched'] += int(key in dm)
            if height < 16:
                survival['small_words'] += 1
                tag = ('detected_both' if key in cm and key in dm else 'clean_only' if key in cm
                       else 'degraded_only' if key in dm else 'missed_both')
                survival[tag] += 1
        for op in set(row['operations']):
            g = groups.setdefault(op, dict(sources=0, clean_counts=[0]*4, degraded_counts=[0]*4,
                                          small_gt=0, small_clean_matched=0, small_degraded_matched=0))
            g['sources'] += 1
            for name, metric in [('clean_counts', c), ('degraded_counts', d)]:
                g[name] = [a+b for a,b in zip(g[name], metric['counts'])]
            g['small_gt'] += c['size_bins']['small_lt16px']['gt']
            g['small_clean_matched'] += c['size_bins']['small_lt16px']['matched']
            g['small_degraded_matched'] += d['size_bins']['small_lt16px']['matched']
    for b in buckets.values():
        b.update(clean_recall=b['clean_matched']/b['gt'], degraded_recall=b['degraded_matched']/b['gt'])
    for g in groups.values():
        g.update(clean=rates(g['clean_counts']), degraded=rates(g['degraded_counts']),
                 small_clean_recall=g['small_clean_matched']/g['small_gt'] if g['small_gt'] else None,
                 small_degraded_recall=g['small_degraded_matched']/g['small_gt'] if g['small_gt'] else None)
    denominator = survival['detected_both']+survival['clean_only']
    survival['retention_of_small_clean_TP'] = survival['detected_both']/denominator if denominator else None
    return dict(height_bins_original_pixels=buckets, small_word_transitions=survival,
                corruption_contains_groups_overlapping_not_causal=groups)


def make_report(experiments, output, config, manifest, training, selections):
    output.mkdir(parents=True, exist_ok=True)
    raw_manifest = list(jsonl(manifest))
    reports, rows = {}, []
    for label, folder in experiments.items():
        metrics = read_json(Path(folder)/'metrics.json')
        words = summarize_words(raw_manifest, list(jsonl(Path(folder)/'per_image.jsonl')))
        reports[label] = dict(metrics=metrics, word_diagnostics=words)
        c, d = metrics['clean_unique'], metrics['degraded_instance']
        def small(condition):
            value = metrics['size_bins'][condition]['small_lt16px']['recall']
            return 100*value if value is not None else None
        rows.append(dict(experiment=label, test_sources=metrics['unique_sources'],
            clean_P=100*c['precision'], clean_R=100*c['recall'], clean_H=100*c['hmean'],
            degraded_P=100*d['precision'], degraded_R=100*d['recall'], degraded_H=100*d['hmean'],
            clean_small_R=small('clean'), degraded_small_R=small('degraded'),
            clean_FP=int(c['counts']['pred']-c['counts']['tp']),
            degraded_FP=int(d['counts']['pred']-d['counts']['tp']),
            clean_FN=int(c['counts']['gt']-c['counts']['tp']),
            degraded_FN=int(d['counts']['gt']-d['counts']['tp'])))
    a, b = VARIANTS
    native, selected = '__native_mask_0p30', '__validation_selected_mask'
    contrasts = {'ASF_only_at_native_threshold': (a+native, b+native),
                 'mask_selection_effect_DBNet': (a+native, a+selected),
                 'mask_selection_effect_ASF': (b+native, b+selected),
                 'ASF_with_equal_validation_calibration_opportunity': (a+selected, b+selected)}
    differences = {}
    for name, (control, method) in contrasts.items():
        ci = method_difference_ci(Path(experiments[control])/'per_image.jsonl',
            Path(experiments[method])/'per_image.jsonl', config['evaluation']['bootstrap_repetitions'], config['seed'])
        for value in ci.values():
            value['hmean_delta_method_minus_control_pp'] = value.pop('hmean_delta_vs_E0_pp')
        differences[name] = dict(control=control, method=method, paired_source_bootstrap=ci)
    save_json(output/'COMPARISON__DBNet_ASF_native_and_selected_masks__all_metrics_and_word_errors.json',
        dict(experiments=reports, comparisons=differences, training=training,
             validation_mask_selections=selections, configuration=config,
             warning='No result is guaranteed; calibration and architecture effects are separate. '
                     'Single training seed; official-train detection folds, TextOCR pretraining overlap.'))
    csv_path = output/'TABLE__DBNet_vs_ASF__clean_degraded_Hmean_small_recall_FP_FN.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# FULL_R1: DBNet versus DBNet+ASF with MonkeyOCRv2-AS', '',
        'Two fresh mixed-data trainings from the same original shared weights; only ASF differs. '
        'Native threshold is 0.30. Alternative mask thresholds are selected on validation only, '
        'with one shared threshold for clean and degraded.', '',
        '| Experiment | Clean P | Clean R | Clean H | Degraded P | Degraded R | Degraded H | Small R clean | Small R degraded | Degraded FP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row[k] for k in ('clean_P','clean_R','clean_H','degraded_P','degraded_R','degraded_H','clean_small_R','degraded_small_R')]
        lines.append('| '+row['experiment']+' | '+' | '.join('N/A' if v is None else f'{v:.3f}' for v in values)+f" | {row['degraded_FP']} |")
    lines += ['', 'All rate columns are percentages; differences are percentage points.', '',
        '## How to read the comparison', '',
        '- Compare native B versus native A for the ASF architecture effect. Compare selected versus native within each checkpoint for the mask effect.',
        '- Four evaluation rows represent two trainings. If validation selects 0.30, native and selected rows intentionally share identical predictions.',
        '- Review 4–8/8–12/12–16 px recalls, small clean-TP survival and corruption groups in the JSON. Corruption groups overlap and do not identify isolated causal effects.',
        '- Small recall uses original GT height <16 px. We do not invent small precision: unmatched predictions have no unique GT size. Overall precision and FP counts are reported.',
        '- Gap shrinkage caused by clean deterioration is not sufficient evidence of robustness. Failed threshold sweeps do not prove feature collapse.',
        '- Synthetic FULL_R1 data. Simulated review labels are not human readability/answerability judgments. TextOCR train overlaps Monkey pretraining sources.',
        '- Detection train/validation/test are 70/15/15 source/clean-SHA groups inside official TextVQA train. Official TextVQA val/test and generator calibration/ablation are excluded.',
        '- Both models finish the same optimizer-update budget and use the best mean clean/degraded validation Hmean checkpoint. Coverage of all training entries during the full schedule does not imply that an earlier selected checkpoint has seen all entries.',
        '- One training seed; source-bootstrap intervals do not measure variability across training seeds. Old10k job7957 is historical context, not the control for this full-data experiment.',
        '- Inference timing shares one P-map forward across thresholds; see per-image timing notes and peak allocated VRAM summaries. No promise of a fixed ASF overhead.', '']
    md = output/'RESULTS__FULL_R1_DBNet_ASF__small_words_corruptions_and_postprocessing.md'
    md.write_text('\n'.join(lines), encoding='utf-8')
    return dict(report=str(md), csv=str(csv_path))
