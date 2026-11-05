"""Report the original detector separately from domain fine-tuning, native and selected points."""
import csv
from pathlib import Path

from .asf_reporting import summarize_words
from .common import jsonl, read_json, save_json
from .reporting import method_difference_ci


def make_report(experiments, selections, manifest, output, config, diagnosis, icdar):
    output = Path(output)
    rows, reports = [], {}
    raw = list(jsonl(manifest))
    for label,folder in experiments.items():
        metric = read_json(Path(folder)/'metrics.json')
        words = summarize_words(raw,list(jsonl(Path(folder)/'per_image.jsonl')))
        reports[label] = dict(metrics=metric,word_diagnostics=words)
        c,d = metric['clean_unique'],metric['degraded_instance']
        row = dict(experiment=label,mask=selections[label.split('__')[0]]['native' if label.endswith('__native') else 'selected'],
            sources=metric['unique_sources'])
        for condition,m in [('clean',c),('degraded',d)]:
            row.update({condition+'_'+key+'_pct':100*m[key] for key in ('precision','recall','hmean')})
            row[condition+'_small_recall_pct'] = (100*metric['size_bins'][condition]['small_lt16px']['recall']
                if metric['size_bins'][condition]['small_lt16px']['recall'] is not None else None)
            row[condition+'_FP'] = int(m['counts']['pred']-m['counts']['tp'])
            row[condition+'_FN'] = int(m['counts']['gt']-m['counts']['tp'])
        rows.append(row)
    contrasts = {point:method_difference_ci(Path(experiments['E0__'+point])/'per_image.jsonl',
        Path(experiments['Aprime__'+point])/'per_image.jsonl',config['evaluation']['bootstrap_repetitions'],config['seed'])
        for point in ('native','selected')}
    result = dict(experiments=reports, mask_selection=selections, Aprime_minus_E0_paired_source_CI=contrasts,
        native_error_diagnosis=diagnosis,ICDAR2015_historical_anchor=icdar,
        limitations=['E0 is the original Monkey detection checkpoint evaluated in FP32; A-prime was mixed fine-tuned in job9793.',
            'The TextVQA source folds are held out from our fine-tuning, not guaranteed unseen by Monkey pretraining (TextOCR train listed).',
            'Synthetic FULL_R1 mixed corruptions; overlapping operation groups are not causal single-corruption effects.',
            'One fine-tuning seed. Source-bootstrap uncertainty does not measure training-seed variance.',
            'Test results have already been inspected; this is exploratory EDA. No test-driven threshold/architecture selection.',
            'The review images are purposive examples and await human labels; geometry signatures are not causal feature diagnoses.',
            'ICDAR uses its historical official evaluator and historical score sweep; TextVQA uses frozen external protocol. Hmeans are not directly comparable.'])
    save_json(output/'COMPARISON_AND_EDA__original_Monkey_vs_mixed_finetuning__native_selected.json',result)
    csv_path = output/'TABLE__E0_original_vs_Aprime_FP32__clean_degraded_Hmean_small_recall_FP_FN.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# FULL_R1: original Monkey checkpoint versus FP32 mixed fine-tuning','',
        'E0 = original checkpoint, no target fine-tuning. A′ = job9793 mixed fine-tuning, 10,000 updates, final weights. '
        'Both use the exact same FP32 inference configuration and external word-polygon protocol.','',
        '| Model/operating point | Mask | Clean P | Clean R | Clean H | Degraded P | Degraded R | Degraded H | Small R clean | Small R degraded |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        keys=['clean_precision_pct','clean_recall_pct','clean_hmean_pct','degraded_precision_pct','degraded_recall_pct',
              'degraded_hmean_pct','clean_small_recall_pct','degraded_small_recall_pct']
        lines.append('| '+r['experiment']+f' | {r["mask"]:.2f} | '+' | '.join('N/A' if r[k] is None else f'{r[k]:.2f}' for k in keys)+' |')
    lines += ['','## Interpretation and next decision','',
        'Native mask 0.30 isolates checkpoint differences at a shared operating point. Selected masks are independently '
        'chosen from the same five candidates by mean validation clean/degraded Hmean, native-first ties. '
        'Small recall may decrease when Hmean increases; both operating points are reported.','',
        'Read DIAGNOSIS for mutually exclusive FN geometry categories, overlapping merge/fragment-compatible signatures, '
        'E0→A′ repaired/lost GT IDs by size, and clean→degraded survival. The complete word CSV stays on HPC. '
        'Corruption-contains tables in COMPARISON overlap and are associations.','']
    for point,conditions in contrasts.items():
        for condition,v in conditions.items():
            lines.append(f'- A′ − E0 ({point}, {condition}): {v["hmean_delta_vs_E0_pp"]:+.3f} pp; '
                         f'95% paired-source CI {v["paired_source_bootstrap_95ci_pp"]}.')
    lines += ['','## Historical ICDAR2015 anchor (not rerun here)','',
        f'500 images; reproduced Hmean {100*icdar["original_report"]["hmean"]:.3f}%. '
        'The separately verified historical EDA is bundled as provenance. This is not a pooled score or an E0 FP32 ICDAR rerun.','',
        '## Evidence still needed before committing to a mechanism','',
        'Review the selected paired crops and record human labels. A no-overlap FN does not prove encoder feature loss; '
        'a merge-compatible polygon does not prove a probability-map bridge. Use this EDA to choose a small validation-only '
        'map/target audit if needed. No new improvement is selected automatically.','',
        '## Limits','']+['- '+s for s in result['limitations']]
    report=output/'REPORT__Monkey_original_baseline_errors_and_what_FP32_finetuning_repairs.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(report=str(report),csv=str(csv_path))
