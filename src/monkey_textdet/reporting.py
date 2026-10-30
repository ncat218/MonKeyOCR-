"""Compare frozen test-source predictions without selecting parameters on test outcomes."""
import csv
import io
import json
from pathlib import Path

import numpy as np

from .common import jsonl,read_json,save_json
from .vendor.evaluate_shared import rates


def method_difference_ci(baseline_file,method_file,repetitions,seed):
    baseline={r['sample_id']:r for r in jsonl(baseline_file)}
    method={r['sample_id']:r for r in jsonl(method_file)}
    if baseline.keys()!=method.keys():raise ValueError('Different evaluated test sample IDs across methods')
    result={}
    for condition in ('clean','degraded'):
        keys=sorted(k for k,r in baseline.items() if r['condition']==condition)
        for key in keys:
            if baseline[key]['source_image_id']!=method[key]['source_image_id']:
                raise ValueError('Source mismatch in method comparison')
        matrix=np.asarray([baseline[k]['counts']+method[k]['counts'] for k in keys],dtype=float)
        c=matrix.sum(axis=0)
        delta=100*(rates(c[4:])['hmean']-rates(c[:4])['hmean'])
        rng=np.random.default_rng(seed);samples=[]
        for _ in range(repetitions):
            summed=matrix[rng.integers(0,len(matrix),len(matrix))].sum(axis=0)
            samples.append(100*(rates(summed[4:])['hmean']-rates(summed[:4])['hmean']))
        result[condition]=dict(hmean_delta_vs_E0_pp=delta,
            paired_source_bootstrap_95ci_pp=np.quantile(samples,[.025,.975]).tolist() if samples else None,
            sources=len(keys))
    return result


def make_report(experiments,output,config,split_report,tiling_selection):
    output.mkdir(parents=True,exist_ok=True)
    result=dict(experiments={},split=read_json(split_report),tiling_selection=read_json(tiling_selection),
                training_budget=config['training'],
                limitations=['Synthetic realistic_mix L2 with image_relative_fallback.',
                    'Phase 2 master-set results were previously inspected; Phase 3 test is held out from optimization only.',
                    'TextOCR train was part of Monkey pretraining sources; not an unseen-data claim.',
                    'E1 and E2 independently initialize from the same official checkpoint.',
                    'Whole suite uses one normal H200 MIG allocation; no measured T4/H200 speed ratio is assumed.'])
    rows=[]
    baseline=experiments['E0_original_checkpoint__global']
    for label,folder in experiments.items():
        metrics=read_json(Path(folder)/'metrics.json')
        result['experiments'][label]=metrics
        c,d=metrics['clean_unique'],metrics['degraded_instance']
        row=dict(experiment=label,test_sources=metrics['unique_sources'],
            clean_precision_pct=100*c['precision'],clean_recall_pct=100*c['recall'],clean_hmean_pct=100*c['hmean'],
            degraded_precision_pct=100*d['precision'],degraded_recall_pct=100*d['recall'],degraded_hmean_pct=100*d['hmean'],
            degraded_minus_clean_hmean_pp=100*(d['hmean']-c['hmean']),
            degraded_small_text_recall_pct=(100*metrics['size_bins']['degraded']['small_lt16px']['recall']
                if metrics['size_bins']['degraded']['small_lt16px']['recall'] is not None else None))
        if label!='E0_original_checkpoint__global':
            result['experiments'][label]['paired_method_improvement_vs_E0']=method_difference_ci(
                Path(baseline)/'per_image.jsonl',Path(folder)/'per_image.jsonl',
                config['evaluation']['bootstrap_repetitions'],config['seed'])
        rows.append(row)
    save_json(output/'comparison__E0_global_E0_overlapping_2x2_E1_clean_E2_clean_degraded__same_test_sources.json',result)
    csv_path=output/'comparison__all_experiments__precision_recall_hmean_small_text_recall.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text=['# Phase 3 Text Detection experiment comparison','',
          'All results use the same source-held-out test split and frozen Phase 2 polygon evaluator.',
          'Checkpoint/merge selection used validation only. E1/E2 have equal optimizer-update and effective-batch budgets.','',
          '| Experiment | Clean P (%) | Clean R (%) | Clean Hmean (%) | Degraded P (%) | Degraded R (%) | Degraded Hmean (%) | Gap (pp) | Degraded small-text R (%) |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append('| '+r['experiment']+' | '+' | '.join(f'{r[k]:.3f}' if r[k] is not None else 'N/A' for k in list(r)[2:])+' |')
    text+=['','## Scope and interpretation','',
        'Small-text means original GT bounding-box height <16 px. Improvement must increase degraded Hmean, '
        'not merely reduce the gap by lowering clean accuracy. Check precision, recall and both clean/degraded results.',
        'Method difference CIs are paired source bootstrap in the JSON report; these are distinct from clean/degraded gap CIs.',
        'No E3 consistency loss, restoration, recognition, Qwen or regeneration of degraded input images was run.','',
        '## Existing data and timing limitations','',*['- '+s for s in result['limitations']],
        '- Tiled latency may combine cached E0 global timing with newly measured tile timing; inspect inference summaries.',
        '- Training targets approximate fragmented valid polygons by convex hull and mask outside the common ROI; '
          'the conversion audit records counts. Exact evaluation geometries remain unchanged.','']
    md=output/'RESULTS__phase3_dbnet_monkeyocrv2_as__global_2x2_E1_E2.md'
    md.write_text('\n'.join(text),encoding='utf-8')
    return md,csv_path
