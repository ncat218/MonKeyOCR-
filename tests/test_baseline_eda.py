"""CPU evidence and controller checks; mocked GPU forwards are never real detector results."""
import argparse
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from shapely.geometry import box, mapping, shape

from test_core import REPO, row, tmp
from test_full_asf import full_fixture
from monkey_textdet import pipeline
from monkey_textdet.baseline_eda import word_geometry, select_review, analyze_native
from monkey_textdet.baseline_reference import (AP, COMPLETE, RECEIPT, INFER_CONFIG, SELECTION, COMPARISON,
    bind_reference, reference_files, verify_prediction_records)
from monkey_textdet.baseline_suite import MARKER
from monkey_textdet.common import identity,jsonl,read_json,save_json,sha256,write_jsonl
from monkey_textdet.data import freeze_splits
from monkey_textdet.full_data import build_full_shared
from monkey_textdet.inference import tasks_for
from monkey_textdet.mask_sweep import prediction_file, threshold_tag
from monkey_textdet.vendor.evaluate_shared import evaluate, evaluate_image


def fixture_predictions(manifest,locations,model,checkpoint,config,output,masks,flags,drop_degraded=False):
    """Fake boxes from GT for a miniature orchestration test, not a model simulation."""
    signature=dict(manifest_sha256=sha256(manifest),checkpoint_sha256=checkpoint,
        inference_config_sha256=config,masks=masks,runtime_math_flags=flags)
    save_json(output/'immutable_map_sweep_identity.json',signature)
    source={r['source_image_id']:r for r in jsonl(manifest)}
    for mask in masks:
        records=[]
        for task in tasks_for(manifest,locations):
            record={k:v for k,v in task.items() if k not in ('path','size')}
            polygons=[list(sum(([x,y] for x,y in shape(g['geometry']).exterior.coords),[]))
                      for g in source[task['source_image_id']]['gt_'+task['condition']]['valid']]
            if drop_degraded and task['condition']=='degraded':
                polygons=[]
            record.update(model=model+'__'+threshold_tag(mask),polygons=polygons,scores=[.9]*len(polygons),
                manifest_sha256=sha256(manifest),checkpoint_sha256=checkpoint,inference_config_sha256=config,
                inference_identity=identity(signature),mask_threshold=mask)
            records.append(record)
        write_jsonl(prediction_file(output,mask),records)
    save_json(output/'COMPLETE__all_requested_mask_predictions.json',dict(samples_per_threshold=len(records)))


def completed_reference(root):
    cfg=read_json(REPO/'configs/phase5_full_r1_E0_vs_Aprime_FP32_inference_EDA_h200_normal.json')
    _,paths=full_fixture(root/'datasets')
    workspace=root/'managed';old=workspace/'completed_FP32_fixture';old.mkdir(parents=True)
    frozen=workspace/'frozen_reference_inputs'
    shared=build_full_shared(paths,frozen/'shared')
    splits=freeze_splits(shared['manifest'],frozen/'splits',dict(seed=42,
        split=dict(train=.7,validation=.15,test=.15),tiling=dict(development_sources_max=1)))
    shared['manifest']=str(shared['manifest'])
    save_json(old/RECEIPT,dict(shared=shared,splits=splits))
    save_json(old/COMPLETE,dict(source_fingerprint='a'*64))
    mmocr=root/'mmocr'
    official=mmocr/'configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py'
    official.parent.mkdir(parents=True);official.write_text('original fixture configuration')
    original=mmocr/'model_weight/dbnet_mkv2vitae_icdar2015.pth'
    original.parent.mkdir();original.write_bytes(b'original fixture checkpoint')
    target=old/INFER_CONFIG;target.parent.mkdir(parents=True)
    target.write_text('model='+repr(dict(backbone=dict(use_bf16=False,model_dir=str(mmocr/'pretrained/monkeyocrv2_as')),
        neck=dict(asf_cfg=None)))+'\ntextdet_parameter_precision="fp32"\n')
    flags=cfg['reference_job9793']['runtime_math_flags']
    locations=dict(clean_root=paths['clean_images_root'],degraded_root=paths['degraded_root'])
    locations={k:str(v) for k,v in locations.items()}
    comparison=dict(experiments={})
    for fold,masks in [('validation',[.3,.2,.15,.1,.05]),('test',[.3,.2])]:
        output=old/'outputs/global_predictions'/AP/fold
        fixture_predictions(splits[fold],locations,AP,'b'*64,sha256(target),output,masks,flags)
        stage=('07_validation_masks_' if fold=='validation' else '10_test_native_and_locked_mask_')+AP
        artifacts=[prediction_file(output,t) for t in masks]+[output/'immutable_map_sweep_identity.json',
                  output/'COMPLETE__all_requested_mask_predictions.json']
        save_json(old/'state'/(stage+'__stage_status.json'),dict(status='completed',
            artifacts={str(p.resolve()):sha256(p) for p in artifacts}))
        if fold=='test':
            for mask,suffix in [(.3,'native_mask_0p30'),(.2,'validation_selected_mask')]:
                with contextlib.redirect_stdout(io.StringIO()):
                    metrics=evaluate(splits['test'],prediction_file(output,mask),frozen/'shared/evaluation_protocol.json',
                                     root/('historical_metrics_'+str(mask)),0)
                comparison['experiments'][AP+'__'+suffix]=dict(metrics=metrics)
    save_json(old/COMPARISON,comparison)
    save_json(old/SELECTION,dict(selected=dict(mask_threshold=.2),candidates=[],fold='validation_only'))
    cfg['reference_job9793'].update(run_name=old.name,source_fingerprint='a'*64,checkpoint_sha256='b'*64,
        master_sha256=sha256(shared['manifest']),protocol_sha256=sha256(frozen/'shared/evaluation_protocol.json'),
        split_sha256={k:sha256(splits[k]) for k in ('train','validation','test')},
        source_counts={k:sum(1 for _ in jsonl(splits[k])) for k in ('train','validation','test')},
        small_file_sha256={n:sha256(old/n) for n in (COMPLETE,RECEIPT,INFER_CONFIG,SELECTION,COMPARISON)})
    cfg['run_name']='new_E0_EDA_fixture';cfg['evaluation']['bootstrap_repetitions']=3
    cfg['eda']['review_words_max']=4
    return cfg,paths,workspace,old,splits,locations,mmocr,official,original


class BaselineEDA(unittest.TestCase):
    def test_geometry_partition_merge_and_invalid_fp_follow_vendor_eligibility(self):
        gt=dict(valid=[dict(ann_id=str(i),original_height=10,geometry=mapping(box(x,10,x+10,20)))
                       for i,x in enumerate((10,22,60))],ignored=[dict(geometry=mapping(box(80,50,99,70)))])
        prediction=dict(polygons=[[10,10,32,10,32,20,10,20],[80,50,99,50,99,70,80,70],[]],scores=[.8,.8,.9])
        protocol=dict(min_prediction_roi_fraction=.5,ignore_prediction_overlap=.5,match_iou=.5,word_coverage=.5)
        metric=evaluate_image(gt,mapping(box(0,0,100,80)),prediction,protocol)
        words,fp,_=word_geometry(gt,mapping(box(0,0,100,80)),prediction,metric,protocol)
        self.assertEqual(metric['counts'][:3],[0,3,2])
        self.assertEqual(sum(w['merge_compatible_FN'] for w in words),2)
        self.assertEqual(words[2]['category'],'FN_no_kept_prediction_IoU_ge0p1')
        self.assertEqual(sum(fp.values()),2)
        self.assertEqual(fp['FP_invalid_geometry'],1)

    def test_review_is_deterministic_bounded_and_not_all_one_source(self):
        candidates=[dict(source_image_id=str(i%5),ann_id=str(i),review_stratum=str(i%3)) for i in range(40)]
        a=select_review(candidates,9,42)
        self.assertEqual(a,select_review(list(reversed(candidates)),9,42))
        self.assertEqual(len(a),9)
        self.assertTrue(all(sum(r['source_image_id']==s for r in a)<=2 for s in {r['source_image_id'] for r in a}))

    def test_fragment_signature_is_overlapping_diagnostic_not_a_true_positive(self):
        gt=dict(valid=[dict(ann_id='word',original_height=10,geometry=mapping(box(0,0,20,10)))],ignored=[])
        prediction=dict(polygons=[[0,0,8,0,8,10,0,10],[12,0,20,0,20,10,12,10]],scores=[.8,.8])
        protocol=dict(min_prediction_roi_fraction=.5,ignore_prediction_overlap=.5,match_iou=.5,word_coverage=.5)
        metric=evaluate_image(gt,mapping(box(0,0,100,80)),prediction,protocol)
        words,fp,_=word_geometry(gt,mapping(box(0,0,100,80)),prediction,metric,protocol)
        self.assertEqual(metric['counts'],[0,1,2,1])
        self.assertTrue(words[0]['fragmentation_compatible_FN'])
        self.assertFalse(words[0]['merge_compatible_FN'])
        self.assertEqual(words[0]['category'],'FN_overlap_below_match_IoU')

    def test_plan_only_needs_no_torch_shapely_or_mmengine(self):
        result=subprocess.run([sys.executable,'-S',str(REPO/'run_all.py'),'--plan-only','--config',
            str(REPO/'configs/phase5_full_r1_E0_vs_Aprime_FP32_inference_EDA_h200_normal.json')],
            capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('training',json.loads(result.stdout)['configuration'])

    def test_reference_rejects_changed_records_and_changed_pinned_config_before_compute(self):
        with tmp() as d,contextlib.redirect_stdout(io.StringIO()):
            cfg,_,workspace,old,splits,locations,*_=completed_reference(Path(d))
            bound=bind_reference(workspace,cfg,locations,Path(d)/'binding.json')
            self.assertEqual(bound['images_rehashed'],2*sum(cfg['reference_job9793']['source_counts'][f] for f in ('validation','test')))
            task=next(tasks_for(splits['test'],locations));image=Path(task['path']);saved=image.read_bytes()
            image.write_bytes(b'new dataset version')
            with self.assertRaisesRegex(ValueError,'Current dataset differs'):
                bind_reference(workspace,cfg,locations,Path(d)/'bad_dataset_binding.json')
            image.write_bytes(saved)
            pred=prediction_file(old/'outputs/global_predictions'/AP/'test',.3)
            records=list(jsonl(pred));records[0]['image_sha256']='0'*64;write_jsonl(pred,records)
            signature=read_json(pred.parent.parent/'immutable_map_sweep_identity.json')
            with self.assertRaisesRegex(ValueError,'task mismatch'):
                verify_prediction_records(pred,splits['test'],locations,'b'*64,sha256(old/INFER_CONFIG),signature)
            with self.assertRaisesRegex(ValueError,'recorded reference artifact'):
                bind_reference(workspace,cfg,locations,Path(d)/'binding2.json')
            (old/INFER_CONFIG).write_text('tampered config')
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                reference_files(workspace,cfg)

    def test_controller_only_forwards_E0_reuses_Aprime_reproduces_metrics_and_resumes(self):
        with tmp() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d)
            previous=tempfile.tempdir;tempfile.tempdir=str(root.resolve())
            self.addCleanup(setattr,tempfile,'tempdir',previous)
            cfg,paths,workspace,old,splits,locations,mmocr,official,original=completed_reference(root)
            repo=root/'repo';(repo/'configs').mkdir(parents=True);(repo/'src').mkdir()
            for name in ('run_all.py','constraints-gpu.txt'):(repo/name).write_text('controller fixture')
            save_json(repo/'configs/verified_phase2_manifest_variants.json',{})
            save_json(repo/'configs/verified_phase1_phase2_sources.json',dict(
                checkpoint_sha256=sha256(original),official_config_sha256=sha256(official)))
            anchor=repo/cfg['icdar_historical_anchor']['relative_path'];anchor.parent.mkdir(parents=True)
            anchor.write_bytes((REPO/cfg['icdar_historical_anchor']['relative_path']).read_bytes())
            config=repo/'configs/eda.json';save_json(config,cfg)
            a=argparse.Namespace(repo=repo,config=config,workspace=workspace,mmocr_root=mmocr,
                logs=root/'logs',dataset_parent=str(root/'datasets'))
            old_hashes={str(p):sha256(p) for p in old.rglob('*') if p.is_file()}
            calls=[]
            def worker(self,command,label,cwd=None,env=None):
                command=[str(v) for v in command]
                self_test.assertIn('monkey_textdet.mask_sweep',command)
                def arg(name):return Path(command[command.index('--'+name)+1])
                label=command[command.index('--model-label')+1]
                self_test.assertTrue(label.startswith('E0_original'))
                self_test.assertEqual(sha256(arg('checkpoint')),sha256(original))
                masks=read_json(arg('thresholds'))['mask_thresholds'];calls.append((label,masks))
                fixture_predictions(arg('manifest'),read_json(arg('locations')),label,sha256(original),
                    sha256(arg('model-config')),arg('output'),masks,cfg['reference_job9793']['runtime_math_flags'],True)
            self_test=self
            with patch('monkey_textdet.common.Processes.run',worker):
                pipeline.run(a)
                self.assertEqual(len(calls),2) # validation + test, never A′ or training
                self.assertEqual(calls[0][1],[.3,.2,.15,.1,.05])
                self.assertEqual(calls[1][1],[.3]) # native-first tie, no test grid sweep
                pipeline.run(a)
                self.assertEqual(len(calls),2) # stage resume does not forward again
            run=workspace/cfg['run_name']
            self.assertTrue((run/MARKER).is_file())
            out=run/'outputs/FINAL_EDA__E0_original_vs_Aprime_FP32__same_FULL_R1_test'
            diag=read_json(out/'DIAGNOSIS__native_E0_Aprime__word_size_geometry_and_paired_transitions.json')
            expected=cfg['reference_job9793']['source_counts']['test']
            self.assertEqual(diag['E0_to_Aprime_by_size']['degraded']['8_12']['repaired_by_finetuning'],expected)
            self.assertEqual(diag['paired_clean_degraded_by_size']['E0']['8_12']['clean_TP_degraded_FN'],expected)
            archive=next(run.glob('RESULTS_AND_EDA*.zip'))
            with zipfile.ZipFile(archive) as z:
                self.assertIsNone(z.testzip())
                self.assertTrue(any(n.endswith('.png') for n in z.namelist()))
                self.assertFalse(any(n.endswith('.pth') or n.endswith('.jsonl') or Path(n).name.startswith('WORDS__') for n in z.namelist()))
            self.assertEqual(old_hashes,{str(p):sha256(p) for p in old.rglob('*') if p.is_file()})


if __name__=='__main__':
    unittest.main()
