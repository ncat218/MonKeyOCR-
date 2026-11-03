"""Frozen-input controls and FP32 orchestration on disposable CPU fixtures only."""
import argparse
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_core import REPO, tmp
from test_full_asf import full_fixture
from monkey_textdet.common import identity, jsonl, read_json, save_json, sha256, write_jsonl
from monkey_textdet.data import freeze_splits, export_training_annotations
from monkey_textdet.full_data import build_full_shared
from monkey_textdet.fp32_checks import update_summary, parameter_group
from monkey_textdet.precision_control import apply_precision, bind_reference, initial_validation_rows, pre_submit_reference, verify_probe_pair
from monkey_textdet.variants import VARIANTS


def reference_fixture(root):
    cfg, paths = full_fixture(root/'datasets')
    new = read_json(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_FP32_control_job9276_h200_normal.json')
    workspace = root/'managed'
    old = workspace/'old_job9276_fixture'
    shared_dir = old/'inputs/02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT'
    shared = build_full_shared(paths, shared_dir)
    locations = dict(clean_root=paths['clean_images_root'], degraded_root=paths['degraded_root'])
    save_json(shared_dir/'dataset_locations.json', locations)
    split_dir = old/'inputs/03_frozen_detection_source_splits__official_train_intersection__70_15_15'
    splits = freeze_splits(shared['manifest'], split_dir, cfg)
    ann_dir = old/'inputs/04_shared_mixed_training_and_exact_validation_annotations'
    annotations = export_training_annotations(splits, locations, ann_dir,
        train_conditions=('E2_clean_and_degraded_balanced',))
    for stage,result,folder in [
        ('01_audit_full_MAIN_and_freeze_official_train_GT_intersection',shared,shared_dir),
        ('02_freeze_same_source_clean_SHA_grouped_train_validation_test',splits,split_dir),
        ('03_export_same_mixed_targets_for_DBNet_and_ASF',annotations,ann_dir)]:
        save_json(old/'state'/(stage+'__stage_status.json'),dict(status='completed', result=result,
            artifacts={str(p.resolve()):sha256(p) for p in folder.iterdir()}))
    new['run_name'] = 'FP32_fixture_new_run'
    ref = new['precision_control']
    ref.update(reference_run=old.name, reference_source_fingerprint='a'*64,
        master_sha256=shared['manifest_sha256'], protocol_sha256=sha256(shared_dir/'evaluation_protocol.json'),
        split_sha256={k:sha256(p) for k,p in splits.items()}, diagnostic_validation_sources=1)
    initial = old/ref['ASF_initial_relative_path']
    initial.parent.mkdir(parents=True)
    initial.write_bytes(b'original frozen ASF initialization fixture')
    ref['ASF_initial_sha256'] = sha256(initial)
    save_json(old/'COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json',
              dict(source_fingerprint='a'*64))
    save_json(old/'run_configuration__FULL_R1_two_architectures_same_budget.json',cfg)
    stage_file=old/'state/01_audit_full_MAIN_and_freeze_official_train_GT_intersection__stage_status.json'
    stage=read_json(stage_file)
    provenance=read_json(shared_dir/'data_audit__FULL_R1_counts_GT_intersection_and_exclusions.json')['generator']
    stage['input_fingerprint']=identity(dict(files=dict(provenance['metadata_sha256'],
        main=sha256(paths['main_image_manifest']),clean=sha256(paths['clean_json']),textocr=sha256(paths['textocr_json'])),source='a'*64))
    save_json(stage_file,stage)
    history=old/'outputs/FINAL_COMPARISON__FULL_R1_DBNet_vs_ASF__native_and_validation_selected_masks/COMPARISON__DBNet_ASF_native_and_selected_masks__all_metrics_and_word_errors.json'
    save_json(history,dict(experiments={v+'__native_mask_0p30':dict(metrics=dict(
        manifest_sha256=sha256(splits['test']),clean_unique=dict(hmean=.2),degraded_instance=dict(hmean=.1)))
        for v in VARIANTS},fixture_only=True))
    ref['reference_comparison_sha256']=sha256(history)
    return new, paths, workspace, old, shared, splits, annotations


class FP32Control(unittest.TestCase):
    def test_initial_validation_subset_includes_largest_global_resized_canvas(self):
        with tmp() as temp:
            path=Path(temp)/'validation.jsonl'
            rows=[dict(source_image_id=name,clean_size=size) for name,size in
                  [('first',[100,100]),('second',[640,480]),('largest',[4000,1000])]]
            write_jsonl(path,rows)
            selected=initial_validation_rows(path,2)
            self.assertEqual([r['source_image_id'] for r in selected],['first','largest'])
            self.assertEqual(initial_validation_rows(path,2),selected)

    def test_published_config_changes_only_precision_and_diagnostic_control_fields(self):
        old=read_json(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json')
        new=read_json(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_FP32_control_job9276_h200_normal.json')
        from monkey_textdet.asf_suite import validate
        self.assertEqual(validate(new)['max_iters'],20000)
        control=new.pop('precision_control')
        self.assertEqual(new['training'].pop('parameter_precision'),'fp32')
        for key in ('run_name','scope'):
            new.pop(key);old.pop(key)
        self.assertEqual(new,old)
        self.assertEqual(control['reference_run'],read_json(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json')['run_name'])

    def test_reference_reuses_exact_files_without_copy_or_mutation_and_rejects_changed_images(self):
        with tmp() as temp:
            cfg,paths,workspace,old,shared,splits,ann=reference_fixture(Path(temp))
            before={str(p):sha256(p) for p in old.rglob('*') if p.is_file()}
            new=workspace/cfg['run_name']
            result=bind_reference(workspace,cfg,paths,new/'shared',new/'splits',new/'annotations')
            self.assertEqual(result['annotations'],ann)
            self.assertEqual(result['splits'],splits)
            self.assertEqual(before,{str(p):sha256(p) for p in old.rglob('*') if p.is_file()})
            self.assertFalse(list(new.rglob('*.pth')))
            self.assertFalse(list(new.rglob('*.jsonl')))
            first=next(jsonl(shared['manifest']))
            image=Path(paths['clean_images_root'])/first['clean_relpath']
            image.write_bytes(b'changed actual source bytes')
            with self.assertRaisesRegex(ValueError,'Frozen source image'):
                bind_reference(workspace,cfg,paths,new/'shared',new/'splits',new/'annotations')

    def test_reference_rejects_missing_artifact_before_submission_and_wrong_frozen_initial(self):
        with tmp() as temp:
            cfg,paths,workspace,old,shared,_,_=reference_fixture(Path(temp))
            Path(shared['manifest']).rename(Path(shared['manifest']).with_suffix('.missing'))
            with self.assertRaisesRegex(RuntimeError,'missing/modified'):
                pre_submit_reference(workspace,cfg)
        with tmp() as temp:
            cfg,paths,workspace,old,_,_,_=reference_fixture(Path(temp))
            (old/cfg['precision_control']['ASF_initial_relative_path']).write_bytes(b'new random weights')
            new=workspace/cfg['run_name']
            with self.assertRaisesRegex(ValueError,'ASF initial bytes changed'):
                bind_reference(workspace,cfg,paths,new/'shared',new/'splits',new/'annotations')

    def test_probe_pair_rejects_legacy_finite_only_probe_and_mismatched_sample_order(self):
        probe=dict(parameter_storage='float32',observed_optimizer_updates=1,
            actual_OptimWrapper_with_declared_accumulation_clipping_and_warmup=True,
            sample_ids_by_microiteration=[['sample1'],['sample2']],
            optimizer_lr_before_each_microiteration=[[2e-6],[2.1e-6]])
        probes={v:copy.deepcopy(probe) for v in VARIANTS}
        verify_probe_pair(probes)
        probes[VARIANTS[1]]['sample_ids_by_microiteration']=[['different_source']]
        with self.assertRaisesRegex(RuntimeError,'sample order'):
            verify_probe_pair(probes)
        probes[VARIANTS[1]]={ 'parameters_with_gradients':372 }
        with self.assertRaisesRegex(RuntimeError,'real FP32'):
            verify_probe_pair(probes)

    def test_module_update_summary_requires_real_movement_and_finite_shapes(self):
        self.assertEqual(update_summary([1.,2.],[1.,2.])['changed_elements'],0)
        self.assertEqual(update_summary([1.,2.],[1.,2.001])['changed_elements'],1)
        self.assertEqual(parameter_group('neck.asf_attn.conv.weight'),'ASF')
        self.assertEqual(parameter_group('neck.lateral_convs.0.weight'),'neck_without_ASF')
        with self.assertRaises(RuntimeError):
            update_summary([1.],[float('nan')])

    def test_controller_fp32_preflights_both_branches_then_trains_and_packages_once(self):
        from mmengine.config import Config
        from monkey_textdet import pipeline
        from monkey_textdet.mask_sweep import prediction_file
        import zipfile
        with tmp() as temp:
            root=Path(temp)
            previous_tempdir=tempfile.tempdir;tempfile.tempdir=str(root.resolve())
            self.addCleanup(setattr,tempfile,'tempdir',previous_tempdir)
            cfg,paths,workspace,old,_,_,_=reference_fixture(root)
            cfg['evaluation']['bootstrap_repetitions']=2
            repo=root/'fake_repo';(repo/'configs').mkdir(parents=True);(repo/'src').mkdir()
            for name in ('run_all.py','constraints-gpu.txt'):(repo/name).write_text('CPU fixture only')
            save_json(repo/'configs/verified_phase2_manifest_variants.json',{})
            mmocr=root/'mmocr'
            official=mmocr/'configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py'
            official.parent.mkdir(parents=True)
            official.write_text("model=dict(type='DBNet',backbone=dict(type='Fixture',use_bf16=True),neck=dict(type='FPNC'),det_head=dict(type='DBHead'))\ntrain_dataloader=dict(dataset=dict(pipeline=[]))\nval_dataloader=dict(dataset=dict(pipeline=[]))\ntest_dataloader=dict(dataset=dict(pipeline=[]))\n")
            original=mmocr/'model_weight/dbnet_mkv2vitae_icdar2015.pth'
            original.parent.mkdir();original.write_bytes(b'fixture original')
            locks=read_json(REPO/'configs/verified_phase1_phase2_sources.json')
            locks.update(official_config_sha256=sha256(official),checkpoint_sha256=sha256(original))
            save_json(repo/'configs/verified_phase1_phase2_sources.json',locks)
            config=repo/'configs/fp32.json';save_json(config,cfg)
            args=argparse.Namespace(repo=repo,config=config,workspace=workspace,mmocr_root=mmocr,
                logs=root/'logs',dataset_parent=str(root/'datasets'))
            calls=[]
            def worker(process,command,label,cwd=None,env=None):
                cmd=[str(x) for x in command];calls.append(label)
                def arg(name):return Path(cmd[cmd.index('--'+name)+1])
                if 'monkey_textdet.variants' in cmd:
                    model=Config.fromfile(str(arg('config')),import_custom_modules=False)
                    self.assertFalse(model.model.backbone.use_bf16)
                    if '--reuse-initial' in cmd:
                        self.assertEqual(arg('output'),old/cfg['precision_control']['ASF_initial_relative_path'])
                    save_json(arg('receipt'),dict(fixture_only=True))
                elif 'monkey_textdet.training' in cmd:
                    model=Config.fromfile(str(arg('config')),import_custom_modules=False)
                    self.assertEqual(model.textdet_parameter_precision,'fp32')
                    self.assertFalse(model.default_hooks.logger.log_metric_by_epoch)
                    if '--probe-only' in cmd:
                        save_json(arg('result'),dict(parameter_storage='float32',observed_optimizer_updates=1,
                            actual_OptimWrapper_with_declared_accumulation_clipping_and_warmup=True,
                            sample_ids_by_microiteration=[['same']],optimizer_lr_before_each_microiteration=[[2e-6]],fixture_only=True))
                    else:
                        work=arg('work-dir');work.mkdir(parents=True,exist_ok=True)
                        weights=work/'FINAL_WEIGHTS_ONLY__fake.pth';weights.write_bytes(b'fixture trained')
                        save_json(arg('result'),dict(evaluation_checkpoint=str(weights),fixture_only=True))
                        save_json(work/'TRAINING_AUDIT__actual_FP32_named_parameter_changes__initial_to_final.json',dict(fixture_only=True))
                elif 'monkey_textdet.mask_sweep' in cmd:
                    model=Config.fromfile(str(arg('model-config')),import_custom_modules=False)
                    self.assertFalse(model.model.backbone.use_bf16)
                    folder=arg('output');folder.mkdir(parents=True,exist_ok=True)
                    predictions=[];attention=[]
                    for row in jsonl(arg('manifest')):
                        for condition in ('clean','degraded'):
                            poly=row['gt_'+condition]['valid'][0]['geometry']['coordinates'][0][:-1]
                            sample='clean::'+row['source_image_id'] if condition=='clean' else 'deg::'+row['variant_id']
                            predictions.append(dict(sample_id=sample,source_image_id=row['source_image_id'],condition=condition,
                                manifest_sha256=sha256(arg('manifest')),model='fixture',polygons=[[v for p in poly for v in p]],scores=[.9]))
                            attention.append(dict(sample_id=sample,source_image_id=row['source_image_id'],condition=condition,
                                **{k:[.5]*4 for k in ('attention_mean','fraction_attention_below_0p05','fraction_attention_above_0p95','feature_rms_before_ASF','feature_rms_after_ASF')}))
                    for t in read_json(arg('thresholds'))['mask_thresholds']:
                        write_jsonl(prediction_file(folder,t),predictions)
                    if '--record-attention' in cmd:
                        write_jsonl(folder/'DIAGNOSTICS__real_ASF_attention_and_feature_RMS__per_image.jsonl',attention)
                    save_json(folder/'COMPLETE__all_requested_mask_predictions.json',dict(fixture_only=True))
                    save_json(folder/'immutable_map_sweep_identity.json',dict(fixture_only=True))
                else:raise AssertionError(cmd)
            before={str(p):sha256(p) for p in old.rglob('*') if p.is_file()}
            with patch('monkey_textdet.common.Processes.run',worker),contextlib.redirect_stdout(io.StringIO()):
                pipeline.run(args)
                first_count=len(calls)
                pipeline.run(args)
            self.assertEqual(len(calls),first_count)
            self.assertLess(next(i for i,c in enumerate(calls) if c.startswith(VARIANTS[1]) and 'compatibility' in c),
                            next(i for i,c in enumerate(calls) if 'mixed_finetune' in c))
            self.assertEqual(before,{str(p):sha256(p) for p in old.rglob('*') if p.is_file()})
            packet=workspace/cfg['run_name']/'RESULTS_AND_DIAGNOSTICS__FULL_R1_FP32_DBNet_ASF__final_weights_only.zip'
            with zipfile.ZipFile(packet) as z:
                self.assertIsNone(z.testzip())
                self.assertFalse(any(n.endswith(('.pth','.jsonl')) for n in z.namelist()))
                self.assertTrue(any('ATTENTION__same_validation_sources' in n for n in z.namelist()))
                self.assertTrue(any('PRECISION_REGIMES__native_BF16_job9276' in n for n in z.namelist()))


if __name__=='__main__':unittest.main()
