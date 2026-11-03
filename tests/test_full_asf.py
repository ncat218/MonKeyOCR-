"""FULL_R1 metadata, fair variants, validation selection and reporting on CPU fixtures."""
import copy
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_core import REPO, tmp
from test_pipeline_integration import miniature_dataset
from monkey_textdet.common import jsonl, read_json, save_json, sha256, write_jsonl
from monkey_textdet.data import discover_inputs, freeze_splits
from monkey_textdet.full_data import FULL_PROTOCOL, build_full_shared, read_metadata
from monkey_textdet.asf_suite import validate
from monkey_textdet.mask_sweep import choose_mask
from monkey_textdet.mask_sweep import prediction_file, threshold_tag
from monkey_textdet.variants import initialization_key_audit
from monkey_textdet.asf_reporting import summarize_words
from monkey_textdet.vendor.evaluate_shared import evaluate_image


def full_fixture(root):
    miniature_dataset(root)
    deg = root/'textvqa_degraded_10k'
    deg.rename(root/'full_textvqa_degraded')
    deg = root/'full_textvqa_degraded'
    main = deg/'main/image_manifest.csv'
    with main.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(source_image_id=r['image_id'], split='train', clean_width='100', clean_height='80',
                 degraded_width='100', degraded_height='80')
    # Official val/test have no matching train TextOCR GT and must not be moved
    # into the detection train split or silently treated as annotated test data.
    for fold in ('val', 'test'):
        r = dict(rows[0], image_id='official_'+fold, source_image_id='official_'+fold, split=fold)
        rows.append(r)
    qa = deg/'metadata/locked_eval_manifest.csv'
    with qa.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['image_id', 'source_image_id', 'split', 'question_id', 'question'])
        w.writeheader()
        for i, r in enumerate(rows):
            for q in range(2):
                w.writerow(dict(image_id=r['image_id'], source_image_id=r['image_id'], split=r['split'],
                                question_id=str(2*i+q), question='fixture question'))
    frozen = deg/'config/frozen_degradation_config.json'
    save_json(frozen, dict(protocol=FULL_PROTOCOL, source_manifest_sha256=sha256(qa),
        human_review_performed=False,
        source_provenance=dict(annotation_sha256=dict(train=sha256(root/'textvqa_clean/TextVQA_0.5.1_train.json')),
                               counts=dict(train=dict(images=12), val=dict(images=1), test=dict(images=1)))))
    save_json(deg/'metadata/split_policy.json', dict(official_splits_preserved=True))
    for r in rows:
        r.update(config_sha256=sha256(frozen), source_manifest_sha256=sha256(qa))
    with main.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cfg = read_json(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json')
    return cfg, discover_inputs(root, cfg['data'])


class FullASF(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('mmengine'), 'MMEngine Config required')
    def test_full_controller_two_trainings_validation_only_selection_and_resume(self):
        from monkey_textdet import pipeline
        with tmp() as d:
            root = Path(d)
            # MMEngine creates intermediate config files. Keep them inside the
            # writable test fixture on Windows, just like test_config_generation.
            prior_tempdir = tempfile.tempdir
            tempfile.tempdir = str(root.resolve())
            self.addCleanup(setattr, tempfile, 'tempdir', prior_tempdir)
            cfg, _ = full_fixture(root/'datasets')
            cfg['evaluation']['bootstrap_repetitions'] = 2
            repo = root/'fake_repo'; (repo/'configs').mkdir(parents=True); (repo/'src').mkdir()
            for name in ('run_all.py', 'constraints-gpu.txt'):
                (repo/name).write_text('controller fixture only')
            save_json(repo/'configs/verified_phase2_manifest_variants.json', {})
            mmocr = root/'mmocr'
            official = mmocr/'configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py'
            official.parent.mkdir(parents=True)
            official.write_text('''
model=dict(type='DBNet',backbone=dict(type='FixtureBackbone'),neck=dict(type='FPNC'),det_head=dict(type='DBHead'))
train_dataloader=dict(dataset=dict(pipeline=[dict(type='PackTextDetInputs',meta_keys=('img_path',))]))
val_dataloader=dict(dataset=dict(pipeline=[dict(type='PackTextDetInputs',meta_keys=('img_path','scale_factor'))]))
test_dataloader=dict(dataset=dict(pipeline=[dict(type='PackTextDetInputs',meta_keys=('img_path','scale_factor'))]))
''')
            original = mmocr/'model_weight/dbnet_mkv2vitae_icdar2015.pth'
            original.parent.mkdir(); original.write_bytes(b'fixture original detector')
            locks = read_json(REPO/'configs/verified_phase1_phase2_sources.json')
            locks.update(official_config_sha256=sha256(official), checkpoint_sha256=sha256(original))
            save_json(repo/'configs/verified_phase1_phase2_sources.json', locks)
            config = repo/'configs/full.json'; save_json(config, cfg)
            a = argparse.Namespace(repo=repo, config=config, workspace=root/'managed',
                mmocr_root=mmocr, logs=root/'logs', dataset_parent=str(root/'datasets'))
            calls, inits, folds = [], [], []
            def fake_worker(self, command, label, cwd=None, env=None):
                command = [str(x) for x in command]; calls.append(label)
                def arg(name): return Path(command[command.index('--'+name)+1])
                if 'monkey_textdet.variants' in command:
                    output = arg('output'); output.parent.mkdir(parents=True, exist_ok=True)
                    if '--verify-only' not in command:
                        output.write_bytes(b'fixture initialized weights')
                    inits.append(str(arg('original')))
                    save_json(arg('receipt'), dict(fixture_only=True))
                elif 'monkey_textdet.training' in command:
                    if '--probe-only' in command:
                        save_json(arg('result'), dict(fixture_only=True)); return
                    work = arg('work-dir'); work.mkdir(parents=True, exist_ok=True)
                    best = work/'FINAL_WEIGHTS_ONLY__fake_completed_training.pth'; best.write_bytes(b'fixture trained')
                    save_json(arg('result'), dict(evaluation_checkpoint=str(best), fixture_only=True))
                elif 'monkey_textdet.mask_sweep' in command:
                    manifest, out = arg('manifest'), arg('output'); out.mkdir(parents=True, exist_ok=True)
                    folds.append(manifest.name)
                    model = command[command.index('--model-label')+1]
                    masks = read_json(arg('thresholds'))['mask_thresholds']
                    for t in masks:
                        predictions = []
                        for row in jsonl(manifest):
                            for cond in ('clean', 'degraded'):
                                poly = row['gt_'+cond]['valid'][0]['geometry']['coordinates'][0][:-1]
                                predictions.append(dict(sample_id=('clean::'+row['source_image_id'] if cond=='clean' else 'deg::'+row['variant_id']),
                                    source_image_id=row['source_image_id'], condition=cond,
                                    manifest_sha256=sha256(manifest), model=model+'__'+threshold_tag(t),
                                    polygons=[[v for p in poly for v in p]], scores=[.9]))
                        write_jsonl(prediction_file(out,t), predictions)
                    save_json(out/'COMPLETE__all_requested_mask_predictions.json', dict(fixture_only=True))
                    save_json(out/'immutable_map_sweep_identity.json', dict(fixture_only=True))
                else:
                    raise AssertionError(command)
            with patch('monkey_textdet.common.Processes.run', fake_worker), contextlib.redirect_stdout(io.StringIO()):
                pipeline.run(a)
                completed = len(calls)
                pipeline.run(a)
            self.assertEqual(len(calls), completed)
            self.assertEqual(inits, [str(original), str(original)])
            self.assertEqual(sum('validation' in f for f in folds), 2)
            self.assertEqual(sum('manifest__test__' in f for f in folds), 2)
            marker = a.workspace/cfg['run_name']/'COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json'
            status = read_json(marker)
            self.assertEqual(len(status['experiments']), 4)
            self.assertTrue(Path(status['final']['report']).is_file())
            annotations = a.workspace/cfg['run_name']/'inputs/04_shared_mixed_training_and_exact_validation_annotations'
            self.assertFalse(list(annotations.glob('*E1_clean_only*')))
            self.assertEqual(original.read_bytes(), b'fixture original detector')

    def test_full_QA_duplicates_are_valid_and_only_official_train_enters_detection(self):
        with tmp() as d:
            root = Path(d)
            cfg, paths = full_fixture(root)
            records, audit = read_metadata(paths)
            self.assertEqual(len(records), 14)
            self.assertEqual(audit['qa_questions'], 28)
            self.assertEqual(audit['main_counts_by_official_split'], dict(train=12, val=1, test=1))
            result = build_full_shared(paths, root/'shared')
            rows = list(jsonl(result['manifest']))
            self.assertEqual(len(rows), 12)
            self.assertEqual({r['official_split'] for r in rows}, {'train'})
            self.assertTrue(all(r['clean_size'] == r['degraded_size'] == [100,80] for r in rows))
            split = freeze_splits(result['manifest'], root/'splits', cfg)
            members = [set(r['source_image_id'] for r in jsonl(split[f])) for f in ('train','validation','test')]
            self.assertEqual(len(set.union(*members)), 12)
            self.assertFalse(members[0]&members[1] or members[0]&members[2] or members[1]&members[2])

    def test_full_main_mixed_config_or_duplicate_variant_fails_before_images(self):
        with tmp() as d:
            _, paths = full_fixture(Path(d))
            main = Path(paths['main_image_manifest'])
            with main.open(newline='') as f:
                records = list(csv.DictReader(f))
            records[0]['config_sha256'] = '0'*64
            with main.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
            with self.assertRaisesRegex(ValueError, 'Mixed FULL_R1'):
                read_metadata(paths)
            records[0]['config_sha256'] = records[1]['config_sha256']
            records.append(records[0])
            with main.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
            with self.assertRaisesRegex(ValueError, 'exactly one MAIN'):
                read_metadata(paths)

    def test_source_pixel_change_is_rejected_not_hidden_as_GT_exclusion(self):
        with tmp() as d:
            root = Path(d)
            _, paths = full_fixture(root)
            p = next((root/'textvqa_clean/train_images').glob('*.png'))
            p.write_bytes(b'corrupted source')
            with self.assertRaisesRegex(ValueError, 'bytes/path'):
                build_full_shared(paths, root/'shared')

    def test_ASF_loading_allows_only_two_new_neck_submodules(self):
        source = dict(backbone=np.zeros((3,2)), head=np.zeros((2,)))
        native = copy.deepcopy(source)
        self.assertEqual(initialization_key_audit(native, source, False), [])
        asf = dict(native, **{'neck.asf_conv.weight':np.zeros((2,2)), 'neck.asf_attn.weight':np.zeros((1,2))})
        self.assertEqual(len(initialization_key_audit(asf, source, True)), 2)
        for unsafe in [dict(asf, head=np.zeros((3,))), dict(asf, **{'backbone.new':np.zeros((2,))}),
                       {k:v for k,v in asf.items() if k != 'backbone'}]:
            with self.assertRaises(ValueError):
                initialization_key_audit(unsafe, source, True)

    def test_validation_selection_balances_conditions_and_native_wins_tie(self):
        candidates = [dict(mask_threshold=.3, balanced_validation_hmean=.6),
                      dict(mask_threshold=.1, balanced_validation_hmean=.6)]
        self.assertEqual(choose_mask(candidates)['mask_threshold'], .3)
        candidates[1]['balanced_validation_hmean'] = .61
        self.assertEqual(choose_mask(candidates)['mask_threshold'], .1)
        candidates[1]['balanced_validation_hmean'] = float('nan')
        with self.assertRaises(ValueError):
            choose_mask(candidates)

    def test_full_config_budget_and_ambiguous_mask_names(self):
        cfg = read_json(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json')
        self.assertEqual(validate(cfg)['max_iters'], 20000)
        cfg['mask_thresholds_validation_only'] = [.3, .301]
        with self.assertRaisesRegex(ValueError, 'unique two-decimal'):
            validate(cfg)

    def test_word_survival_and_corruption_use_same_annotation_IDs(self):
        with tmp() as d:
            root = Path(d)
            _, paths = full_fixture(root)
            shared = build_full_shared(paths, root/'shared')
            row = next(jsonl(shared['manifest']))
            protocol = read_json(root/'shared/evaluation_protocol.json')
            coords = [20,20,40,20,40,30,20,30]
            clean = evaluate_image(row['gt_clean'], row['roi_clean'], dict(polygons=[coords],scores=[.8]), protocol)
            degraded = evaluate_image(row['gt_degraded'], row['roi_degraded'], dict(polygons=[],scores=[]), protocol)
            items = [dict(clean, source_image_id=row['source_image_id'], condition='clean'),
                     dict(degraded, source_image_id=row['source_image_id'], condition='degraded')]
            result = summarize_words([row], items)
            self.assertEqual(result['small_word_transitions']['clean_only'], 1)
            self.assertEqual(result['small_word_transitions']['retention_of_small_clean_TP'], 0)
            self.assertEqual(result['height_bins_original_pixels']['8_12']['clean_recall'], 1)


if __name__ == '__main__':
    unittest.main()
