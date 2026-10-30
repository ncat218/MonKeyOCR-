"""CPU checks on temporary miniature fixtures; never execute model training or create the dataset."""
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box,mapping,shape,Polygon,MultiPolygon
from shapely.ops import unary_union

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from monkey_textdet.common import StageState,canonical,read_json,resume_jsonl,sha256,write_jsonl
from monkey_textdet.data import discover_inputs,split_source_groups,instances_for,export_training_annotations
from monkey_textdet.inference import tasks_for
from monkey_textdet.tiling import tile_boxes,translate_polygon,seam_eligible,merge_views
from monkey_textdet.training import training_schedule
from monkey_textdet.pipeline import validate_config

REPO=Path(__file__).resolve().parents[1]


def tmp():
    parent=REPO/'_workspace_cpu_tests';parent.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def row(source='source_01',digest='a'*64):
    gt=dict(valid=[dict(ann_id=source+'_1',geometry=mapping(box(20,20,40,35)),original_height=15,visibility=1.)],ignored=[])
    return dict(source_image_id=source,variant_id=source+'_variant',clean_sha256=digest,degraded_sha256='b'*64,
        clean_relpath='train_images/'+source+'.jpg',degraded_relpath='main/images/'+source+'.png',
        clean_size=[100,80],degraded_size=[100,80],gt_clean=gt,gt_degraded=copy.deepcopy(gt),
        roi_clean=mapping(box(0,0,100,80)),roi_degraded=mapping(box(0,0,100,80)))


class CoreTests(unittest.TestCase):
    def test_tiles_cover_canvas(self):
        for w,h in [(1024,683),(33,101),(1,1),(2,3)]:
            tiles=tile_boxes(w,h,.2)
            self.assertTrue(unary_union([box(*t) for t in tiles]).equals(box(0,0,w,h)))
            self.assertLessEqual(len(tiles),4)
            self.assertTrue(all(0<=x0<x1<=w and 0<=y0<y1<=h for x0,y0,x1,y1 in tiles))

    def test_tile_overlap_definition(self):
        tiles=tile_boxes(1000,800,.2)
        left,right=tiles[0],tiles[2]
        self.assertAlmostEqual((left[2]-right[0])/(left[2]-left[0]),.2,delta=.005)

    def test_invalid_tile_overlap_rejected(self):
        for overlap in [-1,1]:
            with self.assertRaises(ValueError):tile_boxes(100,80,overlap)

    def test_tile_coordinates_return_to_original_image(self):
        self.assertEqual(translate_polygon([0,0,10,0,10,5,0,5],50,20),[50,20,60,20,60,25,50,25])

    def test_internal_seam_rejection_preserves_outer_image_edge(self):
        self.assertTrue(seam_eligible([0,0,10,0,10,10,0,10],[0,0,60,60],[100,100],2))
        self.assertFalse(seam_eligible([50,20,61,20,61,30,50,30],[0,0,60,60],[100,100],2))

    def test_polygon_nms_suppresses_duplicate(self):
        p=[10,10,30,10,30,30,10,30]
        r=merge_views([dict(polygons=[p,p],scores=[.8,.9],eligible=[True,True])],.5)
        self.assertEqual(r['scores'],[.9]);self.assertEqual(r['suppressed'],1)

    def test_nms_uses_polygon_not_axis_aligned_bbox(self):
        r=merge_views([dict(polygons=[[0,0,10,0,0,10],[10,10,0,10,10,0]],scores=[.9,.8],eligible=[True,True])],.3)
        self.assertEqual(len(r['polygons']),2)

    def test_nms_excludes_seam_candidates_but_keeps_degenerate_fp(self):
        r=merge_views([dict(polygons=[[0,0,5,0,5,5],[1,1,1,1,1,1]],scores=[.9,.8],eligible=[False,True])],.5)
        self.assertEqual(len(r['polygons']),1)

    def test_discovery_accepts_renamed_dataset_parents(self):
        with tmp() as d:
            root=Path(d)
            for rel in ['arbitrary clean name/TextVQA_0.5.1_train.json','GT renamed/TextOCR_0.1_train.json',
                        'different degraded name/textvqa_realistic_v2_10k/main/image_manifest.csv']:
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
            actual=discover_inputs(str(root),{})
            self.assertTrue(actual['degraded_root'].endswith('textvqa_realistic_v2_10k'))

    def test_discovery_fails_on_multiple_versions(self):
        with tmp() as d:
            root=Path(d)
            for rel in ['a/TextVQA_0.5.1_train.json','b/TextVQA_0.5.1_train.json']:
                p=root/rel;p.parent.mkdir();p.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'clean_json'):discover_inputs(str(root),{})

    def test_split_deterministic_and_source_byte_groups_do_not_leak(self):
        rows=[row(str(i),hashlib.sha256(str(i).encode()).hexdigest()) for i in range(20)]
        rows.append(row('duplicate_source_alias',rows[0]['clean_sha256']))
        fractions=dict(train=.7,validation=.15,test=.15)
        first=split_source_groups(rows,fractions,42)
        self.assertEqual(first,split_source_groups(list(reversed(rows)),fractions,42))
        membership={s:k for k,ids in first.items() for s in ids}
        self.assertEqual(membership['0'],membership['duplicate_source_alias'])
        self.assertEqual(len(membership),21)

    def test_split_rejects_source_id_with_different_bytes(self):
        with self.assertRaises(ValueError):split_source_groups([row(),row(digest='c'*64)],dict(train=.7,validation=.15,test=.15),42)

    def test_outside_roi_ignore_does_not_fill_valid_roi_hole(self):
        import collections
        r=row();r['roi_clean']=mapping(box(10,10,90,70));counts=collections.Counter()
        instances=instances_for(r,'clean',counts)
        ignored=[Polygon(np.asarray(x['polygon']).reshape(-1,2)) for x in instances if x['ignore']]
        union=unary_union(ignored)
        self.assertAlmostEqual(union.area,8000-80*60)
        self.assertEqual(union.intersection(box(10,10,90,70)).area,0)

    def test_fragmented_training_target_conversion_does_not_change_eval_gt(self):
        import collections
        r=row();r['gt_clean']['valid'][0]['geometry']=mapping(MultiPolygon([box(20,20,25,25),box(30,20,35,25)]))
        before=canonical(r);audit=collections.Counter()
        records=instances_for(r,'clean',audit)
        self.assertEqual(canonical(r),before)
        self.assertEqual(audit['training_valid_convex_hull_approximations'],1)
        self.assertEqual(len([x for x in records if not x['ignore']]),1)

    def test_export_E1_E2_have_same_source_pool_and_test_not_in_training(self):
        with tmp() as d:
            root=Path(d);splits={}
            for fold,ids in [('train',['a','b']),('validation',['c'])]:
                path=root/(fold+'.jsonl');write_jsonl(path,[row(s,hashlib.sha256(s.encode()).hexdigest()) for s in ids]);splits[fold]=str(path)
            result=export_training_annotations(splits,dict(clean_root='/server/clean',degraded_root='/server/deg'),root/'annotations')
            e1=read_json(result['E1_clean_only'])['data_list'];e2=read_json(result['E2_clean_and_degraded_balanced'])['data_list']
            self.assertEqual({r['source_image_id'] for r in e1},{r['source_image_id'] for r in e2})
            self.assertTrue(all(r['condition']=='clean' for r in e1));self.assertEqual(len(e2),2*len(e1))
            self.assertNotIn('c',{r['source_image_id'] for r in e2})

    def test_model_tasks_contain_no_gt_or_roi(self):
        with tmp() as d:
            p=Path(d)/'m.jsonl';write_jsonl(p,[row()])
            tasks=list(tasks_for(p,dict(clean_root='/clean',degraded_root='/deg')))
            self.assertEqual(len(tasks),2)
            self.assertTrue(all(not any('gt' in k or 'roi' in k for k in t) for t in tasks))

    def test_resume_repairs_final_fragment_only(self):
        with tmp() as d:
            p=Path(d)/'p.jsonl';p.write_bytes(b'{"sample_id":"a"}\n{"sample_id":')
            self.assertEqual(resume_jsonl(p),[dict(sample_id='a')])
            self.assertEqual(p.read_bytes(),b'{"sample_id":"a"}\n')
            self.assertEqual(len(list(Path(d).glob('*.interrupted_tail'))),1)

    def test_resume_rejects_corrupt_middle(self):
        with tmp() as d:
            p=Path(d)/'p.jsonl';p.write_bytes(b'bad\n{}\n')
            with self.assertRaises(RuntimeError):resume_jsonl(p)

    def test_completed_stage_resumes_without_reexecuting(self):
        with tmp() as d:
            root=Path(d);artifact=root/'result.txt';calls=[]
            def action():artifact.write_text('complete');calls.append(1);return dict(ok=True),[artifact]
            state=StageState(root/'state',dict(code='a'))
            state.execute('E1',dict(checkpoint='x'),action);state.execute('E1',dict(checkpoint='x'),action)
            self.assertEqual(calls,[1])
            artifact.write_text('changed')
            with self.assertRaises(RuntimeError):state.execute('E1',dict(checkpoint='x'),action)

    def test_stage_changed_inputs_or_run_identity_rejected(self):
        with tmp() as d:
            state=StageState(Path(d),dict(code='a'))
            state.execute('x',dict(data='a'),lambda:({},[]))
            with self.assertRaises(RuntimeError):state.execute('x',dict(data='b'),lambda:({},[]))
            with self.assertRaises(RuntimeError):StageState(Path(d),dict(code='b'))

    def test_failed_stage_can_retry(self):
        with tmp() as d:
            state=StageState(Path(d),dict(code='a'))
            def fail():raise RuntimeError('injected interruption')
            with self.assertRaises(RuntimeError):state.execute('x',{},fail)
            self.assertEqual(state.execute('x',{},lambda:({'resumed':True},[])),dict(resumed=True))

    def test_optimizer_budget_counts_updates_not_micro_iterations(self):
        cfg=read_json(REPO/'configs/phase3_h200_normal.json')
        s=training_schedule(cfg['training'])
        self.assertEqual(s['max_iters']//s['accumulation'],3000)
        self.assertEqual(s['max_iters']*cfg['training']['micro_batch_size'],12000)
        self.assertEqual(s['val_interval']//s['accumulation'],1000)

    def test_config_validation(self):
        cfg=read_json(REPO/'configs/phase3_h200_normal.json');validate_config(cfg)
        cfg['training']['effective_batch_size']=3
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_vendor_evaluator_has_exact_verified_phase2_hash(self):
        locks=read_json(REPO/'configs/verified_phase1_phase2_sources.json')
        self.assertEqual(sha256(REPO/'src/monkey_textdet/vendor/evaluate_shared.py'),locks['evaluator_sha256'])


if __name__=='__main__':unittest.main()
