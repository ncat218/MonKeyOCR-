"""CPU miniature end-to-end controller test with explicit fake GPU workers, never real results."""
import argparse
import contextlib
import copy
import csv
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from test_core import REPO,tmp
from monkey_textdet.common import canonical,jsonl,read_json,save_json,sha256,write_jsonl
from monkey_textdet import pipeline
from monkey_textdet.vendor import shared_manifest as sm


def miniature_dataset(parent):
    clean=parent/'textvqa_clean';deg=parent/'textvqa_degraded_10k';gt=parent/'textocr_0_1_train_json'
    obsolete=parent/'textvqa_degraded/main/image_manifest.csv'
    obsolete.parent.mkdir(parents=True);obsolete.write_text('obsolete dataset must not be selected')
    records=[];selection=[];imgs={};anns={};mapping={}
    for i in range(12):
        source=f'fixture_{i:02d}';cp=clean/'train_images'/f'{source}.png';dp=deg/'main/images/realistic_mix/L2'/f'{source}.png'
        for path,color in [(cp,(100+i,80,70)),(dp,(70+i,50,40))]:
            path.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(100,80),color).save(path)
        selection.append(dict(_image_id=source,_source_image_sha256=sha256(cp)))
        imgs[source]=dict(id=source,width=100,height=80)
        ann_id=source+'_1';anns[ann_id]=dict(id=ann_id,image_id=source,points=[20,20,40,20,40,30,20,30],utf8_string='TEST')
        mapping[source]=[ann_id]
        records.append(dict(image_id=source,condition='realistic_mix',level='L2',
            params=json.dumps([dict(operation='jpeg_compression')]),source_image_path='/old/clean/'+cp.name,
            output_path='/old/generated/main/images/realistic_mix/L2/'+dp.name,
            source_image_sha256=sha256(cp),output_sha256=sha256(dp),severity_basis='image_relative_fallback'))
    save_json(clean/'TextVQA_0.5.1_train.json',dict(data=[dict(image_id=r['image_id']) for r in records]))
    save_json(gt/'TextOCR_0.1_train.json',dict(imgs=imgs,anns=anns,imgToAnns=mapping))
    sel=deg/'metadata/locked_eval_manifest.csv';sel.parent.mkdir(parents=True)
    with sel.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selection[0]));w.writeheader();w.writerows(selection)
    save_json(deg/'metadata/split_policy.json',dict(protocol=sm.GENERATOR_PROTOCOL,eval_unique_images=12))
    frozen=deg/'config/frozen_degradation_config.json'
    save_json(frozen,dict(protocol=sm.GENERATOR_PROTOCOL,eval_manifest_sha256=sha256(sel),
                         train_annotation_sha256=sha256(clean/'TextVQA_0.5.1_train.json')))
    for r in records:r.update(config_sha256=sha256(frozen),source_manifest_sha256=sha256(sel))
    with (deg/'main/image_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


class PipelineIntegration(unittest.TestCase):
    def test_whole_suite_runs_then_resumes_without_repeating_gpu_stages(self):
        with tmp() as d:
            root=Path(d);data=root/'uploaded datasets';miniature_dataset(data)
            repo=root/'repo';(repo/'configs').mkdir(parents=True);(repo/'src').mkdir()
            for filename in ['run_all.py','constraints-gpu.txt']:(repo/filename).write_text('fixture')
            mmocr=root/'mmocr';official=mmocr/'configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py'
            official.parent.mkdir(parents=True);official.write_text('fixture config')
            ckpt=mmocr/'model_weight/dbnet_mkv2vitae_icdar2015.pth';ckpt.parent.mkdir();ckpt.write_bytes(b'original detector')
            locks=read_json(REPO/'configs/verified_phase1_phase2_sources.json')
            locks.update(checkpoint_sha256=sha256(ckpt),official_config_sha256=sha256(official))
            save_json(repo/'configs/verified_phase1_phase2_sources.json',locks)
            save_json(repo/'configs/verified_phase2_manifest_variants.json',
                      read_json(REPO/'configs/verified_phase2_manifest_variants.json'))
            cfg=read_json(REPO/'configs/phase3_h200_normal.json');cfg['data']['require_verified_phase2_manifest']=False
            cfg['evaluation']['bootstrap_repetitions']=10
            cfg_path=repo/'configs/suite.json';save_json(cfg_path,cfg)
            args=argparse.Namespace(repo=repo,config=cfg_path,workspace=root/'managed',mmocr_root=mmocr,logs=root/'logs',dataset_parent=str(data))
            calls=[];training_initializations=[]
            def fake_config(official,backbone,output):Path(output).write_text('same model architecture')
            def fake_train_config(official,backbone,original,annotations,protocol,work,cfg,name,output):
                save_json(output,dict(experiment=name,initial=str(original),annotations=annotations))
            def fake_worker(self,command,label,cwd=None,env=None):
                calls.append(label);command=[str(x) for x in command]
                def arg(name):return Path(command[command.index('--'+name)+1])
                if 'monkey_textdet.training' in command:
                    result=arg('result')
                    if '--probe-only' in command:save_json(result,dict(probe_only=True));return
                    training_initializations.append(str(arg('initial-checkpoint')))
                    work=arg('work-dir');work.mkdir(parents=True,exist_ok=True)
                    best=work/'best_fake_validation_checkpoint.pth';best.write_bytes(b'fixture trained weights')
                    save_json(result,dict(best_checkpoint=str(best),experiment=command[command.index('--experiment')+1]))
                elif 'monkey_textdet.inference' in command:
                    out=arg('output');out.mkdir(parents=True,exist_ok=True);manifest=arg('manifest')
                    mode=command[command.index('--mode')+1];label=command[command.index('--model-label')+1]
                    predictions=[]
                    for row in jsonl(manifest):
                        for cond in ('clean','degraded'):
                            poly=list(row['gt_'+cond]['valid'][0]['geometry']['coordinates'][0])[:-1]
                            poly=[v for p in poly for v in p]
                            pred=dict(sample_id='clean::'+row['source_image_id'] if cond=='clean' else 'deg::'+row['variant_id'],
                                source_image_id=row['source_image_id'],condition=cond,model=label,manifest_sha256=sha256(manifest),
                                polygons=[poly],scores=[.9])
                            if mode=='views_2x2':pred['views']=[dict(polygons=[poly,poly],scores=[.9,.8],eligible=[True,True])]
                            predictions.append(pred)
                    path=out/'fake_predictions_for_controller_test_only.jsonl';write_jsonl(path,predictions)
                    save_json(out/'inference_complete_summary.json',dict(output=str(path)))
                    save_json(out/'immutable_inference_identity.json',dict(fixture=True))
                else:raise AssertionError(command)
            fake_torch=types.SimpleNamespace(load=lambda *a,**kw:dict(state_dict={}),save=lambda obj,p:Path(p).write_bytes(b'fake exported weights'))
            with patch('monkey_textdet.inference.generate_inference_config',fake_config),\
                 patch('monkey_textdet.training.make_training_config',fake_train_config),\
                 patch('monkey_textdet.common.Processes.run',fake_worker),patch.dict(sys.modules,{'torch':fake_torch}),\
                 contextlib.redirect_stdout(io.StringIO()):
                pipeline.run(args)
                completed=len(calls)
                pipeline.run(args)
            self.assertEqual(len(calls),completed,'Resuming must not rerun completed GPU stages')
            self.assertEqual(training_initializations,[str(ckpt),str(ckpt)],'E2 must not initialize from E1')
            run_root=args.workspace/cfg['run_name']
            status=read_json(run_root/'COMPLETE__all_phase3_experiments_finished.json')
            self.assertEqual(len(status['experiments']),4)
            selection=read_json(run_root/'outputs/tiling_selection__polygon_nms_chosen_on_validation_subset_only.json')
            self.assertEqual(selection['split'],'tiling_development_subset_of_validation_only')
            self.assertTrue(Path(status['final']['report']).is_file())


if __name__=='__main__':unittest.main()
