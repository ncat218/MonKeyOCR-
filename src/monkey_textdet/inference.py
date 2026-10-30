"""Resumable global or global+2x2-view inference, sequential on a single allocated MIG GPU."""
import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

from .common import canonical,identity,jsonl,locked_json,read_json,resume_jsonl,save_json,sha256,stop_signals
from .tiling import tile_boxes,translate_polygon,seam_eligible,merge_views


def tasks_for(manifest,locations):
    seen=set()
    for row in jsonl(manifest):
        source=row['source_image_id']
        conditions=['degraded'] if source in seen else ['clean','degraded']
        seen.add(source)
        for condition in conditions:
            yield dict(sample_id='clean::'+source if condition=='clean' else 'deg::'+row['variant_id'],
                       source_image_id=source,condition=condition,
                       path=str(Path(locations[condition+'_root'])/row[condition+'_relpath']),
                       image_sha256=row[condition+'_sha256'],size=row[condition+'_size'])


def generate_inference_config(official_config,backbone_root,output):
    from mmengine.config import Config
    cfg=Config.fromfile(str(official_config))
    cfg.model.backbone.model_dir=str(backbone_root)
    cfg.model.backbone.weights=str(backbone_root/'model.safetensors')
    cfg.dump(str(output))


def run(a):
    import cv2
    import numpy as np
    import torch
    from mmengine.runner import load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmengine.config import Config
    from mmocr.apis import TextDetInferencer
    from mmocr.utils import register_all_modules
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    a.output.mkdir(parents=True,exist_ok=True)
    manifest_hash=sha256(a.manifest);checkpoint_hash=sha256(a.checkpoint)
    experiment=read_json(a.experiment_config)
    fingerprint=dict(manifest_sha256=manifest_hash,checkpoint_sha256=checkpoint_hash,
                     inference_config_sha256=sha256(a.model_config),source_sha256=sha256(__file__),
                     tiling_source_sha256=sha256(Path(__file__).with_name('tiling.py')),
                     torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                     mode=a.mode,model=a.model_label,tiling=experiment['tiling'] if a.mode=='views_2x2' else None)
    locked_json(a.output/'immutable_inference_identity.json',fingerprint)
    output=a.output/('raw_views__global_and_overlapping_2x2_tiles__clean_and_degraded.jsonl' if a.mode=='views_2x2'
                     else 'predictions__global_full_image__clean_and_degraded.jsonl')
    existing=resume_jsonl(output);seen={r['sample_id'] for r in existing}
    if len(seen)!=len(existing):raise ValueError('Duplicate resumed sample IDs')
    expected={t['sample_id'] for t in tasks_for(a.manifest,read_json(a.locations))}
    if not seen<=expected:raise ValueError('Unexpected prediction sample IDs')
    signature=identity(fingerprint)
    assert all(r['inference_identity']==signature for r in existing),'Mixed resumed predictions'
    if seen==expected:
        save_json(a.output/'inference_complete_summary.json',dict(samples=len(seen),output=str(output),sha256=sha256(output),resumed_all=True))
        return
    cfg=Config.fromfile(str(a.model_config))
    if cfg.get('custom_imports'):import_modules_from_strings(**cfg.custom_imports)
    register_all_modules(init_default_scope=True)
    detector=TextDetInferencer(model=str(a.model_config),weights=str(a.checkpoint),device='cuda:0')
    detector.model.eval()
    load_checkpoint(detector.model,str(a.checkpoint),map_location='cpu',strict=True)
    reusable={r['sample_id']:r for r in jsonl(a.reuse_global)} if a.reuse_global else {}
    for key,r in reusable.items():
        if r['checkpoint_sha256']!=checkpoint_hash or r['inference_config_sha256']!=fingerprint['inference_config_sha256']:
            raise ValueError('Cached global predictions use a different checkpoint/config')
    stop=[False]
    def interrupted(signum,frame):stop[0]=True
    for sig in stop_signals():signal.signal(sig,interrupted)
    def predict(image):
        torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            raw=detector(image,batch_size=1,return_vis=False,save_vis=False,save_pred=False,show=False)['predictions'][0]
        torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
        polygons=[[float(v) for v in p] for p in raw['polygons']]
        scores=[float(s) for s in raw['scores']]
        assert len(polygons)==len(scores)
        assert all(len(p)>=6 and len(p)%2==0 and all(math.isfinite(v) for v in p) for p in polygons)
        assert all(math.isfinite(s) and 0<=s<=1 for s in scores)
        return dict(polygons=polygons,scores=scores),ms
    todo=(t for t in tasks_for(a.manifest,read_json(a.locations)) if t['sample_id'] not in seen)
    started=time.perf_counter();new=0
    torch.cuda.reset_peak_memory_stats()
    with output.open('ab') as sink:
        for task in todo:
            if stop[0]:raise InterruptedError('Inference interrupted between images; partial JSONL remains resumable')
            if sha256(task['path'])!=task['image_sha256']:raise ValueError('Input image bytes changed: '+task['sample_id'])
            sample_start=time.perf_counter();global_ms=0.;tile_ms=0.
            if task['sample_id'] in reusable:
                reused=reusable[task['sample_id']]
                assert reused['image_sha256']==task['image_sha256']
                global_pred={k:reused[k] for k in ('polygons','scores')}
                global_ms=reused['inference_ms_with_decode_and_postprocess']
            else:
                global_pred,global_ms=predict(task['path'])
            row={k:v for k,v in task.items() if k not in ('path','size')}
            row.update(model=a.model_label,manifest_sha256=manifest_hash,checkpoint_sha256=checkpoint_hash,
                       inference_config_sha256=fingerprint['inference_config_sha256'],inference_identity=signature)
            if a.mode=='global':row.update(global_pred)
            else:
                image=cv2.imread(task['path'],cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if image is None:raise ValueError('Image decode failed: '+task['path'])
                h,w=image.shape[:2];assert [w,h]==task['size']
                views=[dict(view='global',eligible=[True]*len(global_pred['scores']),**global_pred)]
                for index,tile in enumerate(tile_boxes(w,h,experiment['tiling']['overlap_fraction_of_tile'])):
                    x0,y0,x1,y1=tile
                    pred,elapsed=predict(image[y0:y1,x0:x1].copy());tile_ms+=elapsed
                    polys=[translate_polygon(p,x0,y0) for p in pred['polygons']]
                    views.append(dict(view=f'tile_2x2_{index}',tile_box=list(tile),polygons=polys,scores=pred['scores'],
                        eligible=[seam_eligible(p,tile,[w,h],experiment['tiling']['interior_seam_margin_px']) for p in polys]))
                row['views']=views
                row['global_view_reused']=task['sample_id'] in reusable
            processing_ms=(time.perf_counter()-sample_start)*1000
            total_ms=global_ms if a.mode=='global' else processing_ms+(global_ms if task['sample_id'] in reusable else 0.)
            row.update(inference_ms_with_decode_and_postprocess=total_ms,
                       measured_processing_ms_this_launch=processing_ms)
            sink.write(canonical(row));sink.flush();seen.add(task['sample_id']);new+=1
            if new%25==0:print('INFERENCE',a.model_label,len(seen),'/',len(expected),'elapsed_s',round(time.perf_counter()-started),flush=True)
    assert seen==expected,'Incomplete predictions; not scored'
    save_json(a.output/'inference_complete_summary.json',dict(samples=len(seen),new_samples=new,output=str(output),
        sha256=sha256(output),wall_seconds=time.perf_counter()-started,
        peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
        latency_note='Tiled latency sums measured global and tile forwards; reused global timings come from earlier E0 inference.'))


def merge_file(raw_path,output,threshold,label,manifest_hash):
    from .common import write_jsonl
    def merged():
        for row in jsonl(raw_path):
            result={k:v for k,v in row.items() if k not in ('views','model')}
            result.update(merge_views(row['views'],threshold),model=label,manifest_sha256=manifest_hash,
                          polygon_nms_iou=threshold)
            yield result
    write_jsonl(output,merged())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('manifest','locations','model-config','checkpoint','experiment-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-label',required=True)
    p.add_argument('--mode',choices=['global','views_2x2'],default='global')
    p.add_argument('--reuse-global',type=Path)
    run(p.parse_args())
