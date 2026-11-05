"""Bounded review images and a small result ZIP; no full datasets or model copies."""
import json
import math
import os
import zipfile
from pathlib import Path

from .common import jsonl, read_json, save_json, sha256


def render_review(manifest, locations, predictions, queue, output, max_side=640):
    from PIL import Image, ImageDraw
    from shapely.geometry import Polygon, shape
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    requests = read_json(queue)['words']
    wanted = {r['source_image_id'] for r in requests}
    rows = {r['source_image_id']:r for r in jsonl(manifest) if r['source_image_id'] in wanted}
    predicted = {m:{r['sample_id']:r for r in jsonl(p) if r['source_image_id'] in wanted} for m,p in predictions.items()}
    evidence = []
    for index,request in enumerate(requests,1):
        row = rows[request['source_image_id']]
        panels, metadata = [], []
        for condition in ('clean','degraded'):
            path = Path(locations[condition+'_root'])/row[condition+'_relpath']
            if sha256(path) != row[condition+'_sha256']:
                raise ValueError('Review image bytes changed: '+str(path))
            entries = row['gt_'+condition]
            ann = next(g for g in entries['valid'] if g['ann_id'] == request['ann_id'])
            x0,y0,x1,y1 = shape(ann['geometry']).bounds
            margin = max(32.,1.5*(y1-y0))
            w,h = row[condition+'_size']
            crop_box = (max(0,math.floor(x0-margin)),max(0,math.floor(y0-margin)),
                        min(w,math.ceil(x1+margin)),min(h,math.ceil(y1+margin)))
            with Image.open(path) as original:
                if original.size != (w,h):
                    raise ValueError('Review image dimensions differ from frozen manifest')
                crop = original.convert('RGB').crop(crop_box)
            draw = ImageDraw.Draw(crop)
            def outline(geometry,color,width):
                if geometry.is_empty:
                    return
                if geometry.geom_type == 'Polygon':
                    points = [(x-crop_box[0],y-crop_box[1]) for x,y in geometry.exterior.coords]
                    draw.line(points,fill=color,width=width)
                elif hasattr(geometry,'geoms'):
                    for part in geometry.geoms:
                        outline(part,color,width)
            from shapely.geometry import box
            window = box(*crop_box)
            for g in entries['ignored']:
                outline(shape(g['geometry']).intersection(window),'gray',1)
            for g in entries['valid']:
                outline(shape(g['geometry']).intersection(window),'lime',2 if g['ann_id'] == request['ann_id'] else 1)
            key = 'clean::'+row['source_image_id'] if condition == 'clean' else 'deg::'+row['variant_id']
            for model,color in [('E0','cyan'),('Aprime','orange')]:
                for coordinates in predicted[model][key]['polygons']:
                    p = Polygon(list(zip(coordinates[::2],coordinates[1::2])))
                    if not p.is_valid:
                        p = p.buffer(0)
                    outline(p.intersection(window),color,1)
            scale = min(1.,max_side/max(crop.size))
            before = crop.size
            if scale < 1:
                crop = crop.resize(tuple(max(1,round(v*scale)) for v in crop.size),Image.Resampling.LANCZOS)
            panel = Image.new('RGB',(max(330,crop.width),crop.height+42),'white')
            panel.paste(crop,(0,42))
            label = ImageDraw.Draw(panel)
            label.text((4,3),condition+' | GT=green E0=cyan A-prime=orange',fill='black')
            label.text((4,20),f'native crop {before[0]}x{before[1]}; display scale {scale:.3f}',fill='black')
            panels.append(panel)
            metadata.append(dict(condition=condition,image_sha256=sha256(path),crop_box=list(crop_box),display_scale=scale))
        combined = Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),'white')
        combined.paste(panels[0],(0,0));combined.paste(panels[1],(panels[0].width,0))
        # IDs may contain punctuation; use ordinal in filename, preserve exact IDs in index.
        filename = f'case_{index:02d}__{request["review_stratum"]}__clean_degraded_GT_E0_Aprime.png'
        combined.save(output/filename,optimize=True)
        evidence.append(dict(**request,panel=filename,source_evidence=metadata,manual_label='',reviewer_notes=''))
    path = output/'REVIEW_INDEX__exact_GT_ID_SHA_crop_bounds_and_manual_labels_pending.json'
    save_json(path,dict(scope='Purposive word-centered examples, native mask. Display-only overlays/resizing; '
        'not new training images, original degradation bytes unchanged. Manual interpretation still pending.',cases=evidence))
    return path


def package(root, marker):
    root, marker = Path(root), Path(marker)
    archive = root/'RESULTS_AND_EDA__FULL_R1_E0_original_vs_Aprime_FP32__native_selected_word_errors_review_crops.zip'
    files = [marker,root/'RUN_CONFIGURATION__E0_original_vs_Aprime_FP32__inference_EDA_only.json',
             root/'READ_ME_FIRST__E0_Aprime_FULL_R1_inputs_outputs_logs.json']
    for directory in ('inputs','outputs/E0_ORIGINAL_CHECKPOINT__global_predictions',
                      'outputs/FINAL_EDA__E0_original_vs_Aprime_FP32__same_FULL_R1_test',
                      'outputs/REVIEW_CROPS__native_GT_E0_Aprime__clean_degraded_pairs'):
        files += [p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.json','.md','.csv','.py','.png')
                  and not p.name.startswith('WORDS__')]
    files = sorted(set(files))
    for p in files:
        p.resolve().relative_to(root.resolve())
        if p.stat().st_size > 8*1024**2:
            raise ValueError('Oversized packet file: '+str(p))
    if sum(p.stat().st_size for p in files) > 64*1024**2:
        raise ValueError('EDA delivery exceeds 64 MiB uncompressed bound')
    inventory = {p.relative_to(root).as_posix():sha256(p) for p in files}
    identity = dict(completion_sha256=sha256(marker),file_sha256=inventory)
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None or json.loads(z.read('DELIVERY_IDENTITY.json')) != identity:
                raise ValueError('Existing delivery archive differs')
        return archive
    temp = archive.with_suffix('.zip.writing')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('DELIVERY_IDENTITY.json',json.dumps(identity))
        for p in files:
            z.write(p,p.relative_to(root).as_posix())
    os.replace(temp,archive)
    return archive
