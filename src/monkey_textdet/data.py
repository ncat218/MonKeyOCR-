"""Discover relocated inputs, freeze the existing paired set, split by source bytes, export targets."""
import collections
import getpass
import math
import os
import random
from pathlib import Path

from .common import jsonl, read_json, save_json, sha256, write_jsonl

PRUNE_NAMES={'.git','.venv','venv','env_py311','__pycache__','node_modules','train_images','val_images',
             'images','huggingface_cache','python_installations','pip_cache','uv_cache','mmcv_src'}


def discover_inputs(parent, overrides, exclude=()):
    roots=[Path(parent).expanduser().resolve()] if parent else [Path('/datastore')/getpass.getuser(),Path('/datastore/shared')]
    roots=[p for p in roots if p.is_dir()]
    if not roots:
        raise FileNotFoundError('Set DATASET_PARENT in run_all.py to the parent containing the three uploaded datasets.')
    blocked={Path(p).resolve() for p in exclude}
    found={'clean_json':set(),'textocr_json':set(),'main_image_manifest':set()}
    subdirectories=overrides.get('dataset_subdirectories',{})
    excluded=set(overrides.get('excluded_dataset_directories',[]))
    filenames=dict(clean_json='TextVQA_0.5.1_train.json',textocr_json='TextOCR_0.1_train.json',
                   main_image_manifest='image_manifest.csv')
    searched={}
    for name in found:
        if overrides.get(name):
            searched[name]=[]
            continue
        relative=Path(subdirectories.get(name,''))
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Dataset subdirectories must be relative to DATASET_PARENT; use file overrides for absolute paths')
        scopes=[root/relative for root in roots]
        searched[name]=[str(p) for p in scopes]
        # A missing configured 10K folder must never fall back to the older degraded dataset.
        for scope in scopes:
            for folder,dirs,files in os.walk(scope):
                base=Path(folder)
                dirs[:]=[d for d in dirs if d not in PRUNE_NAMES and d not in excluded
                         and not d.startswith('_workspace') and (base/d).resolve() not in blocked]
                filename=filenames[name]
                if filename in files and (name!='main_image_manifest' or base.name=='main'):
                    found[name].add((base/filename).resolve())
    result={}
    for name,paths in found.items():
        choices=[Path(overrides[name]).expanduser().resolve()] if overrides.get(name) else sorted(paths)
        if len(choices)!=1 or not choices[0].is_file():
            raise RuntimeError(f'{name}: expected one real input; searched={searched[name]}; candidates={[str(p) for p in choices]}. '
                               'Use the corresponding data override if multiple dataset versions exist.')
        result[name]=str(choices[0])
    result['clean_images_root']=str(Path(overrides['clean_images_root']).expanduser().resolve()) if overrides.get('clean_images_root') else str(Path(result['clean_json']).parent)
    result['degraded_root']=str(Path(overrides['degraded_root']).expanduser().resolve()) if overrides.get('degraded_root') else str(Path(result['main_image_manifest']).parent.parent)
    result['searched_parents']=[str(p) for p in roots]
    result['searched_dataset_subdirectories']=searched
    result['excluded_dataset_directories']=sorted(excluded)
    return result


def build_shared(paths,output,locks,require_verified=True):
    from .vendor import shared_manifest as shared
    # All three files are explicit. Limit the legacy discovery fallback scan to the degraded dataset.
    cfg=dict(input_root=paths['degraded_root'],output_root=str(output),overlay_sources=12)
    cfg.update({k:paths[k] for k in ('clean_json','textocr_json','main_image_manifest','clean_images_root','degraded_root')})
    result=shared.build_shared_manifest(cfg)
    if require_verified and result['manifest_sha256']!=locks['phase2_manifest_sha256']:
        raise RuntimeError('Rebuilt master manifest differs from verified Phase 2. Inspect dataset versions/audit; '
                           'do not silently benchmark another intersection or disable the lock to force completion.')
    assert sha256(output/'evaluation_protocol.json')==locks['evaluation_protocol_sha256']
    assert sha256(Path(__file__).parent/'vendor/evaluate_shared.py')==locks['evaluator_sha256']
    return result


def split_source_groups(rows,fractions,seed):
    """Identical source bytes cannot cross folds, even if represented by multiple IDs."""
    if set(fractions)!={'train','validation','test'} or not math.isclose(sum(fractions.values()),1.):
        raise ValueError('Train/validation/test fractions must sum to one')
    if any(v<=0 for v in fractions.values()):raise ValueError('Every split must be nonempty')
    by_source={}
    for row in rows:
        key=row['source_image_id']
        if key in by_source and by_source[key]!=row['clean_sha256']:
            raise ValueError('One source ID points to multiple source hashes')
        by_source[key]=row['clean_sha256']
    groups=collections.defaultdict(list)
    for source,digest in by_source.items():groups[digest].append(source)
    keys=sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys)<3:raise ValueError('At least three source-byte groups required to split')
    # Round by groups; with the verified source-byte dedup this is also by images.
    train=max(1,min(len(keys)-2,int(len(keys)*fractions['train'])))
    val=max(1,min(len(keys)-train-1,int(len(keys)*fractions['validation'])))
    ranges={'train':keys[:train],'validation':keys[train:train+val],'test':keys[train+val:]}
    return {fold:sorted(s for h in hs for s in groups[h]) for fold,hs in ranges.items()}


def freeze_splits(master,output,config):
    rows=list(jsonl(master));folds=split_source_groups(rows,config['split'],config['seed'])
    result={}
    for fold,ids in folds.items():
        path=output/f'manifest__{fold}__textvqa_clean_degraded_same_textocr_word_gt.jsonl'
        chosen=set(ids)
        write_jsonl(path,(r for r in rows if r['source_image_id'] in chosen))
        result[fold]=str(path)
    # Development subset is drawn only from validation and never from test.
    dev=list(folds['validation']);random.Random(config['seed']+1).shuffle(dev)
    dev=sorted(dev[:config['tiling']['development_sources_max']])
    path=output/'manifest__tiling_parameter_development_subset_of_validation_only.jsonl'
    write_jsonl(path,(r for r in rows if r['source_image_id'] in set(dev)))
    result['tiling_development']=str(path)
    save_json(output/'source_id_split_membership__train_validation_test.json',folds)
    save_json(output/'split_policy_and_counts.json',dict(seed=config['seed'],fractions=config['split'],
        sources={k:len(v) for k,v in folds.items()},tiling_development_sources=len(dev),
        grouping='source_image_id and clean source bytes SHA256',
        disclosure='Whole master set baseline was previously inspected in Phase 2. '
                   'These sources are held out from Phase 3 optimization, not a historically unseen test.',
        manifest_sha256={k:sha256(v) for k,v in result.items()}))
    return result


def polygon_parts(geometry):
    if geometry.geom_type=='Polygon':return [geometry]
    return [p for g in getattr(geometry,'geoms',[]) for p in polygon_parts(g)]


def simple_pieces(geometry):
    """Triangulate only polygons with holes: training ignore masks must not fill the ROI hole."""
    from shapely.ops import triangulate
    for poly in polygon_parts(geometry):
        if not poly.interiors:
            yield poly
        else:
            for triangle in triangulate(poly):
                clipped=triangle.intersection(poly)
                for piece in polygon_parts(clipped):
                    if piece.area>1e-8:
                        if piece.interiors:raise ValueError('Ignore mask decomposition still has holes')
                        yield piece


def instances_for(row,condition,audit):
    from shapely.geometry import shape,box
    result=[]
    def instance(poly,ignored):
        coords=[float(v) for point in list(poly.exterior.coords)[:-1] for v in point]
        if len(coords)<6:return
        result.append(dict(polygon=coords,bbox=[float(v) for v in poly.bounds],bbox_label=0,ignore=bool(ignored)))
    for category in ('valid','ignored'):
        for ann in row['gt_'+condition][category]:
            geom=shape(ann['geometry'])
            # MMOCR takes a single polygon per instance. This affects training targets only.
            if category=='valid' and (geom.geom_type!='Polygon' or any(p.interiors for p in polygon_parts(geom))):
                audit['training_valid_convex_hull_approximations']+=1
                geom=geom.convex_hull
            for piece in simple_pieces(geom):instance(piece,category=='ignored')
    outside=box(0,0,*row[condition+'_size']).difference(shape(row['roi_'+condition]))
    for poly in simple_pieces(outside):
        instance(poly,True);audit['outside_common_roi_training_ignore_pieces']+=1
    return result


def export_training_annotations(splits,locations,output):
    """Training targets may be adapted to MMOCR; evaluation GT remains the frozen exact geometry."""
    output.mkdir(parents=True,exist_ok=True)
    audit=collections.Counter();result={}
    for fold in ('train','validation'):
        rows=list(jsonl(splits[fold]))
        if len({r['source_image_id'] for r in rows})!=len(rows):
            raise ValueError('Phase 3 currently requires one MAIN degraded variant per source for balanced E2 sampling')
        condition_lists={c:[] for c in ('clean','degraded')};exact={}
        for row in rows:
            for cond in ('clean','degraded'):
                key='clean::'+row['source_image_id'] if cond=='clean' else 'deg::'+row['variant_id']
                w,h=row[cond+'_size']
                record=dict(img_path=str(Path(locations[cond+'_root'])/row[cond+'_relpath']),width=w,height=h,
                            sample_id=key,source_image_id=row['source_image_id'],condition=cond,
                            instances=instances_for(row,cond,audit))
                condition_lists[cond].append(record)
                exact[key]=dict(gt=row['gt_'+cond],roi=row['roi_'+cond],condition=cond,source=row['source_image_id'])
        choices={'E1_clean_only':condition_lists['clean'],'E2_clean_and_degraded_balanced':condition_lists['clean']+condition_lists['degraded']} if fold=='train' else {'validation_clean_and_degraded':condition_lists['clean']+condition_lists['degraded']}
        for label,records in choices.items():
            path=output/f'annotations__{fold}__{label}__mmocr_polygon_targets.json'
            save_json(path,dict(metainfo=dict(dataset_type='TextDetDataset',task_name='textdet',category=[dict(id=0,name='text')]),data_list=records))
            result[label]=str(path)
        if fold=='validation':
            path=output/'exact_evaluation_gt__validation_clean_and_degraded__same_phase2_protocol.json'
            save_json(path,exact);result['validation_exact']=str(path)
    path=output/'training_target_conversion_audit__evaluation_gt_unchanged.json'
    save_json(path,dict(counts=dict(audit),policy='Convex hull only for fragmented valid training polygons; '
        'ignored masks and outside-ROI masks decomposed without filling holes. Exact evaluation GT unchanged.',
        E2_balance='Equal clean/degraded entries, same source IDs as E1. Infinite shuffled sampling; same optimizer-update budget.'))
    result['audit']=str(path)
    return result
