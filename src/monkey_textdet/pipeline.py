"""All experiment stages, sequential and resumable inside ONE GPU allocation."""
import argparse
import copy
import math
import os
import signal
import sys
from pathlib import Path

from .common import Processes,StageState,identity,jsonl,locked_json,read_json,save_json,sha256,utc_stamp,write_jsonl,stop_signals


def validate_config(cfg):
    from .training import training_schedule
    schedule=training_schedule(cfg['training'])
    if schedule['val_interval']<1 or schedule['val_interval']>schedule['max_iters']:
        raise ValueError('At least one validation pass is required within the training budget')
    if schedule['max_iters']%schedule['val_interval']:
        raise ValueError('optimizer_updates must be divisible by validation_every_optimizer_updates')
    if schedule['checkpoint_interval']<1:raise ValueError('Invalid checkpoint interval')
    if cfg['tiling']['grid']!=[2,2] or not cfg['tiling']['include_global_view']:
        raise ValueError('This suite implements global + overlapping 2x2, preserving a global comparison')
    if not 0<=cfg['tiling']['overlap_fraction_of_tile']<1:raise ValueError('Invalid tile overlap')
    if not cfg['tiling']['polygon_nms_iou_candidates'] or any(not 0<x<=1 for x in cfg['tiling']['polygon_nms_iou_candidates']):
        raise ValueError('Invalid polygon NMS candidates')
    if cfg['evaluation']['bootstrap_repetitions']<1:raise ValueError('Bootstrap repetitions must be positive')
    if cfg['training']['selection_metric']!='balanced_clean_degraded_hmean':raise ValueError('Unsupported checkpoint selection')
    if set(cfg['split'])!={'train','validation','test'} or not math.isclose(sum(cfg['split'].values()),1.) or min(cfg['split'].values())<=0:
        raise ValueError('Train/validation/test split fractions must be positive and sum to one')
    return schedule


def code_fingerprint(repo):
    files=sorted((repo/'src').rglob('*.py'))+[repo/'run_all.py',repo/'constraints-gpu.txt',repo/'configs/verified_phase1_phase2_sources.json']
    return identity({str(p.relative_to(repo)):sha256(p) for p in files})


def run(a):
    from .data import discover_inputs,build_shared,freeze_splits,export_training_annotations
    from .inference import generate_inference_config,merge_file
    from .training import make_training_config
    from .reporting import make_report
    from .vendor.evaluate_shared import evaluate
    config=read_json(a.config);validate_config(config)
    locks=read_json(a.repo/'configs/verified_phase1_phase2_sources.json')
    root=a.workspace/config['run_name'];inputs=root/'inputs';outputs=root/'outputs'
    for p in (inputs,outputs):p.mkdir(parents=True,exist_ok=True)
    process=Processes(a.logs/'experiment_workers')
    for sig in stop_signals():signal.signal(sig,process.request_stop)
    stable_cfg=copy.deepcopy(config)
    # Data locations are machine-local bindings; their verified content hashes bind model work.
    for k in ('dataset_parent','workspace_root'):stable_cfg.pop(k,None)
    stable_cfg['data']={k:v for k,v in stable_cfg['data'].items() if k=='require_verified_phase2_manifest'}
    fingerprint=code_fingerprint(a.repo)
    state=StageState(root/'state',dict(configuration=stable_cfg,source_fingerprint=fingerprint))
    save_json(root/'run_configuration__explicit_and_resolved_location_overrides.json',dict(configuration=config,dataset_parent=a.dataset_parent))
    resolved=discover_inputs(a.dataset_parent,config['data'],exclude=[a.workspace,a.repo])
    discovery_path=inputs/'01_dataset_discovery__actual_server_locations.json';save_json(discovery_path,resolved)
    print('DATASET LOCATIONS',resolved,flush=True)
    shared_dir=inputs/'02_shared_master_manifest__textvqa_clean_degraded_textocr_word_gt'
    def audit_action():
        result=build_shared(resolved,shared_dir,locks,config['data']['require_verified_phase2_manifest'])
        artifacts=[p for p in shared_dir.iterdir() if p.is_file() and p.name!='dataset_locations.json']
        return result,artifacts
    shared=state.execute('01_verify_and_freeze_existing_10k_dataset_intersection',
        dict(source=fingerprint,annotations={k:sha256(resolved[k]) for k in ('clean_json','textocr_json','main_image_manifest')}),audit_action)
    # Rebind paths every launch, without changing portable manifest bytes.
    locations=dict(clean_root=resolved['clean_images_root'],degraded_root=resolved['degraded_root'])
    location_file=shared_dir/'dataset_locations.json';save_json(location_file,locations)
    split_label='_'.join(f'{name}{100*config["split"][name]:g}'.replace('.','p') for name in ('train','validation','test'))
    split_dir=inputs/('03_frozen_source_id_splits__'+split_label)
    def split_action():
        result=freeze_splits(shared['manifest'],split_dir,config)
        return result,[p for p in split_dir.iterdir() if p.is_file()]
    splits=state.execute('02_freeze_train_validation_test_source_groups',
        dict(master=shared['manifest_sha256'],fractions=config['split'],seed=config['seed'],tiling=config['tiling']),split_action)
    annotation_dir=inputs/'04_mmocr_training_annotations__E1_clean_E2_clean_degraded'
    def target_action():
        result=export_training_annotations(splits,locations,annotation_dir)
        return result,[Path(p) for p in result.values()]
    annotations=state.execute('03_export_training_targets_and_exact_validation_gt',
        dict(splits={k:sha256(v) for k,v in splits.items()},locations=locations,source=fingerprint),target_action)
    official=a.mmocr_root/'configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py'
    original=a.mmocr_root/'model_weight/dbnet_mkv2vitae_icdar2015.pth'
    backbone=a.mmocr_root/'pretrained/monkeyocrv2_as'
    assert sha256(original)==locks['checkpoint_sha256'] and sha256(official)==locks['official_config_sha256']
    model_inputs=inputs/'05_model_and_training_configurations__pinned_phase1_detector'
    model_inputs.mkdir(parents=True,exist_ok=True)
    inference_config=model_inputs/'model_config__E0_E1_E2_same_official_architecture_and_native_inference.py'
    generate_inference_config(official,backbone,inference_config)
    train_configs={};train_dirs={}
    for name in ('E1_clean_only','E2_clean_and_degraded_balanced'):
        train_dirs[name]=outputs/(name+'__training_from_original_checkpoint')
        train_configs[name]=model_inputs/f'training_config__{name}__same_optimizer_update_budget_seed{config["seed"]}.py'
        make_training_config(official,backbone,original,annotations,shared_dir/'evaluation_protocol.json',
                             train_dirs[name],config,name,train_configs[name])
    def probe_action():
        result=outputs/'00_early_training_compatibility_probe__no_checkpoint_changes.json'
        process.run([sys.executable,'-m','monkey_textdet.training','--config',train_configs['E1_clean_only'],
            '--initial-checkpoint',original,'--work-dir',outputs/'00_probe_workdir',
            '--experiment','compatibility_probe_only','--probe-only','--result',result],
            '00_early_mmocr_training_forward_backward_compatibility',cwd=a.mmocr_root)
        return read_json(result),[result]
    state.execute('04_early_training_forward_backward_compatibility_before_expensive_inference',
        dict(config=sha256(train_configs['E1_clean_only']),checkpoint=sha256(original),source=fingerprint),probe_action)

    def infer(stage,manifest,checkpoint,label,mode='global',reuse=None):
        out=outputs/(label+'__'+Path(manifest).stem)
        def action():
            cmd=[sys.executable,'-m','monkey_textdet.inference','--manifest',manifest,'--locations',location_file,
                '--model-config',inference_config,'--checkpoint',checkpoint,'--experiment-config',a.config,
                '--output',out,'--model-label',label,'--mode',mode]
            if reuse:cmd+=['--reuse-global',reuse]
            process.run(cmd,label+'__'+stage,cwd=a.mmocr_root)
            summary=out/'inference_complete_summary.json';info=read_json(summary)
            return dict(predictions=info['output'],directory=str(out)),[summary,Path(info['output']),out/'immutable_inference_identity.json']
        return state.execute(stage,dict(manifest=sha256(manifest),checkpoint=sha256(checkpoint),
            config=sha256(inference_config),mode=mode,label=label,source=fingerprint),action)

    def score(stage,manifest,predictions,label,bootstrap=None):
        out=outputs/'metrics'/label
        def action():
            evaluate(manifest,predictions,shared_dir/'evaluation_protocol.json',out,
                     config['evaluation']['bootstrap_repetitions'] if bootstrap is None else bootstrap)
            return str(out),[p for p in out.iterdir() if p.is_file()]
        return state.execute(stage,dict(manifest=sha256(manifest),predictions=sha256(predictions),
            protocol=sha256(shared_dir/'evaluation_protocol.json'),source=fingerprint,bootstrap=bootstrap),action)

    baseline={}
    for fold in ('validation','test'):
        baseline[fold]=infer('05_E0_global_'+fold,splits[fold],original,'E0_original_checkpoint__global')
    experiments={}
    experiments['E0_original_checkpoint__global']=score('06_score_E0_global_test',splits['test'],
        baseline['test']['predictions'],'E0_original_checkpoint__global__test')
    dev_views=infer('07_E0_overlapping_2x2_development_validation_only',splits['tiling_development'],original,
                    'E0_original_checkpoint__global_plus_2x2','views_2x2',baseline['validation']['predictions'])
    selection_path=outputs/'tiling_selection__polygon_nms_chosen_on_validation_subset_only.json'
    def select_action():
        candidates=[];artifacts=[]
        for threshold in config['tiling']['polygon_nms_iou_candidates']:
            label='development_only__E0_2x2_polygon_nms_'+str(threshold).replace('.','p')
            pred=outputs/'tiling_development_candidates'/('predictions__'+label+'__clean_and_degraded.jsonl')
            merge_file(dev_views['predictions'],pred,threshold,label,sha256(splits['tiling_development']))
            folder=outputs/'tiling_development_candidates'/label
            metrics=evaluate(splits['tiling_development'],pred,shared_dir/'evaluation_protocol.json',folder,0)
            balanced=.5*(metrics['clean_unique']['hmean']+metrics['degraded_instance']['hmean'])
            candidates.append(dict(polygon_nms_iou=threshold,balanced_validation_hmean=balanced,
                clean_hmean=metrics['clean_unique']['hmean'],degraded_hmean=metrics['degraded_instance']['hmean']))
            artifacts.extend([pred,folder/'metrics.json'])
        # Stable tie break prefers the first declared candidate, never test outcomes.
        best=max(candidates,key=lambda c:c['balanced_validation_hmean'])
        result=dict(selected=best,candidates=candidates,split='tiling_development_subset_of_validation_only',
                    selection='mean(clean Hmean,degraded Hmean)',tiling=config['tiling'])
        save_json(selection_path,result)
        return result,artifacts+[selection_path]
    selected=state.execute('08_select_polygon_nms_on_validation_only',
        dict(views=sha256(dev_views['predictions']),candidates=config['tiling']['polygon_nms_iou_candidates'],source=fingerprint),select_action)
    tiled=infer('09_E0_overlapping_2x2_test',splits['test'],original,'E0_original_checkpoint__global_plus_2x2',
                'views_2x2',baseline['test']['predictions'])
    tiled_predictions=Path(tiled['directory'])/'predictions__E0_global_plus_overlapping_2x2__selected_validation_polygon_nms__test_clean_and_degraded.jsonl'
    def merge_action():
        merge_file(tiled['predictions'],tiled_predictions,selected['selected']['polygon_nms_iou'],
                   'E0_original_checkpoint__global_plus_2x2',sha256(splits['test']))
        return str(tiled_predictions),[tiled_predictions]
    state.execute('10_merge_fixed_2x2_test_predictions',dict(views=sha256(tiled['predictions']),selection=selected,source=fingerprint),merge_action)
    experiments['E0_original_checkpoint__global_plus_2x2']=score('11_score_E0_2x2_test',splits['test'],tiled_predictions,
        'E0_original_checkpoint__global_plus_2x2__test')
    for name in ('E1_clean_only','E2_clean_and_degraded_balanced'):
        result_path=train_dirs[name]/f'training_result__{name}__best_validation_checkpoint.json'
        def train_action(name=name,result_path=result_path):
            process.run([sys.executable,'-m','monkey_textdet.training','--config',train_configs[name],
                '--initial-checkpoint',original,'--work-dir',train_dirs[name],'--experiment',name,'--result',result_path],
                name+'__fine_tune_and_validate_same_budget',cwd=a.mmocr_root)
            result=read_json(result_path)
            import torch
            selected_checkpoint=Path(result['best_checkpoint'])
            content=torch.load(selected_checkpoint,map_location='cpu',weights_only=False)
            exported=outputs/'trained_models'/f'detector_weights__{name}__best_balanced_validation_hmean__seed{config["seed"]}.pth'
            exported.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(state_dict=content['state_dict'],meta=dict(experiment=name,training_result=result,
                source_split_sha256=sha256(splits['train']),evaluation_protocol_sha256=sha256(shared_dir/'evaluation_protocol.json'))),exported)
            result.update(exported_detector=str(exported),exported_detector_sha256=sha256(exported))
            save_json(result_path,result)
            return result,[result_path,selected_checkpoint,exported]
        trained=state.execute('12_train_'+name,dict(config=sha256(train_configs[name]),initial=sha256(original),source=fingerprint),train_action)
        prediction=infer('13_test_'+name,splits['test'],trained['exported_detector'],name+'__global')
        experiments[name+'__global']=score('14_score_'+name,splits['test'],prediction['predictions'],name+'__global__test')
    report_dir=outputs/'FINAL_COMPARISON__E0_global_E0_2x2_E1_clean_E2_clean_degraded'
    def report_action():
        report,csv_file=make_report(experiments,report_dir,config,split_dir/'split_policy_and_counts.json',selection_path)
        return dict(report=str(report),csv=str(csv_file)),list(report_dir.iterdir())
    final=state.execute('15_final_same_source_test_comparison',
        dict(metrics={k:sha256(Path(v)/'metrics.json') for k,v in experiments.items()},source=fingerprint),report_action)
    print('\nALL EXPERIMENTS COMPLETED\nREPORT:',final['report'],'\nCSV:',final['csv'],
          '\nMODELS:',outputs/'trained_models','\nLOGS:',root/'logs',flush=True)
    save_json(root/'COMPLETE__all_phase3_experiments_finished.json',dict(final=final,completed_utc=utc_stamp(),
        run_name=config['run_name'],experiments=list(experiments),source_fingerprint=fingerprint))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('repo','config','workspace','mmocr-root','logs'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dataset-parent',default='')
    run(p.parse_args())
