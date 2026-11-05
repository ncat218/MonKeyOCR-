"""One H200 job: original E0 FULL_R1 inference, verified A′ reuse, paired EDA. No training."""
import math
import re
import signal
import sys
from pathlib import Path

from .baseline_reference import AP, COMPARISON, INFER_CONFIG, SELECTION, bind_reference, reference_files
from .common import Processes, StageState, jsonl, read_json, save_json, sha256, stop_signals, utc_stamp
from .mask_sweep import choose_mask, prediction_file, threshold_tag

SUITE = 'full_r1_original_E0_vs_Aprime_FP32_inference_and_EDA_only'
E0 = 'E0_original_Monkey_checkpoint__no_target_finetuning__FP32_global'
MARKER = 'COMPLETE__E0_original_Aprime_FP32_FULL_R1_inference_and_automatic_EDA_finished.json'


def validate(config):
    if config['suite'] != SUITE or 'training' in config or config.get('precision_control'):
        raise ValueError('This is an inference/EDA-only suite; no training/ASF control settings allowed')
    if config['mask_thresholds_validation_only'] != [.3,.2,.15,.1,.05]:
        raise ValueError('Keep the same five validation candidates as A′ job9793')
    if config['evaluation']['bootstrap_repetitions'] < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    if not isinstance(config['eda']['review_words_max'],int) or not 1 <= config['eda']['review_words_max'] <= 32:
        raise ValueError('Review packet must contain 1–32 word pairs')
    if not 128 <= config['eda']['review_max_side'] <= 640:
        raise ValueError('Review display side must be 128–640 pixels')
    ref = config['reference_job9793']
    if not re.fullmatch('[A-Za-z0-9_-]+',ref['run_name']) or ref['run_name'] == config['run_name']:
        raise ValueError('Use a new run directory, distinct from reference job9793')
    hashes = [ref[k] for k in ('source_fingerprint','checkpoint_sha256','master_sha256','protocol_sha256')]
    hashes += list(ref['small_file_sha256'].values())+list(ref['split_sha256'].values())
    if any(not re.fullmatch('[a-f0-9]{64}',h) for h in hashes):
        raise ValueError('Reference identities must be SHA256 digests')
    if set(ref['split_sha256']) != {'train','validation','test'}:
        raise ValueError('Pin all three frozen detection folds')
    return dict(mode='inference_and_EDA_only',new_forward_passes='E0 only; reuse verified A-prime predictions')


def run(a):
    from mmengine.config import Config
    from .data import discover_inputs
    from .pipeline import code_fingerprint
    from .vendor.evaluate_shared import evaluate
    from .baseline_eda import analyze_native
    from .baseline_reporting import make_report
    from .baseline_delivery import render_review, package

    config = read_json(a.config)
    validate(config)
    root = a.workspace/config['run_name']
    inputs,outputs = root/'inputs',root/'outputs'
    inputs.mkdir(parents=True,exist_ok=True);outputs.mkdir(parents=True,exist_ok=True)
    fingerprint = code_fingerprint(a.repo)
    state = StageState(root/'state',dict(configuration=config,source_fingerprint=fingerprint))
    process = Processes(a.logs/'E0_original_and_Aprime_reused__FULL_R1_inference_EDA_workers')
    for sig in stop_signals():
        signal.signal(sig,process.request_stop)
    paths = discover_inputs(a.dataset_parent,config['data'],exclude=[a.workspace,a.repo])
    locations = dict(clean_root=paths['clean_images_root'],degraded_root=paths['degraded_root'])
    locations = {k:str(v) for k,v in locations.items()}
    location_file = inputs/'CURRENT_DATASET_LOCATIONS__bytes_must_match_frozen_job9793.json'
    save_json(location_file,locations)
    save_json(root/'RUN_CONFIGURATION__E0_original_vs_Aprime_FP32__inference_EDA_only.json',config)
    save_json(root/'READ_ME_FIRST__E0_Aprime_FULL_R1_inputs_outputs_logs.json',dict(
        purpose='E0 original Monkey checkpoint vs A-prime FP32 mixed fine-tuning; inference and exploratory EDA only',
        reference_run=str(a.workspace/config['reference_job9793']['run_name']),
        inputs=str(inputs),outputs=str(outputs),logs=str(root/'logs'),
        new_predictions=str(outputs/'E0_ORIGINAL_CHECKPOINT__global_predictions'),
        report=str(outputs/'FINAL_EDA__E0_original_vs_Aprime_FP32__same_FULL_R1_test'),
        review=str(outputs/'REVIEW_CROPS__native_GT_E0_Aprime__clean_degraded_pairs'),
        completion_marker=str(root/MARKER),
        manual_review='Automated tables and review crops only. Human review and causal map/target diagnosis remain separate.'))
    reference_files(a.workspace,config)
    binding_path = inputs/'REFERENCE__verified_frozen_folds_and_Aprime_raw_predictions__read_only.json'
    def audit():
        result=bind_reference(a.workspace,config,locations,binding_path)
        return result,[binding_path]
    reference=state.execute('01_verify_job9793_frozen_folds_image_bytes_and_Aprime_predictions',
        dict(reference=config['reference_job9793'],locations=locations),audit)
    splits={k:Path(v) for k,v in reference['splits'].items()}
    protocol=Path(reference['protocol'])
    original=a.mmocr_root/'model_weight/dbnet_mkv2vitae_icdar2015.pth'
    locks=read_json(a.repo/'configs/verified_phase1_phase2_sources.json')
    official=a.mmocr_root/'configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py'
    if sha256(original) != locks['checkpoint_sha256'] or sha256(official) != locks['official_config_sha256']:
        raise ValueError('Original Monkey checkpoint/config differs from reproduced baseline')
    model_config=inputs/'INFERENCE_CONFIG__identical_to_Aprime_job9793__only_checkpoint_differs.py'
    model_config.write_bytes(Path(reference['model_config']).read_bytes())
    cfg=Config.fromfile(str(model_config))
    if (cfg.model.backbone.use_bf16 is not False or cfg.model.neck.get('asf_cfg') is not None or
            cfg.get('textdet_parameter_precision') != 'fp32'):
        raise ValueError('Expected exact no-ASF FP32 A′ inference config')
    if Path(cfg.model.backbone.model_dir).resolve() != (a.mmocr_root/'pretrained/monkeyocrv2_as').resolve():
        raise ValueError('Copied job9793 inference config points at a different Monkey encoder directory')
    experiments,selections={},{}
    ref_root=Path(reference['reference_run'])
    historical=read_json(ref_root/COMPARISON)
    selection_a=read_json(ref_root/SELECTION)
    a_mask=selection_a['selected']['mask_threshold']
    selections['Aprime']=dict(native=.3,selected=a_mask,validation_evidence=selection_a)

    def sweep(fold,masks):
        out=outputs/'E0_ORIGINAL_CHECKPOINT__global_predictions'/fold
        mask_file=inputs/('E0__'+fold+'__validation_candidates_or_locked_test_masks.json')
        save_json(mask_file,dict(mask_thresholds=masks))
        def action():
            process.run([sys.executable,'-m','monkey_textdet.mask_sweep','--manifest',splits[fold],
                '--locations',location_file,'--model-config',model_config,'--checkpoint',original,
                '--thresholds',mask_file,'--output',out,'--model-label',E0],
                'E0_ORIGINAL_CHECKPOINT__FP32_global__'+fold+'__one_forward_per_image_multiple_native_masks',cwd=a.mmocr_root)
            signature=read_json(out/'immutable_map_sweep_identity.json')
            if signature.get('runtime_math_flags') != config['reference_job9793']['runtime_math_flags']:
                raise ValueError('E0 math flags differ from reference A′; do not compare different inference regimes')
            return str(out),[out/'immutable_map_sweep_identity.json',out/'COMPLETE__all_requested_mask_predictions.json']+[
                prediction_file(out,t) for t in masks]
        return Path(state.execute('02_E0_original_'+fold+'_FP32_global_inference',dict(
            manifest=sha256(splits[fold]),checkpoint=sha256(original),config=sha256(model_config),masks=masks,
            source=fingerprint,locations=locations),action))

    def score(model,fold,mask,prediction):
        folder=outputs/'METRICS__unchanged_external_TextOCR_word_protocol'/model/fold/threshold_tag(mask)
        def action():
            metric=evaluate(splits[fold],prediction,protocol,folder,
                            config['evaluation']['bootstrap_repetitions'] if fold=='test' else 0)
            metric['limitations']=['Original E0 versus mixed fine-tuned A-prime; external word polygon protocol.',
                'Fixed FULL_R1 detection folds within official TextVQA train. TextOCR train is a listed Monkey pretraining source.',
                'Exploratory test EDA, synthetic mixed degradations, one fine-tuning seed.']
            save_json(folder/'metrics.json',metric)
            (folder/'clean_vs_degraded.md').write_text('# '+model+' '+fold+' evaluation\n\n'
                'See the final E0/A-prime comparison. This is inference only; A-prime was fine-tuned previously.\n',encoding='utf-8')
            if model=='Aprime' and fold=='test':
                suffix='native_mask_0p30' if mask==.3 else 'validation_selected_mask'
                previous=historical['experiments'][AP+'__'+suffix]['metrics']
                for condition in ('clean_unique','degraded_instance'):
                    if metric[condition]['counts'] != previous[condition]['counts'] or not math.isclose(
                            metric[condition]['hmean'],previous[condition]['hmean'],abs_tol=1e-12):
                        raise ValueError('Recomputed A′ metrics do not reproduce job9793: '+condition)
            return str(folder),list(folder.iterdir())
        return state.execute('03_score_'+model+'_'+fold+'_'+threshold_tag(mask),dict(
            manifest=sha256(splits[fold]),predictions=sha256(prediction),protocol=sha256(protocol),source=fingerprint),action)

    validation=sweep('validation',config['mask_thresholds_validation_only'])
    candidates=[]
    for mask in config['mask_thresholds_validation_only']:
        metric=read_json(Path(score('E0','validation',mask,prediction_file(validation,mask)))/'metrics.json')
        candidates.append(dict(mask_threshold=mask,clean_hmean=metric['clean_unique']['hmean'],
            degraded_hmean=metric['degraded_instance']['hmean'],balanced_validation_hmean=.5*(
                metric['clean_unique']['hmean']+metric['degraded_instance']['hmean'])))
    choice=choose_mask(candidates)
    selected_path=inputs/'SELECTED_MASK__E0__validation_mean_clean_degraded_Hmean__locked_before_test.json'
    def select():
        result=dict(native=.3,selected=choice['mask_threshold'],validation_evidence=dict(selected=choice,candidates=candidates,
            fold='validation_only',rule='max mean clean/degraded Hmean; native-first stable ties'))
        save_json(selected_path,result)
        return result,[selected_path]
    selections['E0']=state.execute('04_lock_E0_mask_on_validation_before_test',dict(candidates=candidates),select)
    test=sweep('test',list(dict.fromkeys([.3,choice['mask_threshold']])))
    for model in ('E0','Aprime'):
        for point in ('native','selected'):
            mask=selections[model][point]
            pred=prediction_file(test,mask) if model=='E0' else Path(reference['predictions']['test'][str(mask)])
            experiments[model+'__'+point]=score(model,'test',mask,pred)
    native_predictions=dict(E0=prediction_file(test,.3),Aprime=Path(reference['predictions']['test']['0.3']))
    final_dir=outputs/'FINAL_EDA__E0_original_vs_Aprime_FP32__same_FULL_R1_test'
    final_dir.mkdir(parents=True,exist_ok=True)
    native_evaluations={model:experiments[model+'__native'] for model in ('E0','Aprime')}
    def diagnose():
        summary=analyze_native(splits['test'],protocol,native_predictions,native_evaluations,final_dir,config)
        names=('WORDS__native_mask__E0_vs_Aprime__same_GT_ID_clean_degraded.csv',
            'REVIEW_QUEUE__stratified_word_crops__manual_labels_pending.json',
            'DIAGNOSIS__native_E0_Aprime__word_size_geometry_and_paired_transitions.json')
        return summary,[final_dir/name for name in names]
    diagnosis=state.execute('05_EDA_native_word_size_geometry_clean_degraded_and_finetuning_transitions',dict(
        predictions={m:sha256(p) for m,p in native_predictions.items()},source=fingerprint),diagnose)
    icdar_path=a.repo/config['icdar_historical_anchor']['relative_path']
    if sha256(icdar_path) != config['icdar_historical_anchor']['sha256']:
        raise ValueError('Historical verified ICDAR anchor changed')
    icdar=read_json(icdar_path)
    anchor_copy=inputs/'HISTORICAL_ICDAR2015__verified_500_image_original_baseline_EDA__not_rerun.json'
    anchor_copy.write_bytes(icdar_path.read_bytes())
    def report():
        result=make_report(experiments,selections,splits['test'],final_dir,config,diagnosis,icdar)
        return result,[Path(result['report']),Path(result['csv']),
            final_dir/'COMPARISON_AND_EDA__original_Monkey_vs_mixed_finetuning__native_selected.json']
    final=state.execute('06_report_original_baseline_vs_adaptation_native_and_selected',dict(
        metrics={k:sha256(Path(v)/'metrics.json') for k,v in experiments.items()},source=fingerprint),report)
    review_dir=outputs/'REVIEW_CROPS__native_GT_E0_Aprime__clean_degraded_pairs'
    queue=final_dir/'REVIEW_QUEUE__stratified_word_crops__manual_labels_pending.json'
    def review():
        index=render_review(splits['test'],locations,native_predictions,queue,review_dir,config['eda']['review_max_side'])
        return str(index),list(review_dir.iterdir())
    state.execute('07_render_small_review_packet__GT_green_E0_cyan_Aprime_orange',dict(queue=sha256(queue),
        locations=locations,source=fingerprint),review)
    marker=root/MARKER
    if not marker.exists():
        save_json(marker,dict(completed_utc=utc_stamp(),run_name=config['run_name'],source_fingerprint=fingerprint,
            final=final,experiments=list(experiments),scope='Inference and automatic EDA finished; human review/causal diagnosis not yet completed.'))
    archive=package(root,marker)
    print('RESULTS_AND_EDA_ZIP:',archive,'\nREPORT:',final['report'],flush=True)
