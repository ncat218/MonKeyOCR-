"""FULL_R1: two matched mixed-data trainings, native/validation-selected masks."""
import copy
import math
import signal
import sys
from pathlib import Path

from .common import Processes, StageState, jsonl, read_json, save_json, sha256, utc_stamp, stop_signals
from .mask_sweep import choose_mask, prediction_file, threshold_tag
from .variants import VARIANTS, apply_variant

SUITE = 'full_r1_dbnet_vs_asf_and_validation_mask_sweep'


def validate(config):
    from .training import training_schedule
    schedule = training_schedule(config['training'])
    precision = config['training'].get('parameter_precision', 'native')
    if precision not in ('native', 'fp32'):
        raise ValueError('Unknown parameter_precision')
    if config.get('precision_control') and precision != 'fp32':
        raise ValueError('The job9276 precision control requires explicit fp32 storage')
    if config.get('precision_control'):
        from .precision_control import validate_control
        validate_control(config)
    if config['suite'] != SUITE:
        raise ValueError('Unknown full-data suite')
    if config.get('split_scope') != 'official_TextVQA_train_intersection_only':
        raise ValueError('Explicitly agree the detection split scope before running')
    if config['split'] != {'train': .7, 'validation': .15, 'test': .15}:
        raise ValueError('Agreed FULL_R1 detection split is 70/15/15 by source/clean SHA')
    from .training import checkpoint_settings
    checkpoint_settings(config['training'], schedule)
    if config['training']['checkpoint_policy'] != 'final_weights_only':
        raise ValueError('This revised FULL_R1 suite saves final weights only')
    if not 0 < schedule['val_interval'] <= schedule['max_iters'] or schedule['max_iters'] % schedule['val_interval']:
        raise ValueError('Invalid validation interval')
    thresholds = config['mask_thresholds_validation_only']
    if not thresholds or thresholds[0] != .3 or len(set(thresholds)) != len(thresholds):
        raise ValueError('Native 0.3 must be first (stable selection tie break)')
    if any(not math.isfinite(t) or not 0 < t < 1 for t in thresholds):
        raise ValueError('Invalid thresholds')
    if len({threshold_tag(t) for t in thresholds}) != len(thresholds):
        raise ValueError('Thresholds must have unique two-decimal output names')
    if config['evaluation']['bootstrap_repetitions'] < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    return schedule


def run(a):
    from mmengine.config import Config
    from .data import discover_inputs, freeze_splits, export_training_annotations
    from .full_data import build_full_shared, read_metadata
    from .inference import generate_inference_config
    from .pipeline import code_fingerprint
    from .training import make_training_config
    from .vendor.evaluate_shared import evaluate
    from .asf_reporting import make_report
    config = read_json(a.config)
    validate(config)
    locks = read_json(a.repo/'configs/verified_phase1_phase2_sources.json')
    root = a.workspace/config['run_name']
    inputs, outputs = root/'inputs', root/'outputs'
    for p in (inputs, outputs):
        p.mkdir(parents=True, exist_ok=True)
    fingerprint = code_fingerprint(a.repo)
    # Location overrides bind file bytes below; preserve all behavioral data options.
    stable = copy.deepcopy(config)
    stable.pop('dataset_parent', None)
    stable.pop('workspace_root', None)
    for key in ('clean_json', 'textocr_json', 'main_image_manifest', 'clean_images_root', 'degraded_root'):
        stable['data'].pop(key, None)
    state = StageState(root/'state', dict(configuration=stable, source_fingerprint=fingerprint))
    process = Processes(a.logs/'FULL_R1_DBNet_ASF_experiment_workers')
    for sig in stop_signals():
        signal.signal(sig, process.request_stop)
    paths = discover_inputs(a.dataset_parent, config['data'], exclude=[a.workspace, a.repo])
    _, metadata = read_metadata(paths)
    save_json(inputs/'01_FULL_R1_actual_dataset_paths_and_metadata_identity.json', dict(paths=paths, metadata=metadata))
    save_json(root/'run_configuration__FULL_R1_two_architectures_same_budget.json', config)
    index = dict(
        run_name=config['run_name'], purpose='Fresh mixed-data DBNet versus ASF, native and validation-selected mask thresholds',
        detection_split='Official TextVQA train intersection, 70/15/15 grouped by source and clean SHA256',
        outputs=str(outputs), inputs=str(inputs), logs=str(root/'logs'),
        training_checkpoints=str(outputs/'training'),
        global_predictions=str(outputs/'global_predictions'),
        validation_only_selection=str(outputs/'validation_selection'),
        final_comparison=str(outputs/'FINAL_COMPARISON__FULL_R1_DBNet_vs_ASF__native_and_validation_selected_masks'),
        completion_marker=str(root/'COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json'))
    if config.get('precision_control'):
        index.update(parameter_precision='FP32 training/inference tensors; runtime backend flags recorded',
            borrowed_inputs_from=str(a.workspace/config['precision_control']['reference_run']),
            frozen_reference_receipt=str(inputs/'02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT/REFERENCE__job9276_frozen_inputs_and_image_hashes.json'),
            initial_validation_diagnostics=str(outputs/'INITIAL_VALIDATION_DIAGNOSTICS__untrained_A_B__not_test'),
            automatic_delivery_zip=str(root/'RESULTS_AND_DIAGNOSTICS__FULL_R1_FP32_DBNet_ASF__final_weights_only.zip'))
    save_json(root/'READ_ME_FIRST__FULL_R1_DBNet_ASF_run_outputs_and_logs.json', index)
    shared_dir = inputs/'02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT'
    split_dir = inputs/'03_frozen_detection_source_splits__official_train_intersection__70_15_15'
    annotation_dir = inputs/'04_shared_mixed_training_and_exact_validation_annotations'
    reference = None

    if config.get('precision_control'):
        from .precision_control import bind_reference
        def reference_action():
            result = bind_reference(a.workspace, config, paths, shared_dir, split_dir, annotation_dir)
            artifacts = list(shared_dir.iterdir())+list(split_dir.iterdir())+list(annotation_dir.iterdir())
            artifacts += [Path(result['shared']['manifest'])]
            artifacts += [Path(p) for p in result['splits'].values()]
            artifacts += [Path(p) for p in result['annotations'].values()]
            artifacts += [Path(result['ASF_initial'])]
            return result, artifacts
        reference = state.execute('01_FP32_verify_job9276_frozen_inputs_and_all_source_image_hashes',
            dict(reference=config['precision_control'], metadata=metadata, source=fingerprint), reference_action)
        shared, splits, annotations = reference['shared'], reference['splits'], reference['annotations']

    def audit_action():
        result = build_full_shared(paths, shared_dir)
        return result, list(shared_dir.iterdir())

    all_metadata = dict(metadata['metadata_sha256'], main=sha256(paths['main_image_manifest']),
                        clean=sha256(paths['clean_json']), textocr=sha256(paths['textocr_json']))
    if reference is None:
        shared = state.execute('01_audit_full_MAIN_and_freeze_official_train_GT_intersection',
                               dict(files=all_metadata, source=fingerprint), audit_action)
    locations = dict(clean_root=paths['clean_images_root'], degraded_root=paths['degraded_root'])
    location_file = shared_dir/'dataset_locations.json'
    save_json(location_file, locations)

    def split_action():
        result = freeze_splits(shared['manifest'], split_dir, config)
        report = read_json(split_dir/'split_policy_and_counts.json')
        report['scope'] = config['split_scope']
        report['disclosure'] = ('These are detection folds inside official TextVQA train, not official TextVQA val/test. '
            'Previously inspected Phase3 source IDs can recur; both models restart from the same original checkpoint. '
            'TextOCR train was used in Monkey pretraining; no unseen-source claim.')
        save_json(split_dir/'split_policy_and_counts.json', report)
        return result, list(split_dir.iterdir())

    if reference is None:
        splits = state.execute('02_freeze_same_source_clean_SHA_grouped_train_validation_test',
            dict(master=shared['manifest_sha256'], fractions=config['split'], seed=config['seed']), split_action)

    def annotation_action():
        result = export_training_annotations(splits, locations, annotation_dir,
                                             train_conditions=('E2_clean_and_degraded_balanced',))
        return result, [Path(p) for p in result.values()]

    if reference is None:
        annotations = state.execute('03_export_same_mixed_targets_for_DBNet_and_ASF',
            dict(splits={k: sha256(v) for k,v in splits.items()}, source=fingerprint, locations=locations), annotation_action)
    # Training budget covers at least one traversal of both clean/degraded entries.
    train_sources = sum(1 for _ in jsonl(splits['train']))
    student_inputs = config['training']['optimizer_updates']*config['training']['effective_batch_size']
    if student_inputs < 2*train_sources:
        raise ValueError(f'Budget {student_inputs} input draws does not cover {2*train_sources} mixed entries once')
    official = a.mmocr_root/'configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py'
    original = a.mmocr_root/'model_weight/dbnet_mkv2vitae_icdar2015.pth'
    backbone = a.mmocr_root/'pretrained/monkeyocrv2_as'
    if sha256(official) != locks['official_config_sha256'] or sha256(original) != locks['checkpoint_sha256']:
        raise ValueError('Original model identity mismatch')
    configs = inputs/'05_model_configs__only_FPNC_ASF_differs'
    configs.mkdir(parents=True, exist_ok=True)
    experiments, training_results, selections, prepared = {}, {}, {}, {}

    def score(stage, manifest, pred, folder, bootstrap=0):
        def action():
            result = evaluate(manifest, pred, shared_dir/'evaluation_protocol.json', folder, bootstrap)
            result['limitations'] = [
                'Synthetic FULL_R1 realistic_mix/L2; image_relative_fallback severity.',
                'Detection folds within official TextVQA train, not an unseen pretraining/test dataset.',
                'Both architectures fine-tuned on identical mixed entries; one training seed.',
                'Coverage/granularity diagnostics do not establish recognition or VQA correctness.']
            save_json(folder/'metrics.json', result)
            # Vendor evaluator's historical baseline prose is not the scope of this run.
            (folder/'clean_vs_degraded.md').write_text(
                '# FULL_R1 fine-tuned detector evaluation\n\n'
                'Same held-out detection source fold and TextOCR word polygons. '
                'See the final comparison for native versus validation-selected mask thresholds.\n', encoding='utf-8')
            return str(folder), list(folder.iterdir())
        return state.execute(stage, dict(manifest=sha256(manifest), predictions=sha256(pred),
            protocol=sha256(shared_dir/'evaluation_protocol.json'), bootstrap=bootstrap), action)

    def sweep(stage, fold, checkpoint, model_config, label, masks, manifest_override=None):
        manifest = Path(manifest_override) if manifest_override else Path(splits[fold])
        out = outputs/'global_predictions'/label/fold
        mask_file = configs/(label+'__'+fold+'__fixed_mask_thresholds.json')
        save_json(mask_file, dict(mask_thresholds=masks))
        def action():
            command = [sys.executable, '-m', 'monkey_textdet.mask_sweep', '--manifest', manifest,
                '--locations', location_file, '--model-config', model_config, '--checkpoint', checkpoint,
                '--thresholds', mask_file, '--output', out, '--model-label', label]
            record_attention = bool(reference and label == VARIANTS[1])
            if record_attention:
                command.append('--record-attention')
            process.run(command,
                label+'__'+('FP32_' if reference else '')+fold+'__shared_probability_map_global_inference', cwd=a.mmocr_root)
            artifacts = [out/'COMPLETE__all_requested_mask_predictions.json', out/'immutable_map_sweep_identity.json']
            artifacts += [prediction_file(out, t) for t in masks]
            if record_attention:
                artifacts += [out/'DIAGNOSTICS__real_ASF_attention_and_feature_RMS__per_image.jsonl']
            return str(out), artifacts
        return state.execute(stage, dict(manifest=sha256(manifest), checkpoint=sha256(checkpoint),
            config=sha256(model_config), masks=masks, source=fingerprint), action)

    for variant in VARIANTS:
        train_dir = outputs/'training'/variant
        initial = (original if variant == VARIANTS[0] else
                   inputs/'06_initialized_checkpoints'/('initial_detector__'+variant+'__from_same_original.pth'))
        if reference and variant == VARIANTS[1]:
            initial = Path(reference['ASF_initial'])
        receipt = configs/('INITIALIZATION_RECEIPT__'+variant+'__shared_original_weights.json')
        train_config = configs/('training__'+variant+'__same_mixed_entries_and_optimizer_budget.py')
        model_config = configs/('inference__'+variant+'__native_global_resolution.py')
        make_training_config(official, backbone, initial, annotations, shared_dir/'evaluation_protocol.json',
                             train_dir, config, 'E2_clean_and_degraded_balanced', train_config)
        cfg = Config.fromfile(str(train_config), import_custom_modules=False)
        apply_variant(cfg, variant)
        cfg.default_hooks.checkpoint.filename_tmpl = 'FINAL_WEIGHTS_ONLY__'+variant+'__micro_iteration_{}.pth'
        if reference:
            cfg.default_hooks.checkpoint.filename_tmpl = 'FINAL_WEIGHTS_ONLY__FULL_TextVQA_FP32__'+variant+'__micro_iteration_{}.pth'
        cfg.dump(str(train_config))
        generate_inference_config(official, backbone, model_config)
        infer_cfg = Config.fromfile(str(model_config), import_custom_modules=False)
        apply_variant(infer_cfg, variant)
        from .precision_control import apply_precision
        apply_precision(infer_cfg, config['training'].get('parameter_precision', 'native'))
        infer_cfg.dump(str(model_config))

        def initialize_action():
            command = [sys.executable, '-m', 'monkey_textdet.variants', '--config', train_config,
                '--original', original, '--output', initial, '--receipt', receipt, '--variant', variant,
                '--seed', str(config['seed'])]
            if variant == VARIANTS[0]:
                command.append('--verify-only')
            elif reference:
                command += ['--reuse-initial', initial]
            init_label = (variant+'__FP32_verify_exact_job9276_initial_values_no_checkpoint_copy' if reference else
                          variant+'__initialize_only_new_ASF_keys_and_verify_shared_weights')
            process.run(command, init_label, cwd=a.mmocr_root)
            return read_json(receipt), [initial, receipt]

        state.execute('04_initialize_'+variant, dict(config=sha256(train_config), original=sha256(original),
                      source=fingerprint), initialize_action)
        probe = train_dir/'COMPATIBILITY__forward_backward_no_saved_training_weights.json'

        def probe_action():
            process.run([sys.executable, '-m', 'monkey_textdet.training', '--config', train_config,
                '--initial-checkpoint', initial, '--work-dir', train_dir/'compatibility_probe',
                '--result', probe, '--experiment', variant, '--probe-only'],
                variant+('__FP32_first_update_compatibility__real_warmup_accumulation_clipping' if reference else
                         '__forward_backward_optimizer_compatibility'), cwd=a.mmocr_root)
            return read_json(probe), [probe]

        state.execute('05_compatibility_'+variant, dict(config=sha256(train_config), initial=sha256(initial)), probe_action)
        prepared[variant] = dict(train_config=train_config, model_config=model_config,
                                 initial=initial, receipt=receipt, probe=probe)

    # Complete initialization and real-update probes for BOTH branches before
    # spending hours training A and discovering a B preflight failure later.
    if reference:
        from .precision_control import verify_probe_pair, initial_validation_rows
        verify_probe_pair({variant:read_json(prepared[variant]['probe']) for variant in VARIANTS})
        from .common import write_jsonl
        subset = inputs/'07_validation_only_initial_detector_diagnostics'/('manifest__'+
            str(config['precision_control']['diagnostic_validation_sources'])+'_frozen_validation_sources__includes_largest_resized_canvas__not_test.jsonl')
        subset.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(subset, initial_validation_rows(splits['validation'], config['precision_control']['diagnostic_validation_sources']))
        for variant in VARIANTS:
            files = prepared[variant]
            initial_out = Path(sweep('05b_initial_validation_diagnostics_'+variant,
                'initial_validation_diagnostics', files['initial'], files['model_config'], variant, [.3], subset))
            folder = outputs/'INITIAL_VALIDATION_DIAGNOSTICS__untrained_A_B__not_test'/variant
            def initial_score_action():
                metrics = evaluate(subset, prediction_file(initial_out, .3), shared_dir/'evaluation_protocol.json', folder, 0)
                save_json(folder/'SCOPE__initial_weights_validation_subset_only.json', dict(
                    scope='Fixed validation subset including largest resized canvas for memory checking; native mask, no hyperparameter selection, not test.',
                    clean_hmean=metrics['clean_unique']['hmean'], degraded_hmean=metrics['degraded_instance']['hmean']))
                return str(folder), list(folder.iterdir())
            state.execute('05c_score_initial_validation_'+variant,
                dict(manifest=sha256(subset), prediction=sha256(prediction_file(initial_out,.3))), initial_score_action)

    for variant in VARIANTS:
        train_dir = outputs/'training'/variant
        train_config = prepared[variant]['train_config']
        model_config = prepared[variant]['model_config']
        initial = prepared[variant]['initial']
        receipt = prepared[variant]['receipt']
        result_path = train_dir/('TRAINING_COMPLETE__'+variant+'__final_weights_only.json')

        def train_action():
            process.run([sys.executable, '-m', 'monkey_textdet.training', '--config', train_config,
                '--initial-checkpoint', initial, '--work-dir', train_dir, '--result', result_path,
                '--experiment', variant], variant+('__FP32_mixed_finetune_same_budget_10000_updates_final_weights_only' if reference else
                                                   '__mixed_finetune_same_budget'), cwd=a.mmocr_root)
            result = read_json(result_path)
            best = Path(result['evaluation_checkpoint'])
            artifacts = [result_path, best]
            if reference:
                artifacts.append(train_dir/'TRAINING_AUDIT__actual_FP32_named_parameter_changes__initial_to_final.json')
            return result, artifacts

        trained = state.execute('06_train_'+variant, dict(config=sha256(train_config), initial=sha256(initial),
                                source=fingerprint), train_action)
        training_results[variant] = dict(**trained, initialization_receipt=str(receipt))
        checkpoint = Path(trained['evaluation_checkpoint'])
        validation = Path(sweep('07_validation_masks_'+variant, 'validation', checkpoint,
                               model_config, variant, config['mask_thresholds_validation_only']))
        candidates = []
        for t in config['mask_thresholds_validation_only']:
            folder = outputs/'validation_metrics'/variant/threshold_tag(t)
            scored = score('08_validation_score_'+variant+'__'+threshold_tag(t), splits['validation'],
                           prediction_file(validation, t), folder)
            metrics = read_json(Path(scored)/'metrics.json')
            candidates.append(dict(mask_threshold=t, balanced_validation_hmean=.5*(
                metrics['clean_unique']['hmean']+metrics['degraded_instance']['hmean']),
                clean_hmean=metrics['clean_unique']['hmean'], degraded_hmean=metrics['degraded_instance']['hmean']))
        selected = choose_mask(candidates)
        selection = dict(selected=selected, candidates=candidates, fold='validation_only',
                         rule='max mean(clean Hmean, degraded Hmean); native-first stable tie break')
        selection_path = outputs/'validation_selection'/('SELECTED_MASK__'+variant+'__validation_only.json')

        def select_action():
            save_json(selection_path, selection)
            return selection, [selection_path]

        selection = state.execute('09_lock_validation_mask_'+variant,
                                  dict(candidates=candidates, checkpoint=sha256(checkpoint)), select_action)
        selections[variant] = selection
        masks = list(dict.fromkeys([.3, selection['selected']['mask_threshold']]))
        test_out = Path(sweep('10_test_native_and_locked_mask_'+variant, 'test', checkpoint,
                             model_config, variant, masks))
        for choice, t in [('native_mask_0p30', .3), ('validation_selected_mask', selection['selected']['mask_threshold'])]:
            label = variant+'__'+choice
            folder = outputs/'test_metrics'/variant/threshold_tag(t)
            experiments[label] = score('11_test_score_'+variant+'__'+threshold_tag(t), splits['test'],
                prediction_file(test_out, t), folder, config['evaluation']['bootstrap_repetitions'])
    report_dir = outputs/'FINAL_COMPARISON__FULL_R1_DBNet_vs_ASF__native_and_validation_selected_masks'

    def report_action():
        result = make_report(experiments, report_dir, config, splits['test'], training_results, selections)
        return result, list(report_dir.iterdir())

    final = state.execute('12_final_DBNet_ASF_small_word_corruption_and_threshold_comparison',
        dict(metrics={k: sha256(Path(v)/'metrics.json') for k,v in experiments.items()}, source=fingerprint), report_action)
    marker = root/'COMPLETE__FULL_R1_DBNet_ASF_training_threshold_selection_and_test_finished.json'
    completion = dict(completed_utc=utc_stamp(), run_name=config['run_name'], final=final,
                      experiments=list(experiments), source_fingerprint=fingerprint)
    if reference and marker.exists():
        prior = read_json(marker)
        if {k:v for k,v in prior.items() if k != 'completed_utc'} != {k:v for k,v in completion.items() if k != 'completed_utc'}:
            raise RuntimeError('Existing completion marker differs; do not replace evidence')
    else:
        save_json(marker, completion)
    if reference:
        from .precision_delivery import package_results
        delivery = package_results(root)
        print('RESULTS_AND_DIAGNOSTICS_ZIP:', delivery, flush=True)
    print('FULL_R1 DBNET / ASF SUITE COMPLETED\nREPORT:', final['report'], '\nCSV:', final['csv'], flush=True)
