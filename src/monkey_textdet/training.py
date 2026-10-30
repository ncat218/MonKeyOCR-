"""Generate and run controlled E1/E2 fine-tuning using the official MMOCR model/loss."""
import argparse
import math
from pathlib import Path

from .common import locked_json,read_json,save_json,sha256


def training_schedule(training):
    micro=training['micro_batch_size'];effective=training['effective_batch_size']
    if micro<1 or effective<micro or effective%micro:raise ValueError('effective_batch_size must be divisible by micro_batch_size')
    accumulation=effective//micro
    updates=training['optimizer_updates']
    if updates<1:raise ValueError('optimizer_updates must be positive')
    if not math.isfinite(training['learning_rate']) or training['learning_rate']<=0:
        raise ValueError('learning_rate must be positive and finite')
    if not 0<=training['warmup_optimizer_updates']<updates:
        raise ValueError('Warmup must be shorter than the total optimizer-update budget')
    return dict(accumulation=accumulation,max_iters=updates*accumulation,
                val_interval=training['validation_every_optimizer_updates']*accumulation,
                checkpoint_interval=training['checkpoint_every_optimizer_updates']*accumulation,
                warmup_iters=training['warmup_optimizer_updates']*accumulation)


def make_training_config(official,backbone,original_checkpoint,annotations,protocol,work_dir,config,condition,output):
    from mmengine.config import Config
    cfg=Config.fromfile(str(official));t=config['training'];schedule=training_schedule(t)
    assert condition in ('E1_clean_only','E2_clean_and_degraded_balanced')
    cfg.model.backbone.model_dir=str(backbone)
    cfg.model.backbone.weights=str(backbone/'model.safetensors')
    cfg.model.backbone['freeze']=False
    cfg.custom_imports=dict(imports=['monkey_textdet.mmocr_plugins'],allow_failed_imports=False)
    train_pipeline=cfg.train_dataloader.dataset.pipeline
    val_pipeline=cfg.val_dataloader.dataset.pipeline
    # sample_id is supplied explicitly by the annotation records, never inferred from paths.
    for pipeline in (train_pipeline,val_pipeline):
        for transform in pipeline:
            if transform['type']=='PackTextDetInputs':
                transform['meta_keys']=tuple(transform['meta_keys'])+('sample_id',)
    cfg.train_dataloader=dict(batch_size=t['micro_batch_size'],num_workers=t['num_workers'],
        persistent_workers=t['num_workers']>0,pin_memory=True,
        sampler=dict(type='InfiniteSampler',shuffle=True),
        dataset=dict(type='OCRDataset',data_root='',ann_file=annotations[condition],data_prefix=dict(img_path=''),
                     filter_cfg=None,pipeline=train_pipeline,test_mode=False))
    cfg.val_dataloader=dict(batch_size=1,num_workers=min(2,t['num_workers']),persistent_workers=t['num_workers']>0,
        sampler=dict(type='DefaultSampler',shuffle=False),drop_last=False,
        dataset=dict(type='OCRDataset',data_root='',ann_file=annotations['validation_clean_and_degraded'],
                     data_prefix=dict(img_path=''),pipeline=val_pipeline,test_mode=True))
    cfg.val_evaluator=dict(type='SharedPairedHmeanMetric',index_json=annotations['validation_exact'],
                          protocol_path=str(protocol),prefix='shared')
    cfg.train_cfg=dict(type='IterBasedTrainLoop',max_iters=schedule['max_iters'],val_interval=schedule['val_interval'])
    cfg.val_cfg=dict(type='ValLoop')
    cfg.test_cfg=None;cfg.test_dataloader=None;cfg.test_evaluator=None
    cfg.optim_wrapper=dict(type='OptimWrapper',accumulative_counts=schedule['accumulation'],
        optimizer=dict(type='AdamW',lr=t['learning_rate'],weight_decay=t['weight_decay']),
        clip_grad=dict(max_norm=t['gradient_clip_max_norm'],norm_type=2,error_if_nonfinite=True))
    warmup=min(schedule['warmup_iters'],max(0,schedule['max_iters']-1))
    cfg.param_scheduler=[]
    if warmup:
        cfg.param_scheduler.append(dict(type='LinearLR',start_factor=.1,begin=0,end=warmup,by_epoch=False))
    cfg.param_scheduler.append(dict(type='PolyLR',power=.9,eta_min=t['learning_rate']*.01,
                                    begin=warmup,end=schedule['max_iters'],by_epoch=False))
    cfg.default_hooks=dict(timer=dict(type='IterTimerHook'),logger=dict(type='LoggerHook',interval=25),
        param_scheduler=dict(type='ParamSchedulerHook'),sampler_seed=dict(type='DistSamplerSeedHook'),
        checkpoint=dict(type='CheckpointHook',by_epoch=False,interval=schedule['checkpoint_interval'],
                        max_keep_ckpts=2,save_last=True,save_optimizer=True,save_param_scheduler=True,
                        save_best='shared/balanced_clean_degraded_hmean',rule='greater'))
    cfg.custom_hooks=[dict(type='FiniteLossAndInterruptCheckpointHook',accumulation=schedule['accumulation'])]
    cfg.visualizer=dict(type='TextDetLocalVisualizer',vis_backends=[dict(type='LocalVisBackend')],name='visualizer')
    cfg.log_processor=dict(type='LogProcessor',by_epoch=False,window_size=25)
    cfg.work_dir=str(work_dir);cfg.load_from=str(original_checkpoint);cfg.resume=False
    cfg.launcher='none';cfg.auto_scale_lr=None
    cfg.randomness=dict(seed=config['seed'],deterministic=False)
    cfg.dump(str(output))
    return schedule


def run(a):
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner,load_checkpoint
    from mmengine.utils import import_modules_from_strings
    from mmocr.utils import register_all_modules
    assert torch.cuda.device_count()==1,'One allocated MIG instance per suite'
    cfg=Config.fromfile(str(a.config))
    register_all_modules(init_default_scope=True)
    import_modules_from_strings(**cfg.custom_imports)
    a.work_dir.mkdir(parents=True,exist_ok=True)
    run_identity=dict(experiment=a.experiment,config_sha256=sha256(a.config),initial_checkpoint_sha256=sha256(a.initial_checkpoint),
                      training_source_sha256=sha256(__file__),plugins_sha256=sha256(Path(__file__).with_name('mmocr_plugins.py')))
    locked_json(a.work_dir/'immutable_training_identity.json',run_identity)
    torch.cuda.reset_peak_memory_stats()
    if a.probe_only:
        from mmocr.registry import MODELS
        model=MODELS.build(cfg.model)
        model.init_weights()
        load_checkpoint(model,str(a.initial_checkpoint),map_location='cpu',strict=True)
        model.cuda().train()
        loader=Runner.build_dataloader(cfg.train_dataloader,seed=cfg.randomness.seed)
        batch=next(iter(loader))
        processed=model.data_preprocessor(batch,training=True)
        losses=model(**processed,mode='loss')
        loss,logs=model.parse_losses(losses)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training probe loss')
        loss.backward()
        gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients),'Non-finite/missing training gradients'
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.optim_wrapper.optimizer.lr,
                                    weight_decay=cfg.optim_wrapper.optimizer.weight_decay)
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters()),'Non-finite parameters after optimizer probe'
        save_json(a.result,dict(probe_only=True,training_forward_backward_optimizer='passed',loss=float(loss.detach()),
            parameters_with_gradients=len(gradients),peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
            sample_ids=[s.get('sample_id') for s in processed['data_samples']],
            no_checkpoint_weights_modified=True))
        return
    last=a.work_dir/'last_checkpoint'
    if last.is_file():
        checkpoint=Path(last.read_text().strip())
        if not checkpoint.is_file():raise RuntimeError('last_checkpoint points to a missing training checkpoint')
        cfg.resume=True;cfg.load_from=str(checkpoint)
    runner=Runner.from_cfg(cfg)
    # Strictly verify initialization, including unused heads, before MMEngine's resume/load flow.
    runner.model.init_weights()
    load_checkpoint(runner.model,str(a.initial_checkpoint),map_location='cpu',strict=True)
    runner.train()
    candidates=sorted(a.work_dir.glob('best_*.pth'))
    if len(candidates)!=1:raise RuntimeError(f'Expected one best validation checkpoint; found {candidates}')
    best=candidates[0]
    completed=dict(experiment=a.experiment,best_checkpoint=str(best),best_checkpoint_sha256=sha256(best),
        initial_checkpoint_sha256=sha256(a.initial_checkpoint),initialization='same original detector for E1 and E2',
        checkpoint_selection='maximum validation mean(clean Hmean,degraded Hmean); fixed native postprocessing',
        iterations_completed=runner.iter,optimizer_updates=runner.iter//cfg.optim_wrapper.accumulative_counts,
        peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
        resumed_from=str(checkpoint) if last.is_file() and cfg.resume else None)
    save_json(a.result,completed)
    print(completed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','initial-checkpoint','work-dir','result'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--experiment',required=True)
    p.add_argument('--probe-only',action='store_true')
    run(p.parse_args())
