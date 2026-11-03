"""Audit a real first OptimWrapper update, including warmup, clipping and accumulation."""
import copy

import numpy as np

from .common import save_json


def parameter_group(name):
    if name.startswith('backbone.'):
        return 'backbone'
    if name.startswith(('neck.asf_conv.', 'neck.asf_attn.')):
        return 'ASF'
    if name.startswith('neck.'):
        return 'neck_without_ASF'
    if name.startswith('det_head.'):
        return 'det_head'
    return 'other'


def update_summary(before, after):
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Invalid parameter snapshot in FP32 optimizer probe')
    delta = np.abs(b.astype(np.float64)-a.astype(np.float64))
    return dict(elements=int(a.size), changed_elements=int(np.count_nonzero(a != b)),
                sum_abs_delta=float(delta.sum()), max_abs_delta=float(delta.max()))


def attach_attention_recorder(model):
    """Record actual gating/feature magnitudes, not random-feature simulations."""
    import torch
    latest, features = {}, [None]

    def before_attention_conv(module, args):
        features[0] = args[0].detach()

    def after_attention(module, args, output):
        attention = output.detach().float()
        feature = features[0]
        if feature is None or attention.shape[1] != 4 or not torch.isfinite(attention).all():
            raise RuntimeError('Invalid real ASF attention diagnostic')
        grouped = feature.float().reshape(feature.shape[0], 4, -1, *feature.shape[-2:])
        dims = (0, 2, 3)
        before = grouped.square().mean(dim=(0, 2, 3, 4)).sqrt()
        after = (grouped*attention.unsqueeze(2)).square().mean(dim=(0, 2, 3, 4)).sqrt()
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise RuntimeError('Non-finite real FPN feature diagnostic')
        latest.clear()
        latest.update(source_stage_strides=[4, 8, 16, 32],
            region='entire padded shared FPN grid; includes background/padding, not GT-specific',
            attention_mean=attention.mean(dim=dims).cpu().tolist(),
            fraction_attention_below_0p05=(attention < .05).float().mean(dim=dims).cpu().tolist(),
            fraction_attention_above_0p95=(attention > .95).float().mean(dim=dims).cpu().tolist(),
            feature_rms_before_ASF=before.cpu().tolist(), feature_rms_after_ASF=after.cpu().tolist())
        features[0] = None

    handles = [model.neck.asf_conv.register_forward_pre_hook(before_attention_conv),
               model.neck.asf_attn.register_forward_hook(after_attention)]
    return latest, handles


def save_final_update_audit(model, initial_checkpoint, work_dir):
    """Measure final named parameters automatically; buffers are not counted."""
    import torch
    from .precision_control import verify_fp32
    verify_fp32(model)
    state = torch.load(initial_checkpoint, map_location='cpu', weights_only=False)
    source = state.get('state_dict', state)
    groups, tensors = {}, {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        baseline = source[name].float().cpu().numpy()
        stats = update_summary(baseline, p.detach().cpu().numpy())
        tensors[name] = dict(dtype=str(p.dtype), **stats)
        g = groups.setdefault(parameter_group(name), dict(elements=0, changed_elements=0,
            sum_abs_delta=0., max_abs_delta=0.))
        for key in ('elements','changed_elements','sum_abs_delta'):
            g[key] += stats[key]
        g['max_abs_delta'] = max(g['max_abs_delta'], stats['max_abs_delta'])
    for g in groups.values():
        g['changed_fraction'] = g['changed_elements']/g['elements']
    save_json(work_dir/'TRAINING_AUDIT__actual_FP32_named_parameter_changes__initial_to_final.json',
        dict(groups=groups, per_tensor=tensors, scope='Actual named trainable parameters, excludes buffers; '
             'initial values cast to FP32 before comparison. Movement does not measure useful learning.'))


def run_optimizer_probe(cfg, a):
    import torch
    from mmengine.hooks import Hook
    from mmengine.runner import Runner, load_checkpoint
    from .precision_control import enforce_fp32
    from .precision_control import runtime_math_flags
    probe = copy.deepcopy(cfg)
    accumulation = cfg.optim_wrapper.accumulative_counts
    probe.train_cfg.max_iters = accumulation
    probe.train_cfg.val_interval = accumulation+1
    probe.val_cfg = probe.val_dataloader = probe.val_evaluator = None
    probe.default_hooks.checkpoint = None
    probe.custom_hooks = []
    probe.work_dir = str(a.work_dir)
    # Keep the real 20k-iteration schedulers: the first update uses the actual
    # warmup LR, not the higher nominal base LR used by the historical probe.
    probe.load_from = str(a.initial_checkpoint)
    probe.resume = False

    class FirstUpdateAudit(Hook):
        def __init__(self):
            self.initial, self.gradient_flags, self.handles, self.lrs = {}, {}, [], []
            self.sample_ids = []
            self.result = None

        def before_train(self, runner):
            for name, p in runner.model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.dtype != torch.float32:
                    raise RuntimeError('Trainable parameter is not FP32: '+name)
                self.initial[name] = p.detach().cpu().numpy().copy()
                def capture(g, key=name):
                    self.gradient_flags.setdefault(key, []).append(
                        (torch.isfinite(g).all().detach(), torch.count_nonzero(g).detach()))
                self.handles.append(p.register_hook(capture))

        def before_train_iter(self, runner, batch_idx, data_batch=None):
            self.lrs.append([group['lr'] for group in runner.optim_wrapper.optimizer.param_groups])
            self.sample_ids.append([sample.get('sample_id') for sample in data_batch['data_samples']])
            if not self.sample_ids[-1] or any(not isinstance(key,str) or not key for key in self.sample_ids[-1]):
                raise RuntimeError('Missing sample IDs in the real optimizer probe')

        def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            if runner.iter+1 != accumulation:
                return
            groups, gradients, seen = {}, {}, 0
            for name, p in runner.model.named_parameters():
                if name not in self.initial:
                    continue
                group = parameter_group(name)
                stats = update_summary(self.initial[name], p.detach().cpu().numpy())
                g = groups.setdefault(group, dict(elements=0, changed_elements=0, sum_abs_delta=0.,
                    max_abs_delta=0., tensors=0, tensors_with_nonzero_gradients=0))
                for key in ('elements', 'changed_elements', 'sum_abs_delta'):
                    g[key] += stats[key]
                g['max_abs_delta'] = max(g['max_abs_delta'], stats['max_abs_delta'])
                g['tensors'] += 1
                flags = self.gradient_flags.get(name, [])
                if flags:
                    seen += 1
                    if not all(bool(finite.item()) for finite, _ in flags):
                        raise RuntimeError('Non-finite gradient: '+name)
                    g['tensors_with_nonzero_gradients'] += int(any(int(n.item()) > 0 for _, n in flags))
                    gradients[name] = True
            expected = {'backbone', 'neck_without_ASF', 'det_head'}
            if runner.model.neck.asf_cfg is not None:
                expected.add('ASF')
            if not expected <= set(groups):
                raise RuntimeError('Expected trainable module absent in FP32 probe')
            for name in expected:
                g = groups[name]
                if not g['changed_elements'] or not g['tensors_with_nonzero_gradients']:
                    raise RuntimeError('No actual first-update movement/nonzero gradient in '+name)
                g['changed_fraction'] = g['changed_elements']/g['elements']
            states = {}
            for state in runner.optim_wrapper.optimizer.state.values():
                for key, value in state.items():
                    if key in ('exp_avg', 'exp_avg_sq'):
                        states[str(value.dtype)] = states.get(str(value.dtype), 0)+value.numel()
                        if value.dtype != torch.float32:
                            raise RuntimeError('Adam moment state is not FP32')
            self.result = dict(probe_only=True, training_forward_backward_optimizer='passed',
                actual_OptimWrapper_with_declared_accumulation_clipping_and_warmup=True,
                observed_optimizer_updates=1, microiterations=accumulation,
                optimizer_lr_before_each_microiteration=self.lrs,
                sample_ids_by_microiteration=self.sample_ids,
                parameter_storage='float32', optimizer_moment_dtype_elements=states,
                runtime_math_flags=runtime_math_flags(),
                parameter_groups=groups, parameters_with_gradients=seen,
                ASF_parameter_tensors_with_finite_gradients=sum(parameter_group(n)=='ASF' for n in gradients),
                no_checkpoint_weights_modified=True, no_probe_checkpoint_saved=True,
                peak_allocated_gpu_mib=torch.cuda.max_memory_allocated()/1024**2)

    hook = FirstUpdateAudit()
    runner = Runner.from_cfg(probe)
    runner.model.init_weights()
    enforce_fp32(runner.model)
    load_checkpoint(runner.model, str(a.initial_checkpoint), map_location='cpu', strict=True)
    runner.register_hook(hook, priority='LOWEST')
    try:
        runner.train()
        if hook.result is None:
            raise RuntimeError('FP32 optimizer update audit did not execute')
        save_json(a.result, hook.result)
    finally:
        for handle in hook.handles:
            handle.remove()
