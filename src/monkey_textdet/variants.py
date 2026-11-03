"""Official MMOCR ASF variant and strictly verified initialization; no backbone swap."""
import argparse
from pathlib import Path

from .common import save_json, sha256

VARIANTS = ('A_DBNet_noASF__mixed_clean_degraded', 'B_DBNet_plus_ASF__mixed_clean_degraded')


def apply_variant(cfg, variant):
    if variant not in VARIANTS or cfg.model.neck.type != 'FPNC':
        raise ValueError('Expected the pinned Monkey DBNet FPNC and a declared variant')
    cfg.model.neck['asf_cfg'] = (dict(attention_type='ScaleChannelSpatial')
                                if variant == VARIANTS[1] else None)


def initialization_key_audit(model_state, source_state, asf):
    """Reject any mismatch outside the newly added official ASF parameters."""
    added = sorted(set(model_state)-set(source_state))
    unexpected = sorted(set(source_state)-set(model_state))
    mismatch = [k for k in set(model_state)&set(source_state)
                if tuple(model_state[k].shape) != tuple(source_state[k].shape)]
    allowed = ('neck.asf_conv.', 'neck.asf_attn.')
    if unexpected or mismatch or any(not k.startswith(allowed) for k in added):
        raise ValueError(f'Unsafe detector initialization: missing={added}, unexpected={unexpected}, shapes={mismatch}')
    if bool(added) != bool(asf):
        raise ValueError('ASF configuration and newly initialized parameter set disagree')
    if asf and not all(any(k.startswith(p) for k in added) for p in allowed):
        raise ValueError('Incomplete official ASF module')
    return added


def initialize(a):
    import torch
    from mmengine.config import Config
    from mmengine.runner import set_random_seed
    from mmengine.utils import import_modules_from_strings
    from mmocr.registry import MODELS
    from mmocr.utils import register_all_modules
    cfg = Config.fromfile(str(a.config))
    register_all_modules(init_default_scope=True)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    set_random_seed(a.seed)
    model = MODELS.build(cfg.model)
    model.init_weights()
    state = torch.load(a.original, map_location='cpu', weights_only=False)
    source = state.get('state_dict', state)
    # The pinned checkpoint already passed strict loading in Phase1/3.
    added = initialization_key_audit(model.state_dict(), source, a.variant == VARIANTS[1])
    combined = dict(model.state_dict())
    combined.update(source)
    model.load_state_dict(combined, strict=True)
    loaded = model.state_dict()
    for key, value in source.items():
        if not torch.equal(loaded[key].cpu(), value.cpu().to(loaded[key].dtype)):
            raise ValueError('Shared weights changed during initialization: '+key)
    if a.verify_only:
        if a.variant != VARIANTS[0] or a.output.resolve() != a.original.resolve():
            raise ValueError('Verify-only mode reuses only the original noASF detector')
    else:
        if a.output.resolve() == a.original.resolve():
            raise ValueError('Never overwrite the original checkpoint')
        a.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=loaded, meta=dict(variant=a.variant,
            original_checkpoint_sha256=sha256(a.original), initialization_seed=a.seed,
            only_new_keys=added)), a.output)
    save_json(a.receipt, dict(variant=a.variant, original_checkpoint_sha256=sha256(a.original),
        initialized_checkpoint_sha256=sha256(a.output), shared_keys_verified=len(source),
        reused_original_without_duplicate_checkpoint=a.verify_only,
        newly_initialized_keys=added, initialization_seed=a.seed,
        total_parameters=sum(p.numel() for p in model.parameters()),
        new_ASF_parameters=sum(p.numel() for k,p in model.named_parameters() if k in added),
        strict_full_state_load_passed=True, shared_weight_equality_passed=True,
        note='Official MMOCR ASF; no identity-init modification. Fresh ASF weights are trained, never treated as a pretrained DBNet++ checkpoint.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('config', 'original', 'output', 'receipt'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--verify-only', action='store_true')
    initialize(p.parse_args())
