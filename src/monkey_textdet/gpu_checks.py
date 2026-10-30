"""Run inside the private Python 3.11 environment, never on the login node."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from .common import read_json, save_json, sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--download',action='store_true')
    p.add_argument('--mmocr-root',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from mmcv.ops import ModulatedDeformConv2d, nms
    assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'Exactly one allocated GPU/MIG device required'
    expected={'torch':'2.9.0','torchvision':'0.24.0','numpy':'1.26.4','mmengine':'0.10.7',
              'mmcv':'2.0.1','mmdet':'3.1.0','mmocr':'1.0.1','transformers':'4.57.1','shapely':'2.2.0'}
    versions={k:importlib.metadata.version(k) for k in expected}
    assert all(versions[k].split('+')[0]==v for k,v in expected.items()), versions
    boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device='cuda')
    _,keep=nms(boxes,torch.tensor([.9,.8],device='cuda'),.5)
    assert keep.numel()==1
    op=ModulatedDeformConv2d(2,2,3,padding=1).cuda()
    x=torch.randn(1,2,8,8,device='cuda',requires_grad=True)
    y=op(x,torch.zeros(1,18,8,8,device='cuda'),torch.ones(1,9,8,8,device='cuda'))
    y.square().mean().backward()
    assert torch.isfinite(y).all() and x.grad is not None and torch.isfinite(x.grad).all()
    locks=read_json(a.lock)
    checkpoint=a.mmocr_root/'model_weight/dbnet_mkv2vitae_icdar2015.pth'
    backbone=a.mmocr_root/'pretrained/monkeyocrv2_as'
    if a.download:
        from huggingface_hub import hf_hub_download,snapshot_download
        hf_hub_download('HB16888/MonkeyOCRv2_det',checkpoint.name,
                        revision=locks['detector_hf_revision'],local_dir=checkpoint.parent)
        snapshot_download('zenosai/MonkeyOCRv2-AS',revision=locks['backbone_hf_revision'],local_dir=backbone)
    assert sha256(checkpoint)==locks['checkpoint_sha256'], 'Official detector checkpoint hash mismatch'
    cfg=a.mmocr_root/'configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py'
    assert sha256(cfg)==locks['official_config_sha256'], 'Official base config hash mismatch'
    assert sha256(backbone/'model.safetensors')=='8f4d2dc342c263ae2299a66dc39631f9424fa09c4f575b142ea40e1b1cc1bad4'
    for name,expected_hash in locks['backbone_files_sha256'].items():
        assert sha256(backbone/name)==expected_hash, 'Backbone file/code hash mismatch: '+name
    adapter=a.mmocr_root/'mmocr/models/common/backbones/monkeyocr_v2_vitae.py'
    assert sha256(adapter)==locks['backbone_adapter_sha256'],'Installed backbone adapter differs from the pinned official add-on'
    result=dict(versions=versions,gpu_name=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
                vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                cuda=torch.version.cuda,cuda_nms_and_deformconv_forward_backward='passed',
                checkpoint_sha256=sha256(checkpoint),official_config_sha256=sha256(cfg),
                backbone_adapter_sha256=sha256(adapter),verified_backbone_files=locks['backbone_files_sha256'])
    save_json(a.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
