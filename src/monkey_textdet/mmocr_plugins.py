"""MMOCR validation uses EXACT Phase 2 GT/ROI/evaluator, not the official threshold sweep."""
import math
import signal
from pathlib import Path

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.hooks import Hook
from mmocr.registry import METRICS,HOOKS

from .common import read_json,stop_signals
from .vendor.evaluate_shared import evaluate_image,rates


@METRICS.register_module()
class SharedPairedHmeanMetric(BaseMetric):
    default_prefix='shared'

    def __init__(self,index_json,protocol_path,collect_device='cpu',prefix=None):
        super().__init__(collect_device=collect_device,prefix=prefix)
        self.index=read_json(index_json);self.protocol=read_json(protocol_path)

    def process(self,data_batch,data_samples):
        for sample in data_samples:
            key=sample.get('sample_id')
            if key not in self.index:raise ValueError('Validation sample_id missing from exact GT index')
            item=self.index[key];pred=sample.get('pred_instances')
            scores=pred.get('scores')
            if isinstance(scores,torch.Tensor):scores=scores.detach().cpu().tolist()
            polygons=[np.asarray(p,dtype=float).reshape(-1).tolist() for p in pred.get('polygons')]
            result=evaluate_image(item['gt'],item['roi'],dict(polygons=polygons,scores=scores),self.protocol)
            self.results.append(dict(sample_id=key,condition=item['condition'],counts=result['counts']))

    def compute_metrics(self,results):
        if len({r['sample_id'] for r in results})!=len(results) or {r['sample_id'] for r in results}!=set(self.index):
            raise ValueError('Validation prediction coverage mismatch; checkpoint will not be selected')
        totals={condition:np.asarray([r['counts'] for r in results if r['condition']==condition]).sum(axis=0)
                for condition in ('clean','degraded')}
        clean,degraded=rates(totals['clean']),rates(totals['degraded'])
        metrics={f'{c}_{k}':v for c,r in [('clean',clean),('degraded',degraded)] for k,v in r.items()}
        metrics['balanced_clean_degraded_hmean']=.5*(clean['hmean']+degraded['hmean'])
        return metrics


@HOOKS.register_module()
class FiniteLossAndInterruptCheckpointHook(Hook):
    priority='LOWEST'

    def __init__(self,accumulation):
        self.accumulation=accumulation;self.stop=False

    def before_train(self,runner):
        def request(signum,frame):self.stop=True
        for sig in stop_signals():signal.signal(sig,request)

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        for value in (outputs or {}).values():
            if isinstance(value,(int,float)) and not math.isfinite(value):raise RuntimeError('Non-finite training loss/log value')
        # Do not checkpoint midway through gradient accumulation: partial gradients are not serialized.
        if self.stop and (runner.iter+1)%self.accumulation==0:
            filename=f'checkpoint__interrupted_after_micro_iteration_{runner.iter+1}.pth'
            runner.save_checkpoint(runner.work_dir,filename,by_epoch=False,
                                   save_optimizer=True,save_param_scheduler=True)
            (Path(runner.work_dir)/'last_checkpoint').write_text(str(Path(runner.work_dir)/filename))
            runner.logger.info('Interruption checkpoint saved; rerun python run_all.py to resume.')
            raise SystemExit(75)
