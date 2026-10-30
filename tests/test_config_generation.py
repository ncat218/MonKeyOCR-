"""Optional CPU test with real MMEngine Config, without Torch/CUDA/model downloads."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

from test_core import REPO,tmp
from monkey_textdet.common import read_json
from monkey_textdet.inference import generate_inference_config
from monkey_textdet.training import make_training_config


@unittest.skipUnless(importlib.util.find_spec('mmengine'),'Optional: install MMEngine Config dependencies for this test')
class ConfigGeneration(unittest.TestCase):
    def test_real_config_dump_and_reload_preserves_architecture_and_equal_budgets(self):
        from mmengine.config import Config
        with tmp() as d:
            root=Path(d)
            # Some Windows sandbox TEMP directories permit mkdir but deny file creation.
            # MMEngine internally creates temporary config files; scope those to our writable fixture.
            previous_tempdir=tempfile.tempdir
            tempfile.tempdir=str(root.resolve())
            self.addCleanup(setattr,tempfile,'tempdir',previous_tempdir)
            source=root/'official_model_fixture.py'
            source.write_text('''
model=dict(type='DBNet',backbone=dict(type='MonkeyOCRv2ViTAEBackbone',model_dir='old',weights='old.safetensors',use_bf16=True,freeze=False),neck=dict(type='FPNC'),det_head=dict(type='DBHead'))
train_dataloader=dict(dataset=dict(pipeline=[dict(type='PackTextDetInputs',meta_keys=('img_path',))]))
val_dataloader=dict(dataset=dict(pipeline=[dict(type='PackTextDetInputs',meta_keys=('img_path','scale_factor'))]))
test_dataloader=dict(dataset=dict(pipeline=[dict(type='PackTextDetInputs',meta_keys=('img_path','scale_factor'))]))
''')
            cfg=read_json(REPO/'configs/phase3_h200_normal.json')
            annotations={k:str(root/(k+'.json')) for k in ['E1_clean_only','E2_clean_and_degraded_balanced','validation_clean_and_degraded','validation_exact']}
            generated=[]
            for name in ['E1_clean_only','E2_clean_and_degraded_balanced']:
                out=root/(name+'.py')
                make_training_config(source,root/'backbone',root/'original.pth',annotations,root/'protocol.json',root/name,cfg,name,out)
                generated.append(Config.fromfile(str(out),import_custom_modules=False))
            e1,e2=generated
            self.assertEqual(e1.model,e2.model)
            self.assertEqual(e1.load_from,e2.load_from)
            self.assertEqual(e1.optim_wrapper,e2.optim_wrapper)
            self.assertEqual(e1.train_cfg,e2.train_cfg)
            self.assertEqual(e1.val_evaluator,e2.val_evaluator)
            self.assertNotEqual(e1.train_dataloader.dataset.ann_file,e2.train_dataloader.dataset.ann_file)
            self.assertIn('sample_id',e1.val_dataloader.dataset.pipeline[0]['meta_keys'])
            self.assertIsNone(e1.test_dataloader)
            infer=root/'native_inference.py'
            generate_inference_config(source,root/'backbone',infer)
            native=Config.fromfile(str(infer))
            self.assertNotIn('sample_id',native.test_dataloader.dataset.pipeline[0]['meta_keys'])
            self.assertEqual(native.model,e1.model)


if __name__=='__main__':unittest.main()
