import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from test_core import REPO,tmp

spec=importlib.util.spec_from_file_location('run_all_entrypoint',REPO/'run_all.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
    def test_full_suite_plan_needs_only_standard_library_on_login_python(self):
        proc = subprocess.run([sys.executable, '-S', str(REPO/'run_all.py'), '--config',
            str(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json'), '--plan-only'],
            capture_output=True, text=True, check=True)
        plan = json.loads(proc.stdout)
        self.assertEqual(plan['configuration']['training']['optimizer_updates'], 10000)

    def test_full_suite_checks_metadata_then_submits_one_named_job(self):
        config = str(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json')
        with tmp() as d:
            with patch.object(sys,'argv',['run_all.py','--config',config,'--workspace',d,'--dataset-parent',d]), \
                 patch.dict(os.environ,{},clear=True), patch('shutil.which',return_value='/usr/bin/sbatch'), \
                 patch('monkey_textdet.data.discover_inputs',return_value={'fixture':'path check'}), \
                 patch('monkey_textdet.full_data.read_metadata',return_value=([],{'main_counts_by_official_split':{'train':12}})) as audit, \
                 patch('subprocess.check_output',return_value='9000\n') as submit, contextlib.redirect_stdout(io.StringIO()):
                entry.main()
            audit.assert_called_once()
            submit.assert_called_once()
            command=submit.call_args.args[0]
            self.assertIn('--job-name=monkey_FULL_R1_DBNet_ASF',command)
            self.assertTrue(any('FULL_R1_DBNet_vs_ASF' in c and 'stdout.log' in c for c in command))
            self.assertIn('phase4_full_textvqa_dbnet_vs_asf_h200_normal.json',Path(command[-1]).read_text())

    def test_full_bad_metadata_never_spends_a_GPU_submission(self):
        config = str(REPO/'configs/phase4_full_textvqa_dbnet_vs_asf_h200_normal.json')
        with tmp() as d:
            with patch.object(sys,'argv',['run_all.py','--config',config,'--workspace',d,'--dataset-parent',d]), \
                 patch.dict(os.environ,{},clear=True), patch('shutil.which',return_value='/usr/bin/sbatch'), \
                 patch('monkey_textdet.data.discover_inputs',return_value={'fixture':'path check'}), \
                 patch('monkey_textdet.full_data.read_metadata',side_effect=ValueError('wrong generator hash')), \
                 patch('subprocess.check_output') as submit:
                with self.assertRaisesRegex(ValueError,'wrong generator hash'):
                    entry.main()
            submit.assert_not_called()

    def test_plan_does_not_submit_or_bootstrap(self):
        capture=io.StringIO()
        with patch.object(sys,'argv',['run_all.py','--plan-only']),patch('subprocess.check_output') as call,contextlib.redirect_stdout(capture):
            entry.main()
        call.assert_not_called()
        self.assertEqual(json.loads(capture.getvalue())['configuration']['scheduler']['partition'],'normal')

    def test_one_command_submits_exactly_one_job_containing_all_experiments(self):
        with tmp() as d:
            with patch.object(sys,'argv',['run_all.py','--workspace',d,'--dataset-parent','/datastore/user/dataset parent']),\
                 patch.dict(os.environ,{},clear=True),patch('shutil.which',return_value='/usr/bin/sbatch'),\
                 patch('monkey_textdet.data.discover_inputs',return_value={'fixture':'path check only'}),\
                 patch('subprocess.check_output',return_value='12345\n') as submit,contextlib.redirect_stdout(io.StringIO()):
                entry.main()
            self.assertEqual(submit.call_count,1)
            args=submit.call_args.args[0]
            self.assertIn('--partition=normal',args)
            self.assertIn('--gres=gpu:2g.35gb:1',args)
            self.assertIn('--qos=mig2-35',args)
            script=Path(args[-1]).read_text()
            self.assertIn('--in-allocation',script)
            self.assertIn("'/datastore/user/dataset parent'",script)
            self.assertNotIn('python -m monkey_textdet.training',script)

    def test_missing_dataset_metadata_is_rejected_before_gpu_job_submission(self):
        with tmp() as d:
            with patch.object(sys,'argv',['run_all.py','--workspace',d,'--dataset-parent',d]),\
                 patch.dict(os.environ,{},clear=True),patch('shutil.which',return_value='/usr/bin/sbatch'),\
                 patch('subprocess.check_output') as submit:
                with self.assertRaisesRegex(RuntimeError,'clean_json'):
                    entry.main()
            submit.assert_not_called()


if __name__=='__main__':unittest.main()
