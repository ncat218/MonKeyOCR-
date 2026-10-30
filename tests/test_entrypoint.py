import contextlib
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from test_core import REPO,tmp

spec=importlib.util.spec_from_file_location('run_all_entrypoint',REPO/'run_all.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
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
