"""Standard-library tests: evidence parsing and bounded, read-only collection."""
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'asf_diagnostic_collection', REPO / 'scripts/collect_full_r1_asf_training_diagnostics.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class DiagnosticCollection(unittest.TestCase):
    def temporary(self):
        parent = REPO / '_workspace_cpu_tests'
        parent.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent)

    def fixture(self, root):
        workspace = root / 'workspace'
        run = workspace / collector.RUN_NAME
        configs = run / 'inputs/05_model_configs__only_FPNC_ASF_differs'
        configs.mkdir(parents=True)
        for variant in collector.VARIANTS:
            receipt = dict(variant=variant, strict_full_state_load_passed=True,
                           shared_weight_equality_passed=True,
                           newly_initialized_keys=['neck.asf_attn.weight'] if variant.startswith('B_') else [])
            (configs / ('INITIALIZATION_RECEIPT__' + variant + '.json')).write_text(json.dumps(receipt))
            training = run / 'outputs/training' / variant
            training.mkdir(parents=True)
            (training / 'scalars.json').write_text(json.dumps({
                'step': 4000, 'shared/clean_hmean': .6,
                'shared/degraded_hmean': .4, 'shared/balanced_clean_degraded_hmean': .5}) + '\n')
            (training / 'COMPATIBILITY__probe.json').write_text(json.dumps(dict(
                ASF_parameter_tensors_with_finite_gradients=6 if variant.startswith('B_') else 0)))
            (training / 'FINAL_WEIGHTS_ONLY.pth').write_bytes(b'must never be archived or modified')
        return workspace, run

    def test_text_and_scalar_axes_are_not_silently_converted_to_updates(self):
        text = ('INFO Iter(train) [ 3999/20000] loss: 1.0\n'
                'INFO Iter(val) [6524/6524] shared/clean_hmean: 0.6 '
                'shared/degraded_hmean: 4e-1 shared/balanced_clean_degraded_hmean: 0.5\n')
        row = collector.validation_rows(text, 'B', 'training.log')[0]
        self.assertIsNone(row['raw_logger_step'])
        self.assertEqual(row['last_train_microiteration_in_this_text'], 3999)
        self.assertEqual(row['degraded_hmean'], .4)
        scalar = json.dumps({'step': 4000, 'shared/clean_hmean': .6, 'shared/degraded_hmean': .4})
        row = collector.validation_rows(scalar, 'B', 'scalars.json')[0]
        self.assertEqual(row['raw_logger_step'], 4000)
        self.assertIsNone(row['last_train_microiteration_in_this_text'])

    def test_invalid_metrics_and_balanced_only_records_are_not_mistaken_for_clean(self):
        text = '\n'.join([json.dumps({'shared/balanced_clean_degraded_hmean': .5}),
                          json.dumps({'clean_hmean': 60, 'degraded_hmean': 40}),
                          json.dumps({'clean_hmean': float('nan'), 'degraded_hmean': .4})])
        self.assertEqual(collector.validation_rows(text, 'A', 'fixture'), [])

    def test_bundle_preserves_weights_excludes_datasets_and_reports_actual_evidence(self):
        with self.temporary() as temp:
            root = Path(temp)
            workspace, run = self.fixture(root)
            (run / 'inputs/huge_manifest.jsonl').write_bytes(b'dataset must stay out')
            weights = list(run.rglob('*.pth'))
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in weights}
            output, manifest = collector.collect(workspace)
            with zipfile.ZipFile(output) as z:
                self.assertIsNone(z.testzip())
                self.assertFalse(any(n.endswith(('.pth', '.jsonl')) for n in z.namelist()))
                self.assertIn('VALIDATION_CURVES__raw_steps_and_source_lines.csv', z.namelist())
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in weights})
            b = manifest['recorded_evidence'][collector.VARIANTS[1]]
            self.assertTrue(b['shared_weights_equal_recorded'])
            self.assertEqual(b['ASF_tensors_with_finite_probe_gradients'], 6)
            self.assertEqual(b['validation_records_parsed_including_duplicate_log_formats'], 1)

    def test_no_overwrite_no_output_inside_experiment_and_no_run_path_escape(self):
        with self.temporary() as temp:
            workspace, run = self.fixture(Path(temp))
            output = workspace / 'existing.zip'
            output.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):
                collector.collect(workspace, output=output)
            self.assertEqual(output.read_bytes(), b'keep')
            with self.assertRaises(ValueError):
                collector.collect(workspace, output=run / 'new.zip')
            with self.assertRaises(ValueError):
                collector.collect(workspace, run_name='../workspace')

    def test_large_log_is_marked_as_tail_and_large_json_is_explicitly_skipped(self):
        with self.temporary() as temp:
            workspace, run = self.fixture(Path(temp))
            logs = run / 'logs'
            logs.mkdir()
            (logs / 'long.log').write_bytes(b'x' * 1024)
            with patch.object(collector, 'MAX_FILE', 128):
                output, manifest = collector.collect(workspace)
            entry = next(e for e in manifest['entries'] if e['archive_name'].endswith('long.log'))
            self.assertTrue(entry['truncated'])
            self.assertEqual(entry['original_bytes'], 1024)
            self.assertEqual(entry['included_bytes'], 128)
            self.assertTrue(any('Skipped oversized non-log' in w for w in manifest['warnings']))
            self.assertTrue(output.is_file())


if __name__ == '__main__':
    unittest.main()
