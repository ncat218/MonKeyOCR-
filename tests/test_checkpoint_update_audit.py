"""CPU array arithmetic and input guards; real Torch checkpoint execution is on HPC."""
import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('checkpoint_update_audit', REPO/'scripts/audit_full_r1_checkpoint_updates_cpu.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CheckpointUpdateAudit(unittest.TestCase):
    def test_initial_dtype_rounding_is_not_reported_as_training_update(self):
        source = np.array([1.001, 2.003], dtype=np.float32)
        baseline = np.array([1., 2.], dtype=np.float32)
        stats = audit.delta_stats(source, baseline, baseline)
        self.assertEqual(stats['initial_cast_changed_elements'], 2)
        self.assertEqual(stats['changed_elements_vs_raw_source'], 2)
        self.assertEqual(stats['changed_elements_after_initial_cast'], 0)

    def test_training_changes_are_counted_relative_to_load_cast(self):
        stats = audit.delta_stats([1., 2., 3.], [1., 2., 3.], [1., 2.5, 2.])
        self.assertEqual(stats['changed_elements_after_initial_cast'], 2)
        self.assertEqual(stats['sum_abs_delta_after_initial_cast'], 1.5)
        self.assertEqual(stats['max_abs_delta_after_initial_cast'], 1.)

    def test_nonfinite_or_mismatched_state_is_not_silently_compared(self):
        for a,b,c in [([1.], [1., 2.], [1.]), ([1.], [1.], [float('nan')]), ([], [], [])]:
            with self.assertRaises(ValueError):
                audit.delta_stats(a,b,c)

    def test_ASF_and_remaining_neck_are_disjoint(self):
        self.assertEqual(audit.group_for('neck.asf_attn.x'), 'ASF')
        self.assertEqual(audit.group_for('neck.lateral_convs.x'), 'neck_without_ASF')
        self.assertEqual(audit.group_for('backbone.vitae.blocks.0.weight'), 'backbone_vitae')

    def test_hash_and_scope_checks_precede_deserialization_and_config_is_not_executed(self):
        parent = REPO/'_workspace_cpu_tests'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root = Path(temp)
            p = root/'fixture.pth'
            p.write_bytes(b'not a real checkpoint')
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(audit.verified_file(p, root, digest), p.resolve())
            with self.assertRaises(ValueError):
                audit.verified_file(p, root, '0'*64)
            with self.assertRaises(ValueError):
                audit.verified_file(p, root/'different_root', digest)
            cfg = root/'config.py'
            cfg.write_text("raise RuntimeError('must never execute')\nload_from = '/recorded/file.pth'\n")
            self.assertEqual(audit.initial_path(cfg), Path('/recorded/file.pth'))


if __name__ == '__main__':
    unittest.main()
