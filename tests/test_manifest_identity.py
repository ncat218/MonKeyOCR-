import contextlib
import copy
import io
import json
import unittest

from test_core import REPO, row, tmp
from monkey_textdet.common import canonical, read_json, save_json, sha256
from monkey_textdet.manifest_identity import audit_identity, compare_manifest_streams, verify_phase2_manifest


class ManifestIdentityTests(unittest.TestCase):
    def fixture(self, folder):
        from pathlib import Path
        folder = Path(folder)
        manifest = folder / 'shared_manifest_v1.jsonl'
        manifest.write_bytes(canonical(row()))
        digest = sha256(manifest)
        audit = dict(manifest_sha256=digest, counts=dict(evaluated_unique_sources=1),
                     textocr_sha256='a' * 64)
        save_json(folder / 'data_audit.json', audit)
        registry = dict(reference_manifest_sha256=digest,
                        reference_data_audit_identity_sha256=audit_identity(audit), reviewed_variants={})
        path = folder / 'registry.json'
        save_json(path, registry)
        return manifest, audit, registry, path

    def test_exact_reference_is_verified_and_receipt_is_written(self):
        with tmp() as d, contextlib.redirect_stdout(io.StringIO()):
            manifest, audit, registry, path = self.fixture(d)
            result = verify_phase2_manifest(manifest.parent,
                dict(phase2_manifest_sha256=registry['reference_manifest_sha256']), path)
            self.assertEqual(result['mode'], 'byte_exact_phase2_reference')
            self.assertTrue((manifest.parent / 'phase2_manifest_identity_verification.json').is_file())

    def test_only_exact_reviewed_variant_passes_and_does_not_rewrite_gt(self):
        with tmp() as d, contextlib.redirect_stdout(io.StringIO()):
            manifest, audit, registry, path = self.fixture(d)
            r = row(); r['gt_clean']['valid'][0]['visibility'] = 1. - 1e-15
            manifest.write_bytes(canonical(r)); actual = sha256(manifest)
            audit['manifest_sha256'] = actual; save_json(manifest.parent / 'data_audit.json', audit)
            registry['reviewed_variants'][actual] = dict(review='fixture numerical comparison')
            save_json(path, registry)
            result = verify_phase2_manifest(manifest.parent,
                dict(phase2_manifest_sha256=registry['reference_manifest_sha256']), path)
            self.assertEqual(result['mode'], 'byte_exact_reviewed_numerical_variant')
            self.assertEqual(sha256(manifest), actual)
            r['source_image_id'] = 'different_source'; manifest.write_bytes(canonical(r))
            audit['manifest_sha256'] = sha256(manifest); save_json(manifest.parent / 'data_audit.json', audit)
            with self.assertRaisesRegex(RuntimeError, 'unreviewed SHA256'):
                verify_phase2_manifest(manifest.parent,
                    dict(phase2_manifest_sha256=registry['reference_manifest_sha256']), path)

    def test_audit_metadata_change_is_rejected_even_with_approved_manifest_hash(self):
        with tmp() as d:
            manifest, audit, registry, path = self.fixture(d)
            audit['textocr_sha256'] = 'b' * 64; save_json(manifest.parent / 'data_audit.json', audit)
            with self.assertRaisesRegex(RuntimeError, 'Dataset audit differs'):
                verify_phase2_manifest(manifest.parent,
                    dict(phase2_manifest_sha256=registry['reference_manifest_sha256']), path)

    def test_declared_audit_sha_cannot_replace_actual_manifest_hash(self):
        with tmp() as d:
            manifest, audit, registry, path = self.fixture(d)
            manifest.write_bytes(canonical(row(source='changed')))
            with self.assertRaisesRegex(RuntimeError, 'bytes do not match'):
                verify_phase2_manifest(manifest.parent,
                    dict(phase2_manifest_sha256=registry['reference_manifest_sha256']), path)

    def test_review_allows_only_bounded_geometry_roundoff(self):
        a = json.loads(canonical(row())); b = copy.deepcopy(a)
        b['gt_clean']['valid'][0]['geometry']['coordinates'][0][0][0] += 1e-12
        result = compare_manifest_streams(io.BytesIO(canonical(a)), io.BytesIO(canonical(b)))
        self.assertEqual(result['byte_different_rows'], 1)
        self.assertEqual(result['rows'], 1)

    def test_review_rejects_changed_identity_or_material_geometry_or_list_length(self):
        a = json.loads(canonical(row()))
        for change in ('identity', 'geometry', 'annotations', 'image_hash', 'float_type'):
            b = copy.deepcopy(a)
            if change == 'identity': b['source_image_id'] = 'another_source'
            elif change == 'geometry': b['gt_clean']['valid'][0]['geometry']['coordinates'][0][0][0] += .001
            elif change == 'annotations': b['gt_clean']['valid'].append(copy.deepcopy(b['gt_clean']['valid'][0]))
            elif change == 'image_hash': b['clean_sha256'] = 'f' * 64
            else: b['gt_clean']['valid'][0]['visibility'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_manifest_streams(io.BytesIO(canonical(a)), io.BytesIO(canonical(b)))

    def test_review_rejects_reordered_or_missing_sources_and_nonfinite_values(self):
        a, b = canonical(row('a')), canonical(row('b'))
        for candidate in (b + a, a, a.replace(b'1.0', b'NaN', 1) + b):
            with self.assertRaises(ValueError):
                compare_manifest_streams(io.BytesIO(a + b), io.BytesIO(candidate))

    def test_review_report_binds_registered_production_variant(self):
        locks = read_json(REPO / 'configs/verified_phase1_phase2_sources.json')
        registry = read_json(REPO / 'configs/verified_phase2_manifest_variants.json')
        report = read_json(REPO / 'docs/manifest_comparison__phase2_vs_hpc_job7845.json')
        self.assertEqual(report['reference_manifest_sha256'], locks['phase2_manifest_sha256'])
        variant = registry['reviewed_variants'][report['candidate_manifest_sha256']]
        self.assertEqual(variant['comparison_report_sha256'],
                         sha256(REPO / 'docs/manifest_comparison__phase2_vs_hpc_job7845.json'))
        self.assertEqual(report['rows'], locks['phase2_sources'])


if __name__ == '__main__':
    unittest.main()
