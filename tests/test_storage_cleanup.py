"""Deletion guards on disposable files only; never touch real HPC/local results."""
import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from test_core import REPO, tmp

spec = importlib.util.spec_from_file_location('safe_storage_cleanup', REPO/'scripts/cleanup_downloads_and_verified_phase3_zips.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class StorageCleanup(unittest.TestCase):
    def test_only_compressed_downloads_are_removed_installed_libraries_retained(self):
        with tmp() as d:
            root=Path(d).resolve()
            env=root/'environment_and_models__phase1_verified_stack'
            cache=env/'conda_packages__private_cuda128';cache.mkdir(parents=True)
            (cache/'download.conda').write_bytes(b'package archive fixture')
            extracted=cache/'cuda_extracted/bin';extracted.mkdir(parents=True)
            (extracted/'nvcc').write_bytes(b'installed runtime must stay')
            (cache/'urls.txt').write_text('package metadata must stay')
            workspace,plan=cleanup.collect_plan(root)
            self.assertEqual(len(plan),1)
            cleanup.apply_plan(workspace,plan)
            self.assertTrue((extracted/'nvcc').is_file())
            self.assertTrue((cache/'urls.txt').is_file())
            self.assertEqual(cleanup.collect_plan(root)[1],[])

    def test_phase3_export_requires_verified_backup_hash_unknown_ZIPs_retained(self):
        with tmp() as d:
            root=Path(d).resolve()
            archive=root/'phase3_export_fixture.zip';archive.write_bytes(b'fixture backed up elsewhere')
            other=root/'phase3_new_or_unknown.zip';other.write_bytes(b'keep')
            allow={archive.name:hashlib.sha256(archive.read_bytes()).hexdigest()}
            with patch.object(cleanup,'ARCHIVES_BACKED_UP_ON_WINDOWS',allow):
                _,plan=cleanup.collect_plan(root)
                self.assertEqual(len(plan),1)
                archive.write_bytes(b'modified after inspection')
                with self.assertRaisesRegex(RuntimeError,'changed after cleanup'):
                    cleanup.apply_plan(root,plan)
                self.assertTrue(archive.exists())
                self.assertTrue(other.exists())
                self.assertEqual(cleanup.collect_plan(root)[1],[])

    def test_failed_partial_cleanup_never_removes_completed_manifest(self):
        with tmp() as d:
            root=Path(d).resolve();run=root/cleanup.FAILED_V1
            state=run/'state';state.mkdir(parents=True)
            marker=state/'01_audit_full_MAIN_and_freeze_official_train_GT_intersection__stage_status.json'
            marker.write_text(json.dumps(dict(status='interrupted_or_failed',error='OSError: Disk quota exceeded')))
            folder=run/'inputs/02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT';folder.mkdir(parents=True)
            partial=folder/'shared_manifest.jsonl.writing';partial.write_bytes(b'partial')
            complete=folder/'shared_manifest.jsonl';complete.write_bytes(b'completed evidence')
            _,plan=cleanup.collect_plan(root)
            self.assertEqual([Path(p['path']).name for p in plan],[partial.name])
            marker.write_text(json.dumps(dict(status='completed')))
            self.assertEqual(cleanup.collect_plan(root)[1],[])
            self.assertTrue(complete.exists())

    def test_symlink_cannot_redirect_cleanup_outside_workspace(self):
        with tmp() as d:
            root=Path(d).resolve();workspace=root/'work';workspace.mkdir()
            outside=root/'outside.conda';outside.write_bytes(b'not authorized')
            cache=workspace/'environment_and_models__phase1_verified_stack/conda_packages__private_cuda128'
            cache.mkdir(parents=True)
            link=cache/'symlink.conda'
            try:
                link.symlink_to(outside)
            except OSError:
                self.skipTest('Host does not permit creating symlinks')
            self.assertEqual(cleanup.collect_plan(workspace)[1],[])
            self.assertTrue(outside.is_file())


if __name__=='__main__':
    unittest.main()
