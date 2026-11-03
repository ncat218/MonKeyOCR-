#!/usr/bin/env python3
"""Remove only downloaded Conda archives, verified Phase3 exports and failed partial JSONL.

Standard library only. Never removes environments, extracted Conda packages,
datasets, checkpoints, completed manifests or Phase3 result directories.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import zipfile

ARCHIVES_BACKED_UP_ON_WINDOWS = {
    'phase3_dbnet_monkeyocrv2__textvqa_clean_degraded__E0_global_2x2_E1_clean_E2_mixed__results_job7957.zip':
        'be21a45302686e64c0e5c0528c663d5987a68782fd0ba228a936e125b3d266e3',
    'phase3_dbnet_monkeyocrv2__E0_2x2_E1_E2__per_image_metrics_for_size_corruption_errors__job7957.zip':
        'da63a3db6f5ceedd8719f40124b776f6142b338d9eff5b17ca3849f851f8f482',
}
DEBUG_ZIP = 'phase3_manifest_debug_job7845.zip'
DEBUG_MEMBERS_BACKED_UP_ON_WINDOWS = {
    'shared_manifest_v1.jsonl': 'ad43fa08d905fb5e50a6690a34b29da4afa6f19475a9afda4a375ea0171c7fb5',
    'data_audit.json': 'dd17d67caa11c59ba830b48f96f2598e51e93a02f300dd7cc82221c3f0609bf8',
}
FAILED_V1 = 'phase4_FULL_R1_official_train_GT_intersection_DBNet_vs_ASF_mixed_10000updates_mask_validation_seed42_v1'


def stream_sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        h.update(block)
    return h.hexdigest()


def file_sha(path):
    with path.open('rb') as f:
        return stream_sha(f)


def contained_file(workspace, path):
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(workspace):
        return False
    current = path.parent
    while current != workspace:
        if current.is_symlink() or current == current.parent:
            return False
        current = current.parent
    return True


def debug_export_is_backed_up(path):
    """Only inspect members; never extract ZIP paths onto the filesystem."""
    try:
        with zipfile.ZipFile(path) as z:
            for base, expected in DEBUG_MEMBERS_BACKED_UP_ON_WINDOWS.items():
                matches = [n for n in z.namelist() if Path(n).name == base]
                if len(matches) != 1:
                    return False
                with z.open(matches[0]) as f:
                    if stream_sha(f) != expected:
                        return False
        return True
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def collect_plan(workspace):
    workspace = Path(workspace).resolve(strict=True)
    plan = []
    def add(path, reason):
        if contained_file(workspace, path):
            st = path.stat()
            plan.append(dict(path=str(path), reason=reason, size=st.st_size,
                             mtime_ns=st.st_mtime_ns, inode=st.st_ino, sha256=file_sha(path)))
    cache = workspace/'environment_and_models__phase1_verified_stack/conda_packages__private_cuda128'
    for p in sorted(cache.glob('*')):
        if p.name.endswith(('.conda', '.tar.bz2')):
            add(p, 'Downloaded compressed Conda package; extracted packages and installed CUDA retained')
    for name, digest in ARCHIVES_BACKED_UP_ON_WINDOWS.items():
        p = workspace/name
        if contained_file(workspace, p) and file_sha(p) == digest:
            add(p, 'Phase3 export ZIP verified against the saved Windows copy')
        elif p.exists():
            print('KEEP (backup hash does not match):', p)
    debug = workspace/DEBUG_ZIP
    if contained_file(workspace, debug) and debug_export_is_backed_up(debug):
        add(debug, 'Old manifest debug export; master manifest and audit verified against extracted Windows backups')
    elif debug.exists():
        print('KEEP (debug backup members do not match):', debug)
    failed = workspace/FAILED_V1
    markers = sorted((failed/'state').glob('*__stage_status.json'))
    allowed_marker = '01_audit_full_MAIN_and_freeze_official_train_GT_intersection__stage_status.json'
    if len(markers) == 1 and markers[0].name == allowed_marker:
        state = json.loads(markers[0].read_text(encoding='utf-8'))
        if state.get('status') == 'interrupted_or_failed' and 'Disk quota exceeded' in state.get('error', ''):
            if not any(p.is_file() for p in (failed/'outputs').rglob('*')):
                folder = failed/'inputs/02_shared_FULL_R1_official_train_TextVQA_TextOCR_word_GT'
                for p in folder.glob('*.writing'):
                    add(p, 'Incomplete manifest/metadata from failed job9146; not a completed input')
    return workspace, plan


def apply_plan(workspace, plan):
    removed = []
    for record in plan:
        p = Path(record['path'])
        if not contained_file(workspace, p):
            raise RuntimeError('Unsafe cleanup path: '+str(p))
        st = p.stat()
        if (st.st_size, st.st_mtime_ns, st.st_ino) != (record['size'], record['mtime_ns'], record['inode']):
            raise RuntimeError('File changed after cleanup inspection: '+str(p))
        if file_sha(p) != record['sha256']:
            raise RuntimeError('File hash changed after cleanup inspection: '+str(p))
        p.unlink()
        removed.append(record)
        print('REMOVED:', p, '| bytes:', record['size'], flush=True)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Apply the bounded cleanup; otherwise print a plan only')
    args = parser.parse_args()
    workspace = args.workspace.resolve(strict=True)
    with contextlib.ExitStack() as stack:
        # The bootstrap holds this lock for the whole allocation. Refuse to
        # touch partial files while either old or revised Phase4 is running.
        if args.apply and os.name == 'posix':
            import fcntl
            for root in workspace.glob('phase4_FULL_R1_*'):
                lock = root/'run_all__exclusive_process.lock'
                if lock.is_file():
                    f = stack.enter_context(lock.open('rb'))
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as exc:
                        raise SystemExit('A Phase4 controller is active; wait before cleanup: '+str(root)) from exc
        workspace, plan = collect_plan(workspace)
        total = sum(p['size'] for p in plan)
        print('CLEANUP PLAN:', len(plan), 'files | logical GiB:', round(total/1024**3, 3))
        for p in plan:
            print(p['reason'], '|', p['path'])
        removed = apply_plan(workspace, plan) if args.apply else []
        print('APPLIED' if args.apply else 'DRY RUN; pass --apply to delete these files')
        print('Logical bytes removed:', sum(p['size'] for p in removed),
              '(actual quota relief can differ for hard-linked/open files)')
        if args.apply:
            receipt = workspace/'CLEANUP_RECEIPT__Conda_download_archives_verified_Phase3_ZIPs_and_failed_partial_manifest.json'
            prior = json.loads(receipt.read_text()) if receipt.is_file() else []
            prior.append(dict(removed=removed, logical_bytes_removed=sum(p['size'] for p in removed)))
            receipt.write_text(json.dumps(prior, indent=2)+'\n', encoding='utf-8')
            print('RECEIPT:', receipt)


if __name__ == '__main__':
    main()
