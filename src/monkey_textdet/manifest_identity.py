"""Exact hashes of reviewed manifests; numerical comparison is an offline audit only.

Runtime never accepts an arbitrary manifest by rounding or ignoring coordinates.
Each additional byte variant must first be compared against the original Phase 2
manifest, reviewed, and registered by its complete SHA256.
"""
import collections
import hashlib
import itertools
import json
import math
from pathlib import Path

from .common import identity, read_json, save_json, sha256

REGISTRY = Path(__file__).resolve().parents[2] / 'configs/verified_phase2_manifest_variants.json'


def audit_identity(audit):
    return identity({k: v for k, v in audit.items() if k != 'manifest_sha256'})


def verify_phase2_manifest(folder, locks, registry_path=REGISTRY):
    folder = Path(folder)
    registry = read_json(registry_path)
    expected = locks['phase2_manifest_sha256']
    if registry['reference_manifest_sha256'] != expected:
        raise RuntimeError('Manifest review registry does not bind the pinned Phase 2 reference')
    actual = sha256(folder / 'shared_manifest_v1.jsonl')
    audit = read_json(folder / 'data_audit.json')
    if audit['manifest_sha256'] != actual:
        raise RuntimeError('Manifest bytes do not match their data audit SHA256')
    if audit_identity(audit) != registry['reference_data_audit_identity_sha256']:
        raise RuntimeError('Dataset audit differs from Phase 2: inspect source hashes, schema and counts')
    if actual == expected:
        mode = 'byte_exact_phase2_reference'
        review = None
    elif actual in registry['reviewed_variants']:
        mode = 'byte_exact_reviewed_numerical_variant'
        review = registry['reviewed_variants'][actual]
    else:
        raise RuntimeError(
            'Rebuilt master manifest has an unreviewed SHA256: ' + actual + '. '
            'Compare shared_manifest_v1.jsonl and data_audit.json against Phase 2; '
            'do not disable the lock. Unknown numerical variants also require review.')
    receipt = dict(mode=mode, actual_manifest_sha256=actual,
                   reference_manifest_sha256=expected,
                   audit_identity_sha256=audit_identity(audit), review=review,
                   registry_sha256=sha256(registry_path),
                   gt_policy='Use these frozen manifest bytes unchanged for every Phase 3 method and Paddle')
    save_json(folder / 'phase2_manifest_identity_verification.json', receipt)
    print('VERIFIED PHASE 2 MANIFEST IDENTITY:', mode, actual, flush=True)
    return receipt


def numerical_tolerance(path):
    """Absolute bounds used ONLY to review a candidate, not as runtime acceptance."""
    if path and path[0] == 'homography' and all(isinstance(x, int) for x in path[1:]):
        return 1e-14
    if path[:2] in (('roi_clean', 'coordinates'), ('roi_degraded', 'coordinates')):
        return 1e-9 if all(isinstance(x, int) for x in path[2:]) else None
    if (len(path) >= 4 and path[0] in ('gt_clean', 'gt_degraded')
            and path[1] in ('valid', 'ignored') and isinstance(path[2], int)):
        if path[3:] == ('visibility',):
            return 1e-12
        if path[3:5] == ('geometry', 'coordinates') and all(isinstance(x, int) for x in path[5:]):
            return 1e-9
    return None


def compare_manifest_streams(reference, candidate):
    """Bounded-memory comparison of every ordered row, field, type and number."""
    stats = collections.defaultdict(lambda: dict(changed_numbers=0, max_absolute_difference=0.))

    def compare(a, b, path=()):
        if type(a) is not type(b):
            raise ValueError(f'Value type changed at {path}')
        if isinstance(a, dict):
            if a.keys() != b.keys():
                raise ValueError(f'Object fields changed at {path}')
            for key in a:
                compare(a[key], b[key], path + (key,))
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f'List length changed at {path}')
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + (i,))
        elif isinstance(a, float) and (not math.isfinite(a) or not math.isfinite(b)):
            raise ValueError(f'Non-finite value at {path}')
        elif a != b:
            tolerance = numerical_tolerance(path)
            if type(a) is not float or tolerance is None or abs(a - b) > tolerance:
                raise ValueError(f'Non-numerical or excessive change at {path}')
            field = '.'.join('[]' if isinstance(x, int) else x for x in path)
            stat = stats[field]
            stat['changed_numbers'] += 1
            stat['max_absolute_difference'] = max(stat['max_absolute_difference'], abs(a - b))

    def reject_constant(value):
        raise ValueError('Non-finite JSON constant: ' + value)

    old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
    rows = changed_rows = 0
    for rows, (old, new) in enumerate(itertools.zip_longest(reference, candidate), 1):
        if old is None or new is None:
            raise ValueError('Manifest row count changed')
        old_hash.update(old)
        new_hash.update(new)
        # Equal bytes need no numerical walk. Reject non-finite JSON constants.
        a = json.loads(old, parse_constant=reject_constant)
        if old != new:
            compare(a, json.loads(new, parse_constant=reject_constant))
        changed_rows += old != new
    if not rows:
        raise ValueError('Empty manifest')
    return dict(reference_manifest_sha256=old_hash.hexdigest(),
                candidate_manifest_sha256=new_hash.hexdigest(), rows=rows,
                byte_different_rows=changed_rows, numerical_differences=dict(stats),
                all_other_values_types_structure_and_row_order='exactly_equal',
                offline_absolute_bounds=dict(geometry_coordinates=1e-9,
                                             homography=1e-14, visibility=1e-12))
