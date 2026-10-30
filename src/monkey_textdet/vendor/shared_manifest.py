"""Public, detector-independent intersection cell. No model, training or degradation generation.

Supported input metadata schema: the four v2_10k generator notebooks supplied by Tuan.
All paths are discovered from real files; source IDs come from JSON/CSV, never degraded filenames.
The frozen manifest is root-independent. Copy the resulting bytes to Paddle; do not rebuild it.
"""
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box, mapping, shape
try:
    from shapely import transform as array_geometry_transform
except ImportError:
    array_geometry_transform = None
from shapely.ops import transform as legacy_geometry_transform

SCHEMA_VERSION = 'textdet-paired-v1'
GENERATOR_PROTOCOL = 'TEXTVQA_REALISTIC_DEGRADATION_V2_10K'
KNOWN_OPERATIONS = {'motion_camera_shake', 'defocus_blur', 'lowlight_sensor_noise',
                    'overexposure_glare', 'low_resolution', 'perspective', 'jpeg_compression'}
DEFAULT_SHARED_CONFIG = dict(
    input_root='/kaggle/input', output_root='/kaggle/working/textdet_shared_v1',
    clean_json='', textocr_json='', main_image_manifest='',
    clean_images_root='', degraded_root='',
    ignore_transcripts=['.', '###', ''], min_word_visibility=0.80,
    min_prediction_roi_fraction=0.50, ignore_prediction_overlap=0.50,
    match_iou=0.50, coverage_threshold=0.50, overlay_sources=12,
)


def canonical_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode('utf-8')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def write_locked(path, data):
    path = Path(path)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f'Frozen artifact changed: {path}. Use a new output revision; do not overwrite.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def describe_textocr_schema(raw):
    """Small diagnostics, not the entire large annotation JSON or transcripts."""
    report = dict(root_type=type(raw).__name__)
    if isinstance(raw, dict):
        report['top_level_keys'] = sorted(raw.keys())
        fields = {}
        for key, value in raw.items():
            field = dict(type=type(value).__name__)
            if isinstance(value, (dict, list)):
                field['length'] = len(value)
                example = next(iter(value.values()), None) if isinstance(value, dict) else next(iter(value), None)
                if isinstance(example, dict):
                    field['record_keys'] = sorted(example.keys())
                elif example is not None:
                    field['element_type'] = type(example).__name__
            fields[key] = field
        report['fields'] = fields
    return report


def normalize_textocr(raw):
    """Canonicalize explicit IDs and rebuild the redundant image->annotations index.

    The source of the mapping is each annotation's image_id, NOT filenames, list positions,
    or a guessed association. Polygon points remain unchanged. Existing usable indexes
    are checked against these foreign keys before their order is retained.
    """
    report = describe_textocr_schema(raw)
    if not isinstance(raw, dict):
        raise ValueError(f'TextOCR must be an object: {report}')
    def choose(primary, alias):
        if primary in raw:
            return primary, raw[primary]
        if alias in raw:
            return alias, raw[alias]
        raise ValueError(f'Missing {primary}/{alias}; actual keys={report.get("top_level_keys")}')
    image_key, image_records = choose('imgs', 'images')
    annotation_key, annotation_records = choose('anns', 'annotations')
    def canonical_id(value, label):
        if isinstance(value, bool) or not isinstance(value, (str, int)) or value == '':
            raise ValueError(f'{label}: missing/non-scalar explicit ID: {value!r}')
        return str(value)  # Do not strip leading zeroes, change case, or parse filename stems.
    def index_records(records, label):
        if not isinstance(records, (dict, list)):
            raise ValueError(f'{label} must be dict/list; actual type={type(records).__name__}')
        result = {}
        iterator = records.items() if isinstance(records, dict) else ((None, r) for r in records)
        for key, record in iterator:
            if not isinstance(record, dict):
                raise ValueError(f'{label}: expected object records, got {type(record).__name__}')
            item_id = canonical_id(record.get('id') if key is None else key, label)
            if 'id' in record and canonical_id(record['id'], label) != item_id:
                raise ValueError(f'{label}: dictionary key disagrees with record.id for {item_id}')
            if item_id in result:
                raise ValueError(f'{label}: duplicate explicit ID {item_id}')
            result[item_id] = record
        return result
    images = index_records(image_records, image_key)
    annotations = index_records(annotation_records, annotation_key)
    if not images or not annotations:
        raise ValueError('TextOCR images/annotations are empty; test metadata without GT is not usable here')
    rebuilt = defaultdict(list)
    for ann_id, ann in annotations.items():
        image_id = canonical_id(ann.get('image_id'), f'annotation {ann_id}.image_id')
        if image_id not in images:
            raise ValueError(f'Annotation {ann_id} references unknown image_id {image_id}')
        if 'points' not in ann:
            raise ValueError(f'Annotation {ann_id} has no polygon points; actual record keys={sorted(ann)}. '
                             'Do not substitute bbox or a recognition/VQA annotation as polygon GT.')
        rebuilt[image_id].append(ann_id)
    provided_keys = [key for key in ('img2Anns', 'imgToAnns') if key in raw]
    selected_index, validated_keys = None, []
    unusable_index_types = {}
    for key in provided_keys:
        supplied = raw[key]
        if not isinstance(supplied, dict):
            # This is only a redundant index; explicit annotation foreign keys remain authoritative.
            unusable_index_types[key] = type(supplied).__name__
            continue
        normalized = {}
        for image_id, ids in supplied.items():
            image_id = canonical_id(image_id, key)
            if image_id not in images or not isinstance(ids, list):
                raise ValueError(f'{key}: unknown image ID or non-list annotation IDs for {image_id}')
            ann_ids = [canonical_id(item, key) for item in ids]
            if len(set(ann_ids)) != len(ann_ids):
                raise ValueError(f'{key}: duplicate annotation references for {image_id}')
            if set(ann_ids) != set(rebuilt.get(image_id, [])):
                raise ValueError(f'{key}: links disagree with annotations.image_id for {image_id}; do not guess mapping')
            normalized[image_id] = ann_ids
        if any(set(normalized.get(image_id, [])) != set(ids) for image_id, ids in rebuilt.items()):
            raise ValueError(f'{key}: index omits annotations present in anns; inconsistent GT needs audit')
        if selected_index is None:
            selected_index = normalized
        validated_keys.append(key)
    if selected_index is None:
        selected_index = {image_id: sorted(ids) for image_id, ids in rebuilt.items()}
    report.update(image_field=image_key, annotation_field=annotation_key,
                  canonical_images=len(images), canonical_annotations=len(annotations),
                  validated_index_keys=validated_keys, non_dictionary_index_types=unusable_index_types,
                  mapping_source=('validated ' + validated_keys[0]) if validated_keys else 'rebuilt from anns[*].image_id',
                  polygon_source='unchanged points; no bbox/filename fallback')
    return dict(imgs=images, anns=annotations, img2Anns=selected_index), report


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def one_file(explicit, candidates, label):
    paths = [Path(explicit)] if explicit else sorted(set(Path(p) for p in candidates))
    if len(paths) != 1 or not paths[0].is_file():
        raise RuntimeError(f'{label}: expected one real file; candidates={[str(p) for p in paths]}. '
                           'Set the corresponding SHARED_CONFIG path explicitly. Do not guess.')
    return paths[0].resolve()


def discover(config):
    root = Path(config['input_root']).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Attach the three Kaggle datasets; missing {root}')
    # Names are defined by the supplied generator and the original annotation releases.
    clean = one_file(config['clean_json'], root.rglob('TextVQA_0.5.1_train.json'), 'clean_json')
    textocr = one_file(config['textocr_json'], root.rglob('TextOCR_0.1_train.json'), 'textocr_json')
    main_candidates = [p for p in root.rglob('image_manifest.csv') if p.parent.name == 'main']
    main = one_file(config['main_image_manifest'], main_candidates, 'main_image_manifest')
    return dict(clean_json=clean, textocr_json=textocr, main_manifest=main,
                clean_root=Path(config['clean_images_root']).resolve() if config['clean_images_root'] else clean.parent,
                degraded_root=Path(config['degraded_root']).resolve() if config['degraded_root'] else main.parent.parent)


def image_index(root):
    result = defaultdict(list)
    for p in sorted(Path(root).rglob('*')):
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
            result[p.name].append(p.resolve())
    return result


def resolve_recorded_path(recorded, root, index, expected_hash, kind):
    """Rebind a recorded path by exact suffix/basename and mandatory content hash.

    This does NOT infer image IDs from filenames. ID -> recorded file is in generator metadata.
    """
    if not recorded or not expected_hash or len(expected_hash) != 64:
        raise ValueError(f'{kind}: missing recorded path or SHA256')
    root = Path(root).resolve()
    parts = PurePosixPath(str(recorded).replace('\\', '/')).parts
    candidates = []
    direct = Path(recorded)
    if direct.is_file() and direct.resolve().is_relative_to(root):
        candidates.append(direct.resolve())
    if kind == 'degraded':
        anchors = [i for i in range(len(parts) - 1) if parts[i:i + 2] == ('main', 'images')]
        if len(anchors) == 1:
            rebound = root.joinpath(*parts[anchors[0]:])
            if rebound.is_file():
                candidates.append(rebound.resolve())
    # A unique exact basename is allowed only with the metadata hash check below.
    if not candidates:
        candidates = index.get(parts[-1], []) if parts else []
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError(f'{kind}: ambiguous/missing file for metadata path {recorded}: {candidates}')
    p = candidates[0]
    if not p.is_relative_to(root) or file_sha256(p) != expected_hash:
        raise ValueError(f'{kind}: bytes/path do not match generator metadata: {p}')
    return p


def image_size(path):
    with Image.open(path) as im:
        if im.getexif().get(274, 1) not in (None, 1):
            raise ValueError('Non-identity EXIF orientation; coordinate frame needs explicit audit')
        size = im.size
        im.convert('RGB').load()  # Decode every paired image; detect damaged files before inference.
    return size


def homography_from_corners(width, height, corners):
    src = np.array([[0., 0.], [width - 1., 0.], [width - 1., height - 1.], [0., height - 1.]])
    dst = np.asarray(corners, dtype=float)
    if dst.shape != (4, 2) or not np.isfinite(dst).all():
        raise ValueError('Perspective corners must be finite 4x2, in generator source-corner order')
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        b.extend([u, v])
    h = np.append(np.linalg.solve(np.array(a), np.array(b)), 1).reshape(3, 3)
    if not np.isfinite(h).all() or np.linalg.cond(h) > 1e12:
        raise ValueError('Unstable homography')
    projected = project_points(src, h)
    if np.max(np.abs(projected - dst)) > 1e-4:
        raise ValueError('Corner reprojection check failed')
    return h


def project_points(points, h):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    q = np.column_stack([points, np.ones(len(points))]) @ np.asarray(h).T
    if not np.isfinite(q).all() or np.any(np.abs(q[:, 2]) < 1e-10):
        raise ValueError('Homography crosses projective horizon')
    return q[:, :2] / q[:, 2, None]


def warp_geometry(geom, h):
    h = np.asarray(h, dtype=float)
    def project(x, y, z=None):
        xa, ya = np.asarray(x), np.asarray(y)
        den = h[2, 0]*xa + h[2, 1]*ya + h[2, 2]
        if np.any(np.abs(den) < 1e-10):
            raise ValueError('Projective horizon intersects polygon')
        return ((h[0, 0]*xa + h[0, 1]*ya + h[0, 2])/den,
                (h[1, 0]*xa + h[1, 1]*ya + h[1, 2])/den)
    out = (array_geometry_transform(geom, lambda coordinates: project_points(coordinates, h))
           if array_geometry_transform is not None else legacy_geometry_transform(project, geom))
    if not out.is_valid:
        out = out.buffer(0)
    if out.is_empty or out.area <= 0:
        raise ValueError('Empty warped geometry')
    return out


def source_geometry(record, size, output_size):
    if size != output_size:
        raise ValueError('Generator expects unchanged canvas. Unknown resize/crop; metadata audit needed.')
    params = record['params']
    if isinstance(params, str):
        params = json.loads(params)
    if not isinstance(params, list) or not params:
        raise ValueError('Missing per-image operation metadata')
    ops = [p['operation'] for p in params]
    if not set(ops) <= KNOWN_OPERATIONS:
        raise ValueError(f'Unsupported geometric/unknown operations: {set(ops)-KNOWN_OPERATIONS}')
    perspectives = [p for p in params if p['operation'] == 'perspective']
    if len(perspectives) > 1:
        raise ValueError('Multiple perspective transforms require explicit composition')
    w, h = size
    matrix = homography_from_corners(w, h, perspectives[0]['corners']) if perspectives else np.eye(3)
    canvas = box(0, 0, w, h)
    # Reflected border pixels are outside the source field of view, and must not create GT/FPs.
    source_visible = canvas.intersection(warp_geometry(canvas, np.linalg.inv(matrix)))
    if source_visible.is_empty or source_visible.area / canvas.area < 0.50:
        raise ValueError('Implausible visible field of view')
    return matrix, source_visible, ops


def annotation_geometry(annotation):
    arr = np.asarray(annotation['points'], dtype=float)
    if arr.size < 6 or arr.size % 2 or not np.isfinite(arr).all():
        raise ValueError('Invalid TextOCR polygon coordinates')
    geom = Polygon(arr.reshape(-1, 2))
    repaired = not geom.is_valid
    if repaired:
        geom = geom.buffer(0)
    if geom.is_empty or geom.area <= 0 or geom.geom_type not in {'Polygon', 'MultiPolygon'}:
        raise ValueError('Unusable TextOCR polygon')
    return geom, repaired


def gt_for_source(image_id, textocr, roi, config, audit_counts):
    valid, ignored, removed_ids = [], [], []
    for ann_id in textocr['img2Anns'].get(image_id, []):
        ann = textocr['anns'][str(ann_id)]
        if str(ann.get('image_id', image_id)) != image_id:
            raise ValueError('TextOCR img2Anns / anns image_id mismatch')
        try:
            geom, repaired = annotation_geometry(ann)
        except (ValueError, KeyError):
            audit_counts['unusable_gt_annotations'] += 1
            removed_ids.append(str(ann_id))
            continue
        audit_counts['repaired_gt_annotations'] += int(repaired)
        clipped = geom.intersection(roi)
        visibility = clipped.area / geom.area
        if clipped.is_empty or clipped.area <= 0:
            audit_counts['gt_outside_common_field_of_view'] += 1
            removed_ids.append(str(ann_id))
            continue
        unreadable = str(ann.get('utf8_string', '')).strip() in config['ignore_transcripts']
        explicit_ignore = ann.get('ignore', False) is True or ann.get('ignored', False) is True
        partial = visibility < config['min_word_visibility']
        entry = dict(ann_id=str(ann_id), geometry=mapping(clipped),
                     original_height=float(geom.bounds[3] - geom.bounds[1]), visibility=float(visibility))
        if unreadable or explicit_ignore or partial:
            entry['ignore_reason'] = 'unreadable' if unreadable or explicit_ignore else 'cropped_below_visibility'
            ignored.append(entry)
            audit_counts['gt_ignored_' + entry['ignore_reason']] += 1
        else:
            valid.append(entry)
    return dict(valid=valid, ignored=ignored), removed_ids


def overlay(geometry, drawing, color, width=2):
    if geometry.geom_type == 'Polygon':
        drawing.line(list(geometry.exterior.coords), fill=color, width=width)
    elif hasattr(geometry, 'geoms'):
        for part in geometry.geoms:
            overlay(part, drawing, color, width)


def save_gt_overlays(rows, locations, output, limit):
    output.mkdir(parents=True, exist_ok=True)
    chosen = sorted(rows, key=lambda r: (not r['has_perspective'], r['source_image_id'], r['variant_id']))[:limit]
    for i, row in enumerate(chosen):
        for cond in ('clean', 'degraded'):
            p = Path(locations[cond + '_root']) / row[cond + '_relpath']
            with Image.open(p) as im:
                im = im.convert('RGB')
                draw = ImageDraw.Draw(im)
                overlay(shape(row['roi_' + cond]), draw, '#00BFFF', 3)
                for item in row['gt_' + cond]['ignored']:
                    overlay(shape(item['geometry']), draw, '#AAAAAA')
                for item in row['gt_' + cond]['valid']:
                    overlay(shape(item['geometry']), draw, '#00FF00')
                im.thumbnail((1280, 1280))
                im.save(output / f'{i:02d}_{row["source_image_id"]}_{cond}.jpg', quality=88)


def build_shared_manifest(config=None):
    config = dict(DEFAULT_SHARED_CONFIG, **(config or {}))
    output = Path(config['output_root']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert 0 < config['min_word_visibility'] <= 1
    paths = discover(config)
    print('DISCOVERED (real files):', json.dumps({k: str(v) for k, v in paths.items()}, indent=2), flush=True)
    records = csv_rows(paths['main_manifest'])
    required = {'image_id', 'condition', 'level', 'params', 'output_path', 'source_image_path',
                'source_image_sha256', 'output_sha256', 'config_sha256', 'source_manifest_sha256'}
    if not records or not required <= records[0].keys():
        raise ValueError(f'Unsupported generator CSV schema; required={sorted(required)}')
    clean_json = read_json(paths['clean_json'])
    if not isinstance(clean_json.get('data'), list):
        raise ValueError('Unsupported TextVQA schema: expected data list')
    clean_ids = {str(r['image_id']) for r in clean_json['data']}
    del clean_json
    textocr_raw = read_json(paths['textocr_json'])
    schema_path = output / 'textocr_schema.json'
    schema_path.write_bytes(canonical_bytes(describe_textocr_schema(textocr_raw)))
    print('TEXTOCR RAW SCHEMA:', json.dumps(describe_textocr_schema(textocr_raw), indent=2), flush=True)
    try:
        textocr, textocr_schema = normalize_textocr(textocr_raw)
    except ValueError as exc:
        raise ValueError(f'{exc}. Schema diagnostics: {schema_path}') from exc
    schema_path.write_bytes(canonical_bytes(textocr_schema))
    del textocr_raw
    print('TEXTOCR MAPPING:', textocr_schema['mapping_source'], flush=True)
    print('SCHEMAS:', dict(main_columns=sorted(records[0]), main_records=len(records),
                          textvqa_unique_ids=len(clean_ids), textocr_images=len(textocr['imgs'])), flush=True)

    # Verify the actual frozen config against every selected generator record.
    frozen_path = paths['degraded_root'] / 'config/frozen_degradation_config.json'
    policy_path = paths['degraded_root'] / 'metadata/split_policy.json'
    eval_path = paths['degraded_root'] / 'metadata/locked_eval_manifest.csv'
    if not all(p.is_file() for p in (frozen_path, policy_path, eval_path)):
        raise ValueError('Missing frozen config / split policy / locked_eval_manifest. Inspect input; do not guess.')
    frozen, policy = read_json(frozen_path), read_json(policy_path)
    if frozen.get('protocol') != GENERATOR_PROTOCOL or policy.get('protocol') != GENERATOR_PROTOCOL:
        raise ValueError('Not the supplied v2_10k generator protocol; needs an explicit schema adapter')
    frozen_hash, selection_hash = file_sha256(frozen_path), file_sha256(eval_path)
    if frozen.get('eval_manifest_sha256') != selection_hash:
        raise ValueError('Frozen config / selection manifest hash mismatch')
    if frozen.get('train_annotation_sha256') != file_sha256(paths['clean_json']):
        raise ValueError('Attached clean TextVQA JSON is not the source used by the generator')
    selected_rows = csv_rows(eval_path)
    if not selected_rows or '_source_image_sha256' not in selected_rows[0]:
        raise ValueError('Locked selection lacks source-image hashes; cannot verify ID-to-clean mapping')
    selected_ids = {r['_image_id'] for r in selected_rows}
    selected_hashes = {r['_image_id']: r['_source_image_sha256'] for r in selected_rows}
    if len(selected_rows) != len(selected_ids):
        raise ValueError('Locked main selection contains duplicate source IDs')
    if {r['image_id'] for r in records} != selected_ids:
        raise ValueError('Main metadata does not cover the locked source selection exactly')
    if any(r['source_image_sha256'] != selected_hashes[r['image_id']] for r in records):
        raise ValueError('Generator image_id-to-source hash disagrees with locked selection')
    record_keys = [(r['image_id'], r['condition'], r['level']) for r in records]
    if len(set(record_keys)) != len(records):
        raise ValueError('Duplicate main image metadata records')
    if any(r['config_sha256'] != frozen_hash or r['source_manifest_sha256'] != selection_hash for r in records):
        raise ValueError('Main records use different frozen config/selection; do not mix runs')

    clean_files, degraded_files = image_index(paths['clean_root']), image_index(paths['degraded_root'] / 'main')
    counts = Counter(main_metadata_instances=len(records), main_unique_sources=len(selected_ids),
                     textvqa_unique_sources=len(clean_ids), textocr_sources=len(textocr['imgs']))
    candidates, excluded = defaultdict(list), []
    for i, r in enumerate(sorted(records, key=lambda x: (x['image_id'], x['condition'], x['level']))):
        image_id = str(r['image_id'])
        if not re.fullmatch(r'[A-Za-z0-9_-]+', image_id):
            raise ValueError('Unsupported source ID characters; do not infer or normalize IDs')
        def exclude(reason, detail=''):
            counts['excluded_' + reason] += 1
            excluded.append(dict(image_id=image_id, condition=r['condition'], level=r['level'],
                                 reason=reason, detail=str(detail)))
        if image_id not in clean_ids:
            exclude('not_in_textvqa'); continue
        if image_id not in textocr['imgs'] or not textocr['img2Anns'].get(image_id):
            exclude('not_in_textocr_or_no_gt'); continue
        counts['intersection_before_file_geometry_checks'] += 1
        try:
            clean = resolve_recorded_path(r['source_image_path'], paths['clean_root'], clean_files,
                                          r['source_image_sha256'], 'clean')
            degraded = resolve_recorded_path(r['output_path'], paths['degraded_root'], degraded_files,
                                             r['output_sha256'], 'degraded')
            size, degraded_size = image_size(clean), image_size(degraded)
            meta = textocr['imgs'][image_id]
            if size != (int(meta['width']), int(meta['height'])):
                raise ValueError('Clean pixels and TextOCR GT dimensions differ')
            h, source_visible, ops = source_geometry(r, size, degraded_size)
        except (ValueError, OSError, KeyError, np.linalg.LinAlgError) as exc:
            exclude('file_or_geometry_invalid', exc); continue
        candidates[image_id].append(dict(record=r, clean=clean, degraded=degraded, size=size,
                                         matrix=h, source_visible=source_visible, ops=ops))
        if (i + 1) % 250 == 0:
            print('AUDIT:', i + 1, '/', len(records), flush=True)

    rows = []
    for image_id, variants in sorted(candidates.items()):
        roi = box(0, 0, *variants[0]['size'])
        for v in variants:
            roi = roi.intersection(v['source_visible'])
        gt_clean, removed_ids = gt_for_source(image_id, textocr, roi, config, counts)
        if not gt_clean['valid']:
            counts['excluded_source_no_usable_valid_gt'] += 1
            for v in variants:
                excluded.append(dict(image_id=image_id, condition=v['record']['condition'],
                                     level=v['record']['level'], reason='no_usable_valid_gt', detail=''))
            continue
        for v in variants:
            r, h = v['record'], v['matrix']
            gt_degraded = {key: [dict(item, geometry=mapping(warp_geometry(shape(item['geometry']), h)))
                                for item in gt_clean[key]] for key in ('valid', 'ignored')}
            variant_id = hashlib.sha256(canonical_bytes([image_id, r['condition'], r['level'],
                                                        r['output_sha256'], r['config_sha256']])).hexdigest()[:24]
            rows.append(dict(schema_version=SCHEMA_VERSION, source_image_id=image_id, variant_id=variant_id,
                             clean_relpath=v['clean'].relative_to(paths['clean_root']).as_posix(),
                             degraded_relpath=v['degraded'].relative_to(paths['degraded_root']).as_posix(),
                             clean_sha256=r['source_image_sha256'], degraded_sha256=r['output_sha256'],
                             clean_size=list(v['size']), degraded_size=list(v['size']),
                             condition=r['condition'], severity=r['level'], operations=v['ops'],
                             has_perspective='perspective' in v['ops'], homography=h.tolist(),
                             roi_clean=mapping(roi), roi_degraded=mapping(warp_geometry(roi, h)),
                             gt_clean=gt_clean, gt_degraded=gt_degraded, removed_gt_ids=removed_ids,
                             source_config_sha256=r['config_sha256'],
                             severity_basis=r.get('severity_basis', 'not_recorded')))
    rows.sort(key=lambda r: (r['source_image_id'], r['variant_id']))
    if not rows:
        raise RuntimeError('Empty valid intersection. STOP and inspect audit; no full inference.')
    counts['evaluated_instances'] = len(rows)
    counts['evaluated_unique_sources'] = len({r['source_image_id'] for r in rows})
    counts['perspective_instances'] = sum(r['has_perspective'] for r in rows)
    protocol = dict(schema_version=SCHEMA_VERSION, matching='maximum_cardinality_polygon_iou',
                    match_iou=config['match_iou'], word_coverage=config['coverage_threshold'],
                    min_prediction_roi_fraction=config['min_prediction_roi_fraction'],
                    ignore_prediction_overlap=config['ignore_prediction_overlap'],
                    ignore_transcripts=config['ignore_transcripts'], min_word_visibility=config['min_word_visibility'],
                    score_policy='No additional score filtering/sweep; preserve each detector fixed postprocessing',
                    geometry_policy='common source field of view across variants; same valid annotation IDs in clean/degraded',
                    split_policy='baseline_all_sources_no_training; no unseen-data claim; split before later fine-tuning')
    manifest = output / 'shared_manifest_v1.jsonl'
    digest = write_locked(manifest, b''.join(canonical_bytes(row) for row in rows))
    protocol_hash = write_locked(output / 'evaluation_protocol.json', canonical_bytes(protocol))
    write_locked(output / 'shared_manifest_v1.sha256', (digest + '\n').encode())
    counts['excluded_instances_total'] = len(excluded)
    audit = dict(counts=dict(counts), manifest_sha256=digest, protocol_sha256=protocol_hash,
                 generator_frozen_sha256=frozen_hash, generator_selection_sha256=selection_hash,
                 actual_generator_policy=policy, textocr_sha256=file_sha256(paths['textocr_json']),
                 textocr_schema=textocr_schema,
                 clean_annotation_sha256=file_sha256(paths['clean_json']),
                 gt_identity_policy=protocol['geometry_policy'], scope='MAIN only; excludes calibration and ablation')
    write_locked(output / 'data_audit.json', canonical_bytes(audit))
    write_locked(output / 'excluded.jsonl', b''.join(canonical_bytes(row) for row in excluded))
    clean_by_source = {r['source_image_id']: r for r in rows}
    index_tables = {
        'clean_index.csv': [dict(image_id=s, relative_path=r['clean_relpath'], sha256=r['clean_sha256'])
                            for s, r in sorted(clean_by_source.items())],
        'degraded_index.csv': [dict(image_id=r['source_image_id'], variant_id=r['variant_id'],
                                   relative_path=r['degraded_relpath'], sha256=r['degraded_sha256'],
                                   geometry='homography' if r['has_perspective'] else 'identity',
                                   homography_json=json.dumps(r['homography'], separators=(',', ':')))
                               for r in rows],
    }
    for name, records_out in index_tables.items():
        buf = io.StringIO(newline='')
        writer = csv.DictWriter(buf, fieldnames=list(records_out[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records_out)
        write_locked(output / name, buf.getvalue().encode('utf-8'))
    locations = dict(clean_root=str(paths['clean_root']), degraded_root=str(paths['degraded_root']))
    # Machine-local roots are outside the portable frozen manifest.
    (output / 'dataset_locations.json').write_bytes(canonical_bytes(locations))
    save_gt_overlays(rows, locations, output / 'gt_overlays', config['overlay_sources'])
    report = '\n'.join(['# Phase 2 data audit', '', f'Manifest SHA256: `{digest}`', '',
                        'MAIN only; IDs from generator metadata, verified source/output hashes.', '',
                        *[f'- {k}: {v}' for k, v in sorted(counts.items())], '',
                        'Cyan: valid field of view. Green: valid GT. Gray: ignored/cropped GT.',
                        'Review overlays before full inference. This run does not fine-tune or generate images.'])
    (output / 'data_audit.md').write_text(report + '\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return dict(manifest=str(manifest), manifest_sha256=digest, output_root=str(output),
                locations=str(output / 'dataset_locations.json'), protocol=str(output / 'evaluation_protocol.json'),
                counts=dict(counts))
