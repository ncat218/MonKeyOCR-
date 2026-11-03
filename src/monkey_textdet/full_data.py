"""Adapter for Tuan's FULL_R1 metadata; never reinterpret its QA rows as word GT."""
import collections
import csv
import shutil
from pathlib import Path

from .common import identity, read_json, save_json, sha256, write_jsonl

FULL_PROTOCOL = 'TEXTVQA_REALISTIC_DEGRADATION_V2_FULL_R1'


def metadata_paths(paths):
    root = Path(paths['degraded_root'])
    return dict(frozen=root/'config/frozen_degradation_config.json',
                policy=root/'metadata/split_policy.json',
                qa=root/'metadata/locked_eval_manifest.csv')


def read_metadata(paths):
    """Light pre-submission check: metadata only, no image or TextOCR decoding."""
    files = metadata_paths(paths)
    for p in files.values():
        if not p.is_file():
            raise FileNotFoundError(f'Missing FULL_R1 metadata: {p}')
    frozen, policy = read_json(files['frozen']), read_json(files['policy'])
    if frozen.get('protocol') != FULL_PROTOCOL:
        raise ValueError('Expected explicit FULL_R1 protocol; do not use the old 10K adapter')
    if not policy.get('official_splits_preserved'):
        raise ValueError('FULL_R1 requires recorded official source splits')
    if sha256(paths['clean_json']) != frozen['source_provenance']['annotation_sha256']['train']:
        raise ValueError('TextVQA train JSON differs from FULL_R1 source provenance')
    with files['qa'].open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'image_id', 'split', 'question_id'} <= set(reader.fieldnames or []):
            raise ValueError('FULL_R1 locked_eval_manifest must be the QA index')
        qa = {}
        questions = set()
        for r in reader:
            source, fold = r['image_id'], r['split']
            if fold not in ('train', 'val', 'test') or qa.setdefault(source, fold) != fold:
                raise ValueError('Conflicting official splits in QA index')
            key = (fold, r['question_id'])
            if key in questions:
                raise ValueError('Duplicate QA question key')
            questions.add(key)
    # QA may be copied/serialized to locked_eval_manifest. Bind the actual source
    # QA bytes to the frozen hash, not merely to a guessed filename.
    expected = frozen['source_manifest_sha256']
    candidates = [files['qa']] + sorted((Path(paths['degraded_root'])/'metadata').rglob('*.csv'))
    source_qa = next((p for p in dict.fromkeys(candidates) if sha256(p) == expected), None)
    if source_qa is None:
        raise ValueError('No metadata CSV matches frozen source_manifest_sha256; inspect generator metadata')
    if source_qa != files['qa']:
        with source_qa.open(encoding='utf-8-sig', newline='') as f:
            original = list(csv.DictReader(f))
        with files['qa'].open(encoding='utf-8-sig', newline='') as f:
            locked_keys = {(r['split'], r['question_id'], r['image_id']) for r in csv.DictReader(f)}
        if {(r['split'], r['question_id'], r['image_id']) for r in original} != locked_keys:
            raise ValueError('Locked QA selection differs from the frozen full source QA')
    with Path(paths['main_image_manifest']).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'image_id', 'source_image_id', 'split', 'condition', 'level', 'params',
                    'output_path', 'source_image_path', 'source_image_sha256', 'output_sha256',
                    'config_sha256', 'source_manifest_sha256', 'clean_width', 'clean_height',
                    'degraded_width', 'degraded_height'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f'Unsupported FULL_R1 MAIN columns; missing {required-set(reader.fieldnames or [])}')
        records = list(reader)
    ids = set()
    frozen_hash = sha256(files['frozen'])
    for r in records:
        source = r['image_id']
        if source in ids or r['source_image_id'] != source:
            raise ValueError('Expected exactly one MAIN variant per source ID')
        ids.add(source)
        if (r['condition'], r['level']) != ('realistic_mix', 'L2'):
            raise ValueError('This controlled experiment supports MAIN realistic_mix/L2 only')
        if qa.get(source) != r['split']:
            raise ValueError('MAIN split/ID differs from locked full QA index')
        if r['config_sha256'] != frozen_hash or r['source_manifest_sha256'] != expected:
            raise ValueError('Mixed FULL_R1 generator versions in MAIN')
    if not records or ids != set(qa):
        raise ValueError('MAIN does not cover the full locked QA source IDs exactly')
    splits = dict(collections.Counter(r['split'] for r in records))
    for fold, count in splits.items():
        if frozen['source_provenance']['counts'][fold]['images'] != count:
            raise ValueError('MAIN counts disagree with frozen source provenance')
    return records, dict(protocol=FULL_PROTOCOL, main_counts_by_official_split=splits,
                         qa_questions=len(questions), source_qa=str(source_qa),
                         metadata_sha256={k: sha256(p) for k, p in files.items()},
                         source_qa_sha256=expected, human_review_performed=frozen.get('human_review_performed'))


def build_full_shared(paths, output):
    from shapely.geometry import box, mapping, shape
    from .vendor import shared_manifest as sm
    output.mkdir(parents=True, exist_ok=True)
    records, provenance = read_metadata(paths)
    for name, path in metadata_paths(paths).items():
        shutil.copyfile(path, output/('GENERATOR_INPUT__'+name+path.suffix))
    source_qa = Path(provenance['source_qa'])
    if sha256(source_qa) != sha256(metadata_paths(paths)['qa']):
        shutil.copyfile(source_qa, output/'GENERATOR_INPUT__source_QA_manifest__frozen_hash_verified.csv')
    # Full audit does not substitute simulated answerability labels for TextOCR GT.
    save_json(output/'full_generator_metadata__verified_source_splits_and_QA_identity.json', provenance)
    textvqa = read_json(paths['clean_json'])
    clean_ids = {str(r['image_id']) for r in textvqa['data']}
    raw = read_json(paths['textocr_json'])
    textocr, schema = sm.normalize_textocr(raw)
    del raw, textvqa
    save_json(output/'textocr_schema.json', schema)
    counts = collections.Counter(main_records=len(records))
    clean_index = sm.image_index(Path(paths['clean_images_root']))
    degraded_index = sm.image_index(Path(paths['degraded_root'])/'main')
    rows, excluded = [], []
    for i, r in enumerate(sorted(records, key=lambda r: r['image_id'])):
        source = r['image_id']
        reason = None
        if r['split'] != 'train':
            reason = 'outside_agreed_official_train_scope__'+r['split']
        elif source not in clean_ids:
            raise ValueError('MAIN official train source absent from verified TextVQA train JSON')
        elif source not in textocr['imgs'] or not textocr['img2Anns'].get(source):
            reason = 'no_TextOCR_GT'
        if reason:
            counts['excluded_'+reason] += 1
            excluded.append(dict(source_image_id=source, official_split=r['split'], reason=reason))
            continue
        cp = sm.resolve_recorded_path(r['source_image_path'], Path(paths['clean_images_root']),
                                     clean_index, r['source_image_sha256'], 'clean')
        dp = sm.resolve_recorded_path(r['output_path'], Path(paths['degraded_root']),
                                     degraded_index, r['output_sha256'], 'degraded')
        size, dsize = sm.image_size(cp), sm.image_size(dp)
        expected_size = (int(r['clean_width']), int(r['clean_height']))
        if size != expected_size or dsize != (int(r['degraded_width']), int(r['degraded_height'])):
            raise ValueError('Pixels differ from FULL_R1 recorded image dimensions')
        meta = textocr['imgs'][source]
        if size != (int(meta['width']), int(meta['height'])):
            raise ValueError('TextOCR coordinate frame differs from clean pixels')
        h, roi, operations = sm.source_geometry(r, size, dsize)
        gt, removed = sm.gt_for_source(source, textocr, roi, sm.DEFAULT_SHARED_CONFIG, counts)
        if not gt['valid']:
            counts['excluded_no_usable_valid_GT'] += 1
            excluded.append(dict(source_image_id=source, official_split='train', reason='no_usable_valid_GT'))
            continue
        deg_gt = {key: [dict(a, geometry=mapping(sm.warp_geometry(shape(a['geometry']), h)))
                        for a in gt[key]] for key in ('valid', 'ignored')}
        rows.append(dict(schema_version=sm.SCHEMA_VERSION, source_image_id=source,
            variant_id=identity([source, r['output_sha256'], r['config_sha256']])[:24],
            official_split='train', clean_relpath=cp.relative_to(paths['clean_images_root']).as_posix(),
            degraded_relpath=dp.relative_to(paths['degraded_root']).as_posix(),
            clean_sha256=r['source_image_sha256'], degraded_sha256=r['output_sha256'],
            clean_size=list(size), degraded_size=list(dsize), condition=r['condition'], severity=r['level'],
            operations=operations, has_perspective='perspective' in operations, homography=h.tolist(),
            roi_clean=mapping(roi), roi_degraded=mapping(sm.warp_geometry(roi, h)),
            gt_clean=gt, gt_degraded=deg_gt, removed_gt_ids=removed,
            source_config_sha256=r['config_sha256'], severity_basis=r.get('severity_basis', 'not_recorded')))
        if (i+1) % 250 == 0:
            print('FULL_R1 IMAGE/HASH/GT AUDIT', i+1, '/', len(records), flush=True)
    if len(rows) < 3:
        raise ValueError('Fewer than three valid source groups; no training')
    manifest = output/'shared_manifest__FULL_R1_official_train_TextVQA_TextOCR_intersection.jsonl'
    write_jsonl(manifest, rows)
    write_jsonl(output/'excluded_sources__explicit_reason_and_official_split.jsonl', excluded)
    protocol = dict(schema_version=sm.SCHEMA_VERSION, matching='maximum_cardinality_polygon_iou',
        match_iou=.5, word_coverage=.5, min_prediction_roi_fraction=.5, ignore_prediction_overlap=.5,
        ignore_transcripts=['.', '###', ''], min_word_visibility=.8,
        score_policy='Native postprocessing or fixed mask threshold selected on validation only; no test sweep',
        geometry_policy='Common source FOV and valid annotation IDs; perspective GT warped from clean',
        split_policy='Official TextVQA train intersection, source/clean-SHA grouped 70/15/15 detection split')
    save_json(output/'evaluation_protocol.json', protocol)
    counts.update(evaluated_unique_sources=len(rows), evaluated_instances=len(rows),
                  valid_words_per_condition=sum(len(r['gt_clean']['valid']) for r in rows))
    audit = dict(counts=dict(counts), generator=provenance, manifest_sha256=sha256(manifest),
        textocr_sha256=sha256(paths['textocr_json']), clean_annotation_sha256=sha256(paths['clean_json']),
        scope='All eligible MAIN official-train sources, not the old10k. Official val/test, calibration and ablation excluded.',
        disclosure='Synthetic data; simulated review labels are NOT human readability GT. TextOCR train was used in Monkey pretraining.')
    save_json(output/'data_audit__FULL_R1_counts_GT_intersection_and_exclusions.json', audit)
    return dict(manifest=str(manifest), manifest_sha256=sha256(manifest), sources=len(rows))
