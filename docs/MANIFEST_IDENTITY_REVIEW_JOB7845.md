# Phase 2 / HPC job 7845 manifest comparison

Job 7845 stopped during the dataset audit, before the Phase 3 training probe,
E0, tiling, E1 or E2. Its rebuilt manifest failed the original byte-exact lock.
That check was too restrictive for the observed numerical roundoff between
the original Kaggle run and the HPC rebuild. The precise library/CPU cause of
the roundoff has not been isolated; it is not inferred from the GPU model.

The supplied `phase3_manifest_debug_job7845.zip` was compared against the
previously audited Phase 2 artifact, streaming all 9,898 ordered source rows.
Both complete manifests were hashed during the comparison:

- Original Phase 2: `94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242`.
- HPC job 7845: `ad43fa08d905fb5e50a6690a34b29da4afa6f19475a9afda4a375ea0171c7fb5`.

Forty rows have different bytes. All differences are floating-point values in
homographies, ROI/GT coordinates or visibility ratios. Maximum absolute changes:

| Quantity | Maximum difference |
|---|---:|
| Geometry coordinates | 9.322320693172514e-12 pixels |
| Homography coefficients | 2.220446049250313e-16 |
| Visibility ratios | 4.440892098500626e-15 |

Every other field, value type, list length and row order is exactly equal.
This includes source/variant IDs, clean/degraded image hashes, relative paths,
dimensions, annotation IDs and valid/ignored membership, removed annotation IDs,
operation names and generator configuration hashes. No source was substituted.
The complete `data_audit.json` objects are also equal except `manifest_sha256`;
the counts, TextVQA/TextOCR hashes, generator hashes and protocol hash all match.

The machine-readable comparison is
`manifest_comparison__phase2_vs_hpc_job7845.json`. It records field-level change
counts and maxima without publishing images, transcripts or GT coordinates.

The audit is reproducible with both original artifacts available locally:

```bash
python scripts/compare_phase2_manifest_variants.py \
  --reference /path/to/original_phase2/shared_manifest_v1.jsonl \
  --candidate /path/to/phase3_manifest_debug_job7845.zip \
  --report /path/to/comparison.json
```

This offline tool allows absolute differences only in designated geometry,
homography and visibility floats, bounded by 1e-9, 1e-14 and 1e-12 respectively.
Changes elsewhere, type/structure changes and missing/reordered rows fail.
These bounds are for review; they are **not** a runtime fuzzy matching rule.

## Runtime correction and recovery

`configs/verified_phase2_manifest_variants.json` binds the original Phase 2
manifest hash and the complete audit identity, excluding only its manifest hash.
It additionally registers the exact complete SHA256 of the reviewed job 7845
variant and the comparison report SHA256. Runtime accepts the original or that
reviewed byte variant, checks the actual file rather than trusting its audit,
and continues checking the exact protocol and evaluator hashes. Unknown variants
still stop, even if their differences might also be numerical. Adding another
variant requires another comparison and review.

No GT is rounded, rewritten or substituted. Runtime saves
`phase2_manifest_identity_verification.json` with the actual hash, original hash,
registry hash and review. The HPC variant has negligible numerical differences
from the Phase 2 bytes, not byte-identical coordinates. Every Phase 3 method uses
the same newly frozen manifest; share those exact bytes with Paddle as well.
Phase 3 E0 is evaluated again in the same environment and on the same split as
the other Phase 3 methods.

The default run name now ends in `seed42_v2`. This preserves job 7845's failed
`v1` inputs, logs and immutable state. The same workspace is retained so its
verified CUDA/Python/model cache can be reused. Bootstrap code and model/source
lock files are unchanged in this correction. Do not disable dataset verification
or delete the previous workspace to bypass the failure.

This fixes the observed manifest rejection. It does not establish that subsequent
GPU training stages have already completed or will pass their runtime probes.
