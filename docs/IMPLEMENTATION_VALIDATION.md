# Verification before first H200 run

The implementation has not executed Phase 3 training or inference on a GPU during development. No hardware speedup, training VRAM guarantee, or accuracy improvement is claimed.

CPU checks cover:

- Complete overlapping 2x2 image coverage, tile-coordinate inversion, internal-seam filtering, polygon rather than bbox NMS, and duplicate suppression.
- Discovery with renamed dataset parents and rejection of ambiguous versions.
- Deterministic source-ID/source-byte grouped splits with no train/validation/test leakage.
- E1/E2 source pools, mixed-condition training annotations, outside-ROI ignore masks with holes, and training-only conversion of fragmented polygons.
- Exact frozen Phase 2 evaluator source hash; evaluation GT is not replaced by training approximations.
- Truncated-last-line prediction recovery and rejection of corrupt middle lines.
- Completed-stage output hashes, changed-input rejection, failed-stage retry, and exclusive stage identity.
- One command submitting one Slurm allocation; plan-only does not submit or bootstrap.
- A miniature end-to-end orchestration test with explicitly fake GPU workers: E0 global, validation-only tiled selection, tiled test, independent E1 and E2 initialization, final comparisons, then resume without repeating GPU stages. Fake outputs exist only inside temporary test directories.
- Optional real MMEngine 0.10.7 Config generation and reload: model architecture, native inference, same optimizer budget and same original checkpoint across E1/E2, distinct training annotations, and validation sample IDs.

The runtime performs additional checks before expensive experiments:

1. Dataset locations exist and are unambiguous before the CUDA stack is compiled.
2. A real compute allocation exposes one CUDA device; toolkit selection is recorded. If no compatible CUDA 12.x compiler is found, the configured fallback installs NVIDIA CUDA Toolkit 12.8.1 in a separate workspace Conda prefix, then checks headers/libraries and compiles a tiny CUDA translation unit before the large stack setup. This does not install a driver or modify Conda base.
3. MMCV NMS and deformable convolution forward/backward execute on the allocated device.
4. Original checkpoint/config/backbone hashes match verified Phase 1 revisions.
5. The real MMOCR training data pipeline, loss, backward pass and AdamW optimizer are probed on a minibatch before long inference. The in-memory probe model is discarded and the original checkpoint file is unchanged.
6. Every inference input is checked against its frozen image hash; incomplete predictions are not scored.

Exact inference bitwise equality across T4 and H200 is not assumed. The Phase 3 E0 baseline is re-evaluated on the same test sources and H200 environment used for the other methods. Dataset master identity and evaluator/protocol hashes remain pinned to the Phase 2 baseline.

If a runtime check fails, the suite stops with a named log and resumable artifacts. It never emits a completed comparison in place of missing experiment results.
