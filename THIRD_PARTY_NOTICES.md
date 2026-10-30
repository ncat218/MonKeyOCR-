# Sources and runtime dependencies

This repository contains experiment orchestration and the project's detector-independent Phase 2 manifest/evaluator implementation. It does not redistribute model weights, datasets, or full third-party source checkouts.

Runtime setup downloads exact source revisions of MonkeyOCRv2, MMOCR, MMCV and exact model revisions from Hugging Face. Their upstream licenses, notices and model/dataset usage terms continue to apply. See `configs/verified_phase1_phase2_sources.json` and the upstream repositories for authoritative licenses.

MMOCR/MMCV/MMEngine are OpenMMLab projects. The official MonkeyOCRv2 detection add-on supplies the backbone adapter and integration patches; setup executes its pinned installer. Our setup additionally builds MMCV using C++17 for PyTorch 2.9 and the allocated GPU architecture, following the verified Phase 1 setup.

The Qwen baseline repository was consulted as an organization reference only. No Qwen model, recognition or VQA evaluation is part of this text detection experiment suite.
