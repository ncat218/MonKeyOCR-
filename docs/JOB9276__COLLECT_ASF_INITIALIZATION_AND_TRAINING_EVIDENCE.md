# Job9276: collect evidence before attributing the ASF regression

The completed FULL_R1 run shows worse micro word-level Hmean for the newly added
ASF branch, especially small clean words. This is a result for the tested
initialization and fine-tuning schedule, not proof that ASF is generally worse.
Strict loading, finite probe gradients and successful completion do not establish
that newly initialized attention learned useful weights or reached convergence.

## Collect existing files on the HPC login node

This command uses only the Python standard library. It does not execute models,
request a GPU, submit Slurm, change packages, remove files or touch checkpoints.
It writes one new ZIP outside the existing experiment directory.

```bash
cd /datastore/dungmt/textvqa/Tuan/MonKeyOCR-
git pull --ff-only

python scripts/collect_full_r1_asf_training_diagnostics.py \
  --workspace /datastore/dungmt/textvqa/Tuan/monkeyocr_workspace \
  --job-id 9276
```

Download the exact path printed after `ZIP_TO_SEND` with WinSCP. The descriptive
filename includes initialization, training, validation, source snapshot and job9276.
Do not rerun `run_all.py` for this collection step.

## Contents and limits

- Existing generated A/B model configurations and initialization receipts.
- Existing A/B compatibility probes, training metadata, scalar JSON and text logs.
- Worker/Slurm logs, completion marker and run index.
- Current repository source/config snapshots and installed upstream FPNC/head/
  postprocessor/backbone and MMEngine initialization/optimizer snapshots when present;
  these are explicitly labelled as collected
  now, not automatically authenticated as the original job source.
- Parsed validation CSV with source filename, source line, raw scalar step and
  preceding textual train microiteration in separate fields. Fractions are not
  percentages. Duplicate JSON/text representations are intentionally retained.
- Collection manifest with per-file hashes, sizes, missing files and truncations.

No images, weights, raw predictions or dataset manifests are included. Logs over
8 MiB are represented by their last 8 MiB; oversized non-logs are skipped. Source
payload is bounded to 64 MiB and all omissions are reported. Collection warnings
do not mean training failed. They mean that an evidence file needs inspection.

## Questions to answer from the packet

1. Was ASF enabled in both generated training and inference configurations?
2. Which parameter keys were newly initialized? Did recorded shared-weight equality,
   strict loading and ASF probe gradient checks pass?
3. What happened to clean/degraded validation Hmean at each recorded training point?
   Confirm logger step semantics before calling the x-axis optimizer updates.
4. Did the final weights follow the declared fixed-update rule? A rising or flat
   validation curve alone does not prove an optimization root cause.
5. Are there warnings, nonfinite values or load messages that require deeper review?
6. Does the installed adapter cast backbone parameter storage to BF16, and does
   the optimizer preserve FP32 master weights? Source/config evidence alone cannot
   establish what fraction of actual backbone weights changed during training.

This packet does not measure attention maps, feature norms or actual parameter
updates, and does not evaluate the initial untrained ASF detector. Those would
require a separate explicitly identified model diagnostic after reviewing logs.

## Hypotheses that must not be reported as measured results

- Kaiming fan-out initialization on the four-output attention projection can give
  large logits; its actual saturation and selected scales require real activations.
  A random-feature simulation does not establish saturation in job9276.
- A BF16 backbone with ordinary AdamW and a small learning rate may lose small
  updates to rounding. `requires_grad=True` and finite gradients do not prove that
  stored parameters changed. Compare original and final tensors on CPU, accounting
  for the initial dtype conversion and distinguishing parameters from buffers.
- A low step-zero validation score measures a changed initial detector, not by
  itself a programming bug. A rising final validation point does not prove that
  extending training will repair the regression.
