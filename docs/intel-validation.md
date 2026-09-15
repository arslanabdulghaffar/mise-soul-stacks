# Intel validation handoff

The full-table controller is a deterministic camera-guided contact controller.
The exported OpenVINO model is a validated learned **mug-only** controller. Its
action-context policy passed ten fresh held-out development-host trials; the result
does not establish learned full-table control or Intel quality preservation. Failed
attempts must stay in the evidence.

## Files to transfer to the target

Clone the repository with its submodule, install the pinned runtime and learning
dependencies, and create the verified transfer ZIP with
`python3 -m scripts.package_contact_policy`. Extract it into the clone, preserving
paths; it supplies `artifacts/models/contact_act_context/` (checkpoint, OpenVINO
XML/BIN, registration, and manifest), the frozen scene, and the ten-seed evaluation.
These large/generated artifacts are not tracked in Git. Keep the frozen scene at the
same relative path so its mesh paths resolve into `vendor/`.

The evaluation checks export/checkpoint hashes before running and records the scene,
mesh, source, and per-attempt checksums. It rejects existing output directories.
Use a new experiment directory for a new declared configuration and retain failures.

## Paired commands on the actual Core Ultra Series 2/3 target

The following comparison changes CPU inference threads from 2 to 4. This is a
candidate setting, not a promised speedup. Choose a supported device/precision
before collecting submission seeds; for GPU use `--device GPU --precision f16` in
both candidate commands. CPU thread options are not passed to GPU/NPU plugins.

```bash
python3 bench/intel_bench.py --device CPU --precision f32 --threads 2 \
  --intel-core-ultra-series 2 \
  --json-output artifacts/intel/reference-benchmark.json \
  --csv-output artifacts/intel/reference-benchmark.csv
MUJOCO_GL=osmesa python3 scripts/evaluate_contact_policy.py \
  --backend openvino --device CPU --precision f32 --threads 2 \
  --seeds 40000 40001 40002 40003 40004 40005 40006 40007 40008 40009 \
  --output artifacts/intel/reference

python3 bench/intel_bench.py --device CPU --precision f32 --threads 4 \
  --intel-core-ultra-series 2 \
  --json-output artifacts/intel/candidate-benchmark.json \
  --csv-output artifacts/intel/candidate-benchmark.csv
MUJOCO_GL=osmesa python3 scripts/evaluate_contact_policy.py \
  --backend openvino --device CPU --precision f32 --threads 4 \
  --seeds 40000 40001 40002 40003 40004 40005 40006 40007 40008 40009 \
  --output artifacts/intel/candidate

python3 scripts/validate_intel_pair.py \
  --reference artifacts/intel/reference --candidate artifacts/intel/candidate \
  --reference-benchmark artifacts/intel/reference-benchmark.json \
  --candidate-benchmark artifacts/intel/candidate-benchmark.json \
  --output artifacts/intel/paired-validation.json
```

Use `--intel-core-ultra-series 3` only for Series 3. Detection also checks the CPU
SKU; a declaration cannot turn a Xeon, Series 1, or unknown CPU into verified
Series 2/3 evidence. Virtualized or unrecognized names remain unverified.

Both benchmark and controller share static input shapes, latency hint, precision
hint, and CPU thread configuration. Reports distinguish requested configuration
from resolved device properties. A precision hint does not prove that every
operator executes in that precision. An unsupported device or precision must fail
visibly; no fallback measurement may be relabeled as the requested device.

The checker returns exit code 1 with a retained report when the strict gate fails:
at least ten paired successful reference **and** candidate attempts, 100 measured
iterations after ten warmups, lower median and no worse p95, matching model/source/
scene/device settings, and verified target identity. Structural mismatches, missing
attempts, or broken checksums stop validation without a claim. Zero successes on
both sides cannot qualify as preserved quality. This conservative internal gate is
not a guarantee of the judges' scoring.

Also run and record the actual full-table MuJoCo demo on the target (`python3 -m
mise.cli full --seed 1001 --output artifacts/intel/full-task`). Retain its host and
timing evidence separately; its success is not a learned ACT success.

References: [OpenVINO CPU precision and performance configuration](https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device.rst),
[Intel Core Ultra Series 1/2/3 SKU comparison](https://cdrdv2-public.intel.com/851467/Intel-Core-Ultra-Series1-Series2-Series3-Comparison.pdf).
