# Soul Stacks submission checklist

This checklist maps the five required deliverables and six judging categories in
`Online_Physical_AI_Challenge_Online (1).pdf` to retained evidence. Empty hardware
fields must remain empty until the final Intel Core Ultra Series 2/3 run.

## Required deliverables

- [x] Reproducible GitHub repository: pinned Python and npm dependencies, submodule,
  Dockerfile, CI, setup commands, source, tests, training, evaluation, and inference.
- [x] Reproducible MuJoCo simulation: dual SO-101 scene builders, randomized scenario
  configuration, fixed seeds, isolated evaluators, and recorded run contracts.
- [x] Intel benchmark program: `bench/intel_bench.py` measures a real OpenVINO IR,
  retains raw timings, p50/p95, throughput, device, precision, model hashes, and host.
- [x] Demonstration video: one unassisted full-task recording for every predeclared
  seed 1001–1010; the 24 FPS compilation and result manifests are in `docs/evidence`.
- [x] Technical README and architecture summary: see `README.md`,
  `docs/implementation-status.md`, and `docs/adaptive-recovery.md`.

## Judging evidence

| Category | Current evidence | Final action |
| --- | --- | --- |
| Task completion and bimanual manipulation (30) | Seven physical-contact steps, passive drawer, four placed objects, direct A-to-B spoon handoff, independent evaluator, 10/10 fixed-seed headless and recorded suites | Replay the primary uninterrupted run during judging |
| VLA / multi-modal reasoning (20) | Natural-language task graph, raw RGB object localization, robot-joint history, action context, learned ACT mug experiment, visual recovery trigger | State clearly that the winning full-task controller is the validated hybrid deterministic baseline; do not call it an end-to-end VLA |
| Robustness and generalization (15) | Predeclared seeds, randomized position/mass/friction, displaced and reduced-friction presets, matched recovery comparison, ten successful seed videos with no selective reruns | Present the retained results table and recovery comparison |
| OpenVINO and Intel optimization (20) | Real ACT export with numerical parity and benchmark tooling | Run reference and chosen optimized precision on Intel Core Ultra Series 2/3; retain paired closed-loop outcomes |
| Technical quality and reproducibility (10) | Locked dependencies, Dockerfile, submodule, CI, tests, checksums, immutable run archives | Verify Docker on a clean machine |
| Innovation and demonstration (5) | Cost-selected bounded recovery, persistent failure memory, dependency graph, resource locks, safe parallel drawer/mug execution | Show the seed-1001 recovery timeline and matched 9/10 vs 10/10 comparison |

## Final Intel commands

Install the learning dependencies and create or copy the exported ACT model. On the
actual target, replace `2` with `3` only when the machine is Series 3.

```bash
python3 -m pip install -r requirements-learning.txt
python3 bench/intel_bench.py --device CPU --precision f32 --intel-core-ultra-series 2
python3 scripts/evaluate_contact_policy.py \
  --model artifacts/models/contact_act_final --backend openvino \
  --output artifacts/learned_checks/intel-reference
```

Run the benchmark again for each supported device or precision that will be claimed.
Do not label a result optimized until the same frozen evaluation seeds preserve task
quality. The benchmark intentionally records `quality_preservation_verified=false`;
update the retained submission evidence only after the paired evaluation is reviewed.

## Ten-seed recording protocol

Use a fresh archive so every seed has one visible, unassisted attempt and failures
remain visible:

```bash
for seed in 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010; do
  MUJOCO_GL=osmesa python3 -m mise.cli full \
    --seed "$seed" --output artifacts/submission-runs
done
python3 scripts/build_submission_video.py --check-only
make submission-video
```

The default compilation accelerates each complete run 6x and interpolates the
presentation copy to 24 FPS; original recordings remain unchanged. The video
manifest binds every segment to its run ID, seed, preset, playback speed, and SHA256.
Present one uninterrupted full run in the main pitch and make the complete ten-seed
compilation available with the repository or submission link.
