# SeoulStack submission checklist

This checklist maps the five required deliverables and six judging categories in
`Online_Physical_AI_Challenge_Online (1).pdf` to retained evidence. Empty hardware
fields must remain empty until the final Intel Core Ultra Series 2/3 run.

This is not a claim of maximum points or complete submission readiness. See
[the current audit](final-audit.md) for measured learned-policy and robustness gaps.

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
- [x] Narration and shot list prepared in `docs/demo-guide.md`; final capture awaits
  the target-hardware measurement insert.

## Judging evidence

| Category | Current evidence | Final action |
| --- | --- | --- |
| Task completion and bimanual manipulation (30) | Seven physical-contact steps, passive drawer, four placed objects, direct A-to-B spoon handoff, independent evaluator, 10/10 fixed-seed headless and recorded suites | Replay the primary uninterrupted run during judging |
| VLA / multi-modal reasoning (20) | Natural-language task graph, raw RGB object localization, robot-joint history, action context, learned ACT mug experiment, visual recovery trigger | The validated full-task controller is deterministic. Reliable learned full-task VLA execution is not demonstrated |
| Robustness and generalization (15) | Predeclared seeds, randomized position/mass/friction, displaced and reduced-friction presets, matched recovery comparison, ten successful seed videos with no selective reruns | Additional stress audits passed 18/50 initially and 24/50 with an intermediate perception change; altered shape and combined variants passed 0/10 each. See final-audit.md for final-source validation and retained failures |
| OpenVINO and Intel optimization (20) | Real ACT export with numerical parity and benchmark tooling | Run reference and chosen optimized precision on Intel Core Ultra Series 2/3; retain paired closed-loop outcomes |
| Technical quality and reproducibility (10) | Locked dependencies, Dockerfile, submodule, CI, tests, checksums, immutable run archives | Hosted image build and autonomous mug smoke passed; seven artifact hashes verified |
| Innovation and demonstration (5) | Cost-selected bounded recovery, persistent failure memory, dependency graph, resource locks, safe parallel drawer/mug execution | Show the seed-1001 recovery timeline and matched 9/10 vs 10/10 comparison |

## Camera presentation update

New console runs offer Economy (256×256), Balanced (384×384), and Detail
(720×720), with overhead, two wrist, and two fixed arm-side views. These are
presentation settings; the calibrated controller observations stay unchanged.
New runs capture every view at the same instant and rate, with smaller inset
frames: Fast live 256/160 pixels at 2 FPS, Balanced 384/256 at 3 FPS, Detail
720/384 at 2 FPS. Capture rates and per-camera dimensions are retained in each run. Higher display quality is
not evidence of improved task success or Intel performance. Historical recordings
are preserved at their original resolution.

## Final Intel commands

Use [the complete Intel validation procedure](intel-validation.md) for matching
reference/candidate device, precision and thread settings and evidence verification.
The user's Core Ultra 7 155H is **Series 1**, so it must not be marked Series 2/3.
It can be used for development tests while an eligible target or organizer exception
is arranged. This category is scored on evidence, not hardware ownership alone.

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
`scripts/validate_intel_pair.py` writes a separate checked result after paired
evaluation without modifying either original measurement.

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

Hosted verification: [run 34852939210](https://github.com/arslanabdulghaffar/mise-soul-stacks/actions/runs/34852939210)
passes 81 Python tests, the frontend build, and the actual Docker runtime smoke test.
