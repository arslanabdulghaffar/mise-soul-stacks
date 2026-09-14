# SeoulStack final demonstration guide

This is the recording plan for the MISE hackathon submission. Use the validated
runtime and retained evidence; do not improvise a new controller during recording.

## Recording setup

1. Run `make submission-check` before recording.
2. Start the console with `make console`, or use `make replay` when presenting the
   retained evidence on a machine that cannot run MuJoCo smoothly.
3. Record at 1080p with the browser near 1440 × 900. Keep the overhead camera visible
   for the full task and use wrist-camera clips only as short supporting views.
4. Select **Contact expert**, **Nominal scene**, **Monitored · cost selected**, seed
   **1001**, and enter `Set the table.`
5. Do not pause or manually intervene in the primary run because the evidence will
   correctly label that attempt as assisted.
6. Capture the Intel benchmark panel only after running on an actual Intel Core Ultra
   Series 2 or 3 machine. Leave it marked pending until then.

## Three-minute narration and shot list

**0:00–0:20 — Problem and instruction.** Show the Live run page and command box.

> We are SeoulStack. MISE converts a natural-language goal into safe, observable
> bimanual manipulation. Here the user asks two simulated SO-101 arms to set the
> table in one shared MuJoCo scene.

**0:20–0:45 — Plan and parallel execution.** Show the dependency graph, then the
drawer and mug moving together.

> The planner validates a dependency graph rather than a fixed video sequence. It
> reserves arms, objects and workspace zones. Independent work runs together: arm A
> opens the passive drawer while arm B places the mug. Dependent utensil work waits
> for the drawer, and the spoon handoff reserves both arms.

**0:45–1:35 — Physical task.** Show plate placement, spoon retrieval and the direct
A-to-B handoff.

> Objects are passive. The controller uses overhead RGB, robot joints, the command
> graph and action history. It never teleports objects or reads evaluator-only poses.
> Arm A places the plate, retrieves the spoon, and presents it. Arm B establishes
> contact before arm A releases, creating a verified physical handoff.

**1:35–2:05 — Recovery.** Use seed 1001 and show the failure and correction timeline.

> When the spoon misses its goal, the visual verifier detects the failure immediately.
> The supervisor ranks registered corrections by measured success and estimated cost,
> continues from the current state, and records the outcome in persistent memory. The
> arm clears the camera, regrips the spoon directly, verifies the correction, and parks
> concurrently while arm A starts the fork step.

**2:05–2:30 — Independent verification.** Show Full-task evidence and the Evidence
page.

> A separate evaluator checks drawer contact, utensil retrieval, handoff provenance,
> stable released placements and forbidden collisions. Ten predeclared camera runs
> passed ten out of ten with no assistance and zero forbidden collisions. A matched
> comparison improved from nine out of ten without monitored recovery to ten out of
> ten with it.

**2:30–2:50 — Intel/OpenVINO.** Insert the measured target-hardware panel and replace
the bracketed values only with values produced by `bench/intel_bench.py`.

> The experimental ACT mug policy is exported as a real OpenVINO IR. On our Intel
> Core Ultra Series [2/3] target, the measured [device and precision] configuration
> achieved [p95 latency] with [paired quality result].

If target measurements are unavailable, say this instead:

> The OpenVINO export and benchmark pipeline are complete. Target Intel Core Ultra
> measurements remain pending, so we do not present development-host numbers as
> target results.

**2:50–3:00 — Close.** Show the final table and the repository.

> MISE combines transparent planning, physical bimanual execution, independent
> verification and recovery that learns from prior outcomes. Every run retains its
> seed, configuration, trace, videos and checksums for review.

## Claims to use

- Complete seven-node, physical-contact table-setting task in simulation.
- Direct contact-verified A-to-B spoon handoff.
- Camera-guided deterministic full-task controller with an experimental learned ACT
  mug policy exported to OpenVINO.
- Ten out of ten predeclared recorded seeds, unassisted, with zero forbidden
  collisions.
- Cost-selected bounded recovery with versioned persistent outcome memory.
- Validated drawer/mug parallel execution using DAG resource contracts.

Do not call the successful full-task controller an end-to-end learned VLA. Do not
claim Intel Core Ultra validation, optimized precision, or preserved learned-policy
quality until the paired target-machine evidence exists.
