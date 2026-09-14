# Revised MISE implementation

The source of requirements is `MISE_project_description.docx`. Its complete paragraph
and table text is retained in [project-brief.txt](project-brief.txt); the DOCX remains
the reference for the two figures. Performance and success targets are goals, not results.

## First integrated milestone

The console uses React, TypeScript and Vite, served with FastAPI. A separate spawned
process runs MuJoCo. The worker writes durable events and video; a bounded queue
delivers display frames without waiting for a browser. Each run gets a unique ID,
seed, command, scene/configuration hashes, hardware identity and artifact checksums.
The cameras are actual MuJoCo renders. No model inference is implemented in the fixture.

The legacy drawer expert operates a separate drawer actuator. Its arm motion is
not evidence of opening the drawer through contact. It remains useful for checking
rendering, pause/stop, telemetry, recordings and replay, and is explicitly labeled
`drawer_fixture`. Its outcome never counts toward the ten-seed full-task target.
Collection and fork-label jobs built on this expert are integration fixtures too;
they must not be used as evidence of robot-grasp or deployed recovery capability.

The planner supports a disclosed grammar and dependency graphs. Runtime contracts
reserve arms and objects, preserve dependencies and bound recovery. The physics
evaluator is isolated from the observation contract, which exposes RGB and joints.
The full shared-scene expert now validates the configured contact geometry against
the complete task. Hardware geometry calibration remains separate.

`set the table` produces independent drawer, plate and mug nodes, with drawer and
spoon-transfer dependencies on the utensil branches. Explicitly ordered clauses
produce the corresponding ordered graph. The second arm's base now faces into the
workspace; cyan and violet materials identify the arms. Wrist views remain attached
to their physical gripper bodies.

## Contact manipulation milestone

The default console now executes one real physical skill: mug pick/place with
arm B to the upper-right. The controller localizes the blue cylinder in overhead
RGB, projects that observation using the known camera calibration, and solves
robot-only IK. Cartesian trajectories keep the fingers vertical through approach,
lift and placement; a clearance waypoint avoids the inner reach limit. The passive
44 mm cylinder moves only through MuJoCo contacts, with no attachment, object
actuator, teleport or applied object force during execution.

The dedicated scene raises the mounts by 10 cm and replaces concave jaw collision
hulls with finite frictional pads. These are disclosed simulation approximations,
not validated hardware geometry. Other objects remain contextual and unsupported.
There is no drawer actuator in this scene. The original fixture remains available.

A separate read-only verifier requires simultaneous contact from both pads,
sustained airborne lift, and a released, upright, table-supported placement within
2 cm for one second. It cannot change targets or feed object poses into control.
The controller also checks the final RGB location. Only both checks together can
produce `contact_skill_success=true`; this standalone skill retains its own scope.

Actual physics regression covers seeds 1001–1004, including a 2 cm displaced start
and reduced object friction. It checks that the controller never mutates object
state, that no grasp equality constraints or object forces exist, and that merely
being at the goal cannot produce grasp/lift evidence. These are development tests,
not the ten-seed full-task evaluation. Runs retain all actions, phases, camera
localizations and separate contact verification.

## Complete shared-scene milestone

`Set the table.` now executes the canonical seven-step graph in one passive MuJoCo
scene. Arm A physically opens the drawer and places the plate, retrieves the spoon,
and presents it in the shared workspace. Arm B establishes bilateral contact before
arm A releases, retains the spoon, and places it on the right. Arm A retrieves and
places the fork, and arm B places the mug upright in the upper-right.

The controller derives object and handle targets from overhead RGB fiducials plus
robot joint history. It does not read evaluator object poses or contacts and never
writes free-object state. The scene has twelve robot actuators, no grasp equality
constraints, and no drawer actuator. The isolated evaluator requires drawer contact
by arm A, both utensil retrievals, direct A-to-B handoff provenance, four stable
released table placements, and zero forbidden collisions.

Production worker run `d47ac0311efe` passed all predicates in 138.30 simulated
seconds and 152.60 wall seconds on the Xeon development host. It retained a complete
trace, checksum manifest, timing CSV, and 70 frames from each of three cameras.
This is verified deterministic expert behavior, not learned-policy success or Intel
Core Ultra validation.

The fixed-seed headless suite now passes 10/10 across seeds 1001–1010 with bounded
object-position, mass and friction variation. Seed 1001 triggers a camera-observed
spoon goal miss and executes one arm-B table regrasp from the current state. Live run
`6b3a1c29e9d7` records `failure_detected`, `recovery_selected` and
`recovery_succeeded`, completes every step, and reports zero collisions. Persistent
recovery-memory ranking is now connected to two registered physical correction
profiles. Live outcomes update a versioned SQLite context; failed candidates are not
repeated in the same unchanged episode.

The matched seeds 1001–1010 were rerun with recovery disabled, with a blind repeat,
and with monitored cost-selected recovery. The first two variants passed 9/10. The
monitored variant passed 10/10 with one recovery attempt. Raw results are retained in
`artifacts/evaluations/recovery_matched_seeds.json` and CSV.

## Team suggestions adopted

**Predict the best recovery and account for cost.** Represent actual candidate
actions with their evidence, resource requirements and estimated cost. First reject
unregistered or infeasible candidates and any that violate required arms or completed
goals. Rank the remaining candidates using explicit success/cost estimates. Stop when
none is viable. The initial selector is deterministic; learned candidate-conditioned
predictions require student-rollout data and validation before replacing estimates.
The initial cost score is `(duration_s + extra_cost) / success_rate`, after applying
a minimum success threshold and rejecting actions that exceed the remaining budget.
It is a ranking heuristic, not a calibrated prediction of bounded-episode wall time.
The retained comparison uses the same seeds, scene, verifier and time budget and
reports completion, duration and recovery attempts for all three variants.

**Dependency graphs and independent arms.** Keep plans as DAGs; a sequence is a
special case. Arm/object locks and dependencies identify independent work. For the
goal-level command, arm A now opens the drawer while arm B places the mug in one
merged 30 Hz physical control group. All ten fixed seeds pass the independent
collision and task evaluator with this pair. Other combinations remain locked until
their geometry and holding behavior are validated. Hand-off reserves both arms. An
arm switch remains disallowed when it conflicts with the instruction.

## Remaining deployment milestones

1. Run the completed OpenVINO benchmark on the Intel target, then retain paired
   reference/optimized closed-loop outcomes before claiming preserved quality.
2. Expand the current nominal suite only if time permits; keep new stress results
   separate from the frozen hackathon evidence.

The ten-seed camera suite is complete: all ten unassisted runs passed with zero
forbidden collisions. The 24 FPS accelerated compilation, selection manifest, and
per-seed results are retained under `docs/evidence`.

Physical recovery, deterministic camera verification, and the validated drawer/mug
parallel group are complete in the declared simulation envelope. Reliable learned
full-task control, broader concurrent combinations, and an Intel target demonstration
are not complete.

The spoon correction was subsequently tightened so arm B verifies from a nearby
camera-clear pose instead of parking and returning after a miss. After a verified
placement, arm B parks concurrently with arm A's fork motion. The updated controller
still passes seeds 1001–1010 (10/10); seed 1001 decreases from 138.8 to 138.3 simulated
seconds. The retained camera compilation remains immutable evidence from the earlier
validated controller revision, while the updated headless report is retained locally
as `artifacts/evaluations/full_task_post_recovery_optimization.json`.
Recorded run `102d95237489` verifies the updated path with cameras and evidence:
autonomous full-task success, one recovery, zero forbidden collisions, 138.3 simulated
seconds, and all seven manifest file hashes valid.
No public live relay or deployment is configured. Public/replay use is read-only.

## Runtime limits to measure

The worker captures three 256×256 views. Standalone skills use 3 FPS; the full task
uses 2 FPS for the overhead view and 0.25 FPS for each wrist. The task retains its
180-second simulation-time budget, with a separate 300-second stalled-worker wall guard.
Software rendering and encoding still consume worker time; display and recording
overhead are measured in run timing CSVs, and the nominal rate is not a measured
throughput promise. A browser queue overflow drops display delivery only. Events,
recordings and final outcomes remain on disk. Pause freezes simulation at a control
boundary and marks the run assisted. A 300-second stalled-worker wall guard applies.
Stopping does not reset the simulator or relabel an attempt as a success.

## Contact milestone verification

- Full suite: **66 tests passed**, including complete-task physics, adaptive-memory persistence, submission evidence tooling, and actual physics under four seeded
  start/friction conditions and an exclusive archive-writer check.
- TypeScript and the Vite production build passed.
- Production-worker run `d1ca00774909` passed all seven steps with the drawer/mug
  parallel group and one
  cost-selected regrasp, zero forbidden collisions, one persisted successful
  recovery outcome, 265 overhead frames, 67 frames per wrist, and a valid seven-file
  checksum manifest. Its 143.56-second wall time remains inside the 180-second budget.
- The terminal command completed run `0fb44b3b25c7` (seed 1004, low friction),
  returned exit code 0 and retained all three recordings in `artifacts/contact_runs`.
- Real browser-run `eebaca53cdfd` (seed 1003, displaced objects) passed independent
  grasp, lift and one-second released placement checks. Final physical goal error
  was 5.31 mm. Live display, replay, three-camera artifact downloads and the 390 px
  layout passed with no browser errors.
- This recorded run took 131.30 s wall time for 20.90 s simulation, including
  software rendering and all evidence. It is not real-time performance or an ACT
  inference measurement. Timing depends on concurrent work on the host.
- The earlier nominal contact run `ebf4fa122bab` also passed, with a 4.69 mm goal
  error. Its trajectory preceded the clearance-waypoint change; its original
  artifacts are preserved.
- The scene is now regenerated before a new run's configuration hash is recorded.
  Earlier records retain their original hashes; the builder source hashes identify
  the scene-generation code used for those development runs.

## Earlier console milestone verification

- `make test`: 35 tests passed across physics, evaluator, planning, recovery,
  scheduling, data contracts and API behavior.
- `npm run build --prefix web`: TypeScript and production bundle succeeded.
- Chromium checks passed at 1440 px and 390 px: all views, recorded replay, camera
  switching and artifact downloads; no page errors or horizontal overflow.
- Browser-driven pause/resume/stop produced an assisted, stopped record with all
  operator events retained. A completed physical fixture produced three camera
  recordings and a passed drawer travel/hold predicate, with full-task success null.
- The initial software-rendered fixture took about 40 s for 4 s of simulation on
  this Xeon host. This is a local fixture measurement, not an Intel Core Ultra or
  model-inference benchmark. Individual run CSVs retain their exact measurements.
- Docker packaging was updated; a Docker engine is unavailable in this workspace,
  so the container build and clean-machine reproduction remain unverified.

Implementation reference: [FastAPI WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
and [lifespan management](https://fastapi.tiangolo.com/advanced/events/).

## Final audit — 2026-09-14

Current branding is **SeoulStack**; original recordings retain historical labels.
The final local submission gate passes **79 tests**, the production frontend build,
and all 70 original worker artifact checksums. The evidence-check command is now
read-only, and new compilation selection requires one attempt per declared seed.
The optimized 10/10 numerical report is published under
`docs/evidence/optimized-full-task-results.json`; its source hash identifies the
recorded motion revision. New validation and bookkeeping changes preserve that
trajectory.

The Docker image now uses a source-relative editable install, and hosted CI builds
and smoke-tests the actual image. Local Docker execution remains restricted.
Shared OpenVINO compilation settings and a strict paired-evidence checker are
implemented; see `docs/intel-validation.md`. A one-seed Xeon development pair passed
with two and four CPU inference threads. This does not establish ten-seed quality
preservation or Intel target compliance. The available Core Ultra 7 155H is Series 1;
Series 2/3 or organizer acceptance remains necessary for the declared final target.
