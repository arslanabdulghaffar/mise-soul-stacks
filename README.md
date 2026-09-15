# MISE · Robotics Console

Built by **SeoulStack**. The existing repository URL remains `mise-soul-stacks`.

**Multi-modal Instruction to Skill Execution** — two simulated SO-101 arms,
verified table-setting outcomes, bounded recovery, and inspectable run evidence.
The updated requirements are in `MISE_project_description.docx`.

The current build runs the **complete camera-guided physical table-setting task**
from the console. Enter `Set the table.` to execute seven graph steps: arm A opens
the passive drawer and places the plate, retrieves the spoon, transfers it to arm B,
then both arms finish the spoon, fork and mug placements. A separate evaluator checks
contact provenance, the A-to-B handoff, stable released goals and collisions.

**Submission evidence:** [watch the 10-seed demonstration](docs/evidence/ten-seed-demo.mp4)
or inspect the [per-seed results](docs/evidence/ten-seed-results.csv). The fixed,
predeclared suite passed **10/10** with zero forbidden collisions and no assisted runs.
The final recording and narration procedure is in [the demonstration guide](docs/demo-guide.md).

With **Contact expert** selected, `Open the drawer with arm A.` runs a second
physical skill. Arm A localizes the yellow knob from RGB, grasps it, pulls the
passive drawer and releases it. The independent verifier requires contact during
the pull and at least 80% travel held open after release. Use the nominal preset
for this standalone skill. `Place the mug in the upper-right with arm B.` remains
available as another focused physical check.

The complete run is a deterministic camera-guided engineering expert with approximate
finger pads and a solid 44 mm mug cylinder. A separately selectable **Learned ACT ·
mug only** controller is available when its validated model bundle is installed: it
uses three RGB cameras, robot joints, prior commanded actions, and elapsed skill time.
Its frozen OpenVINO policy passed 10/10 fresh held-out mug trials and a recorded
console run. It is deliberately scoped to the mug skill and is not a learned
full-table policy. The legacy drawer actuator fixture remains separately selectable
and labeled.

## Start the console

Python 3.12, Node.js 22.12+ (24 used here), and the official robot submodule are
required. On a headless Debian/Ubuntu machine install `libosmesa6` and `libgl1`.

```bash
git submodule update --init --recursive
make setup
make console
```

For a remote IDE or container, start the API with
`MUJOCO_GL=osmesa python3 -m mise.cli serve --host 0.0.0.0 --port 8000`,
then forward port **8000** in the IDE’s **Ports** panel and open its forwarded URL.
The workspace’s `127.0.0.1` address is local to that workspace, not your laptop.

Open **http://127.0.0.1:8000**. Start `Set the table.` with a selected
seed. Switch overhead, wrist, and arm-side cameras, pause/resume at simulation step boundaries,
or stop the attempt. Completed and failed runs are retained under `artifacts/runs`.
Pauses mark a run assisted. Operator actions appear in the trace. `constraints.txt`
pins the tested Python environment; `web/package-lock.json` pins frontend dependencies.

The console shows one large camera and four smaller views together. Click a small
view to enlarge it; replay play/pause, seeking, and playback speed stay synchronized.
The combined layout reuses the existing captured frames.

Camera quality applies to the next run: **Fast live** (the browser default, API
`economy`) captures a 256×256 overhead view and 160×160 close-ups at 2 FPS.
**Balanced** captures 384×384 overhead and 256×256 close-ups at 3 FPS.
**Detail** captures 720×720 overhead and 384×384 close-ups at 2 FPS.
All five views capture the same simulation instant at the same rate. These are real
frames per simulation second, not promised live wall-clock frame rates. Software
rendering can take longer than the simulated episode. All five views are recorded
and checksummed; switching replay cameras retains the playback position. Older
recordings keep their original resolution and expose only their recorded views.
Display rendering is separate from the unchanged 256×256 controller observations.
Software rendering defaults to one Mesa thread to avoid oversubscribing cloud CPUs;
set `LP_NUM_THREADS` explicitly to benchmark a different value.

Read [the current audit](docs/final-audit.md) before describing the project as
submission-ready: broader full-task robustness has measured gaps, and final Intel
validation is pending.

For frontend development, run `python3 -m mise.cli serve` and, in another terminal,
`cd web && npm run dev`; Vite proxies the API and WebSocket to port 8000. Rebuild
with `make web` after frontend changes when using the compiled console.

```bash
make replay           # serve existing evidence with all operator commands disabled
make demo             # record the legacy drawer fixture to MP4
make contact          # one physical mug run; evidence in artifacts/contact_runs
make full             # one complete seven-step run; evidence in artifacts/full_runs
make eval             # ten retained full-task attempts from eval/seeds.yaml
make eval-headless    # faster fixed-seed physics regression with JSON/CSV results
make check-plan       # deterministic scheduling checks only
make test             # planner, recovery, evaluator, API and simulation checks
make bench            # measured OpenVINO latency/throughput and host identity
make submission-video # concatenate ten verified full-task seed recordings
make submission-check # tests, production UI build, and retained evidence validation
make data EPISODES=20 # aligned drawer fixture data (not robot grasp demonstrations)
make planner-data PLANNER_SAMPLES=100
make labels LABEL_SAMPLES=12 # privileged fixture-label plumbing, not deployed recovery evidence
```

For a single recorded contact run without the browser:

```bash
MUJOCO_GL=osmesa python3 -m mise.cli contact --seed 1003 --preset displaced_objects
```

The CLI exits unsuccessfully if placement is not physically verified. Its default
archive is `artifacts/contact_runs`; the console uses `artifacts/runs`. A runtime
holds an exclusive lock on its writable archive. Use `--output` to choose another
archive, or stop its current runtime before reusing it. Legacy batch checks remain
available with `python3 -m mise.cli eval --controller scripted_drawer`.

For learning, install `requirements-learning.txt` in a separate environment and
install this repository there. `scripts/collect_contact_data.py` records physical
mug demonstrations with frozen scene/source provenance and predeclared splits.
The current collection contains 30 training, 10 validation and 10 test expert
episodes; all 50 expert attempts succeeded. These are **expert** results, not
learned-policy or full-task success rates.

```bash
python scripts/collect_contact_data.py --output data/contact --workers 3
make train MODEL_DIR=artifacts/models/contact_act_local
make export MODEL_DIR=artifacts/models/contact_act_local
python scripts/evaluate_contact_policy.py --model artifacts/models/contact_act_local \
  --backend openvino --output artifacts/learned_checks/local-test
```

Evaluation uses RGB/robot-joint completion and a separate physics scorer. Every
attempt is retained. Successful export and small action error do not establish
successful manipulation: the newer 30-episode checkpoint stalled before release
on validation seed 30001, while the earlier pilot passed validation seed 30000.
The console continues to default to the tested deterministic expert.

Planner distillation, RL/MJX expansion and a second policy backend are deferred.
On a validated graphics setup use `MISE_MUJOCO_GL=egl make console`.

## What is implemented

- Camera-guided mug grasp, lift, transport and release through physical contacts;
  seeded starting positions, a clearance waypoint, and independent outcome evidence.
- Physical arm-A drawer opening, with recorded bilateral grasp and pull provenance.
- Complete shared-scene execution for plate, spoon, fork and upright mug placement,
  including a contact-verified spoon handoff from arm A to arm B.
- Dependency-graph planning with a disclosed supported grammar, explicit arms,
  timeouts, preconditions and rejection of unsupported pouring requests.
- DAG scheduling with arm/object resource contracts. The goal-level task runs the
  independent drawer and mug nodes concurrently; explicitly ordered commands remain
  ordered. Recovery outcomes are bounded to done, continue, retry, replan or stop.
- Persistent recovery-outcome memory that adapts estimates, retains unknown
  outcomes without scoring them, and excludes repeated failed corrections in an
  unchanged episode context. [Details and current limits](docs/adaptive-recovery.md).
  The full task connects this memory to two validated arm-B regrasp profiles.
- Three selectable recovery modes (`none`, `blind_retry`, and `adaptive`) and a
  matched-seed evaluator. The console displays the retained comparison artifact.
- A scene containing the drawer, plate, mug, spoon and fork; configurable goal and
  stability predicates in an evaluator that is isolated from policy observations.
- Live/replay/disconnected UI states, synchronized event identifiers, camera
  switching, pause/stop at control boundaries, artifact downloads and seed accounting.
- Real run identity, local host inventory, configuration/source hashes, JSONL
  traces, summary JSON, per-camera MP4, timing CSV and a checksum manifest.

Each worker writes complete evidence before browser delivery. A bounded display
queue drops frames if necessary; slow browsers never own the control loop.
Rendering and recording take worker time and are included in each run’s timing.
Full-task success requires every independent physics predicate, both utensil
retrievals, a verified handoff, stable placements and zero violations. The legacy
fixture and standalone contact skills keep their separate outcomes.
Standalone skills capture 3 FPS per simulated second. Full episodes capture the main
overhead view at 2 FPS and both wrist evidence views at 0.25 FPS. The task keeps its
180-second simulation-time limit; a separate 300-second wall guard allows software
rendering below real time to finish recording.
Control remains 30 Hz and physics 500 Hz; each timing CSV records the actual cadence,
wall time and real-time factor.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Host, mode and actual capability flags |
| `POST /api/plans` | Validate a supported command and preview its deterministic task graph |
| `POST /api/runs` | Start the full task, a supported contact skill, or the legacy fixture |
| `GET /api/runs`, `GET /api/runs/{id}` | Run archive and snapshot |
| `WS /api/runs/{id}/events` | Sequenced events with a reconnection snapshot |
| `GET /api/runs/{id}/camera/{name}` | MJPEG with capture timestamp headers |
| `POST /api/runs/{id}/control` | Pause, resume or stop |
| `GET /api/runs/{id}/trace` | Complete structured event trace |
| `GET /api/runs/{id}/artifacts/{file}` | Download recordings and checksum-bound evidence |
| `GET /api/evaluations`, `GET /api/benchmarks` | Recorded evidence; unmeasured outcomes stay empty |
| `GET /api/recovery-comparison` | Retained matched-seed recovery results |

The server defaults to loopback and one worker/operator. A public live-control
relay is not implemented. Use `--read-only` for evidence serving.

## Remaining deployment work

The functional simulation submission is complete: command planning, full contact
execution, camera verification, constrained recovery, persistent outcome learning,
recording, replay, and matched evaluation are integrated. Actual Intel Series 2/3
deployment and paired policy-quality measurements remain for the final hardware phase.
The current ACT experiment is retained as research evidence but is not promoted to the
full-task controller because it did not pass closed-loop validation.

The team's recovery-cost, adaptive-memory and dependency-graph suggestions are
integrated. Physical concurrency covers the validated drawer/mug pair and Arm B's
post-verification parking during Arm A's fork step;
other combinations remain resource-locked. See the
[implementation status](docs/implementation-status.md).

The required deliverables and final Intel commands are tracked in the
[submission checklist](docs/submission-checklist.md). The OpenVINO benchmark writes
machine-readable JSON and CSV under `artifacts/benchmarks`; the console reads those
files on refresh. It only marks Intel Core Ultra verification when host identity and
an explicit Series 2/3 declaration agree.
The [Intel validation guide](docs/intel-validation.md) provides matching benchmark
and closed-loop commands plus a checker for retained reference/candidate evidence.

To continue from another Codex or ChatGPT account, start with the authoritative
[project handoff](docs/HANDOFF.md); account-level chat history is not required.

## Verified complete run

The production worker integration run `d47ac0311efe` completed the full seven-step
task for seed 1001 in 152.60 s wall time and 138.30 s simulation time. Its independent
evaluator recorded all four stable placements, drawer opening by arm A, spoon and fork
retrieval, the A-to-B spoon handoff, zero forbidden collisions, and
`full_task_success=true`. It retained 70 frames from each camera plus the complete
trace, summary, timing CSV and checksum manifest under
`artifacts/full_integration_check_v2/d47ac0311efe`.

The current seeded controller was then evaluated on the predeclared seeds 1001–1010
with small bounded position, mass and friction variations: **10/10 passed** with zero
forbidden collisions. Seed 1001 visibly missed the first spoon placement, selected a
cost-selected arm-B table regrasp, recovered without resetting, and preserved the
earlier goals. Final production run `d1ca00774909` also executes the independent
drawer and mug nodes concurrently. With the smoother overhead stream, it passed in
143.56 s wall time and 132.30 s
simulation time with zero forbidden collisions, one persisted recovery outcome,
265 overhead frames, 67 frames from each wrist, and a valid checksum manifest. Raw fixed-seed results are in
`artifacts/evaluations/full_task_fixed_seeds.json` and the adjacent CSV. This small
suite validates the declared envelope; it does not establish broad generalization.

A retained earlier matched comparison ran the same ten seeds, scene, task, time budget, and
visual verifier in three modes. No recovery passed **9/10**; a blind repeat also
passed **9/10**; monitored cost-selected recovery passed **10/10** with one physical
regrasp and no reset. The raw JSON and CSV are
`artifacts/evaluations/recovery_matched_seeds.*`. These counts belong to that recorded
controller revision; the current optimized ten-seed suite uses eight recoveries and
passes 10/10. Run `make eval-recovery` to measure the current revision separately.

The final camera recording suite also passed **10/10** without selective reruns.
Its compact GitHub preview is 1 minute 5 seconds; the full 3 minute 46 second 24 FPS presentation copy,
selection manifest, per-seed JSON, and CSV are retained under `docs/evidence`.

## Container

```bash
docker build -t mise:console .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/artifacts:/app/artifacts" mise:console
docker run --rm mise:console make test
```

The image bundles the compiled frontend and the official
[SO-101 assets](https://github.com/TheRobotStudio/SO-ARM100). Clone with submodules
before building. Software rendering is a local development path;
Intel device/driver access must be tested separately on the final demonstration host.
