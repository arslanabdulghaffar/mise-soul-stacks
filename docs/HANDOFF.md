# SeoulStack project handoff

Use this document when continuing the project in a new Codex or ChatGPT account.
The repository is the durable source of context; chat history and account memory are
not expected to transfer.

## Latest continuation — 2026-09-15

Read [final-audit.md](final-audit.md) first. The camera synchronization fix and
new robustness/learned-policy evaluations supersede earlier readiness estimates.
The successful original nominal demonstrations do not establish generalization.
Do not say only deployment remains: the learned policy and shape robustness are
not reliable yet. Intel demonstration is explicitly deferred by the user.

## Project identity

- Team: **SeoulStack**
- Project: **MISE — Multi-modal Instruction to Skill Execution**
- Repository: <https://github.com/arslanabdulghaffar/mise-soul-stacks>
- Challenge: Intel Physical AI Online Challenge, dinner-table option
- Runtime: Python 3.12, MuJoCo, FastAPI, React/TypeScript/Vite, OpenVINO
- Robot: two simulated SO-101 arms using the upstream `vendor/SO-ARM100` submodule

The authoritative challenge text is `Online_Physical_AI_Challenge_Online (1).pdf`.
The revised design is `MISE_project_description.docx`; its searchable text is
`docs/project-brief.txt`.

Current synchronized recording: `artifacts/final-runs/2f710b64aab8` completed
all seven steps with one recovery and zero collisions. All five videos contain
277 matching-time frames at 2 FPS; 138.3 simulation seconds took 204.00 wall seconds.
See `docs/evidence/synchronized-camera-validation.json`. The original lower-rate
close-up recording was faster; do not promise unchanged wall-clock speed.

## Implemented and verified

The console accepts supported natural-language commands and produces a dependency
graph with arm and object resource contracts. `Set the table.` executes the complete
seven-step physical-contact task in one shared MuJoCo scene:

1. Arm A opens the passive drawer.
2. Arm A places the plate.
3. Arm A retrieves and presents the spoon.
4. Arms A and B perform a contact-verified handoff.
5. Arm B places the spoon vertically on the right.
6. Arm A places the fork vertically on the left.
7. Arm B places the mug in the upper right; for the goal-level command this node is
   scheduled concurrently with step 1.

The isolated evaluator verifies every goal, handoff provenance, stability, and
forbidden collisions.

Control uses calibrated RGB fiducials and robot-joint/action history. Evaluator
object poses and contacts do not enter controller observations. Free objects are not
teleported, attached to grippers, directly actuated, or forced during execution.

The three team suggestions are implemented:

- recovery candidates are feasibility-filtered and ranked by expected cost;
- task plans are DAGs, with validated drawer/mug parallel execution and resource locks;
- versioned SQLite recovery memory changes later candidate ranking and prevents the
  same failed correction from repeating in an unchanged episode.

Current retained evidence:

- 83 Python tests pass locally, including actual MuJoCo contact, camera isolation,
  and evaluator regressions;
- the React/TypeScript production build passes;
- fixed headless full-task suite: 10/10 seeds pass;
- recorded camera suite: 10/10 predeclared seeds pass, zero forbidden collisions,
  no assistance, and no selective reruns;
- matched recovery comparison: none 9/10, blind retry 9/10, adaptive 10/10;
- all 70 files in the ten recorded run manifests passed checksum verification;
- `docs/evidence/ten-seed-demo.mp4` is the compact GitHub preview;
- the full 3:46, 256×256, 24 FPS compilation is local at
  `artifacts/submission/ten-seed-demo.mp4`.

After the final spoon-flow optimization, recorded seed-1001 run `102d95237489`
completed autonomously in 138.3 simulation seconds with one direct recovery, zero
forbidden collisions, and seven valid artifact checksums. It is retained locally at
`artifacts/final-runs/102d95237489`; the arm verifies from a nearby camera-clear pose
and parks concurrently with the fork motion instead of parking before recovery.

## Honest capability boundary

The successful full-task controller is a deterministic camera-guided hierarchical
baseline with adaptive recovery. Do not describe it as an end-to-end learned VLA.
A real compact ACT-style contact policy was trained and exported to OpenVINO for the
arm-B mug skill, but it did not generalize reliably enough to replace the full-task
expert. The earlier retained pilot passed 3/10 test seeds. A complete audit of
`contact_act_final` passed 2/10 test seeds. A backbone-refined checkpoint passed
1/5 development validation seeds despite better action-prediction error. A
five-step temporal-ensemble pilot passed 0/5 validation seeds. These experiments
are retained locally and are not replacements for the validated expert.

Physical SO-101 hardware is not required by the online challenge. Final MuJoCo and
OpenVINO execution on Intel Core Ultra Series 2 or 3 is required for the 20-point
Intel category and has not yet been performed.

## Remaining work

1. Improve and independently validate learned-policy execution and robustness;
   the retained stress matrix contains failures and must not be replaced or hidden.
2. On an Intel Core Ultra Series 2/3 machine, install `requirements-learning.txt`,
   generate or copy the exported ACT model, and run `bench/intel_bench.py` with the
   correct `--intel-core-ultra-series` value.
3. Run reference and proposed optimized configurations on identical frozen seeds.
   Only mark quality preservation verified after reviewing paired task outcomes.
4. Record the narrated pitch using `docs/demo-guide.md`; its script and shot list are
   prepared, but the Intel segment must wait for measured target results.
5. A public interactive runtime is optional. GitHub Pages cannot host FastAPI or
   MuJoCo; any public site must be a labeled read-only replay unless backed by a
   suitable server.

The user now has a Core Ultra 7 **155H**, which Intel identifies as **Series 1**.
It is useful for development measurements, but the PDF's final demonstration
requirement specifies Series 2/3. Do not label this laptop as compliant without an
organizer-approved exception. Hardware work is deferred until the local work is
finished, as requested. Use `docs/intel-validation.md` for the paired procedure.

Latest code audit fixes: full-task validation rejects unsupported dependency orders;
aborted/occluded recoveries retain unknown outcomes without influencing memory;
video `--check-only` verifies all original worker artifact hashes without overwriting
the compilation manifest; and the Docker image preserves source-relative runtime
assets. Shared OpenVINO compile settings now connect benchmark and actual learned
execution. One Xeon paired smoke passed for seed 40000 with two and four CPU threads;
this is not a ten-seed or target-hardware quality claim.

The team name is now **SeoulStack**. Existing recordings retain their original labels
and provenance. The repository URL is unchanged. Higher display resolution and two fixed side cameras are now implemented. The
browser defaults to Fast live (`economy`, 256×256) for shorter render times;
All five views now capture together: Fast live uses 256×256 overhead and
160×160 insets at 2 FPS; Balanced uses 384×384/256×256 at 3 FPS; Detail uses
720×720/384×384 at 2 FPS. Historical archives retain their original rates. All five cameras are recorded. A separate display
renderer keeps calibrated controller images unchanged. Camera FPS refers to
simulation time; live wall speed depends on rendering hardware.

Do not expand into pouring, accounts, voice control, another policy backend, or
unvalidated parallel arm combinations before the required Intel evidence is secure.

## First commands in a new workspace

```bash
git submodule update --init --recursive
make setup
make test
npm run build --prefix web
MUJOCO_GL=osmesa python3 -m mise.cli serve --host 0.0.0.0 --port 8000
```

Open or forward port 8000 and execute `Set the table.` with the contact expert.
Use `make submission-video` only after the dedicated ten-seed source archive is
available; the generated artifacts directory is intentionally excluded from Git.

## Files to read before editing

1. `README.md` and `docs/final-audit.md`
2. `docs/HANDOFF.md`
3. `docs/implementation-status.md`
4. `docs/submission-checklist.md`
5. `docs/adaptive-recovery.md`
6. `Online_Physical_AI_Challenge_Online (1).pdf`

The central implementation paths are:

- planning and scheduling: `src/mise/planner.py`, `scheduler.py`, `supervisor.py`;
- physical full task: `src/mise/full_task.py`, `full_vision.py`, `worker.py`;
- evaluation and evidence: `src/mise/evaluation.py`, `telemetry.py`, `server.py`;
- adaptive recovery: `src/mise/recovery_memory.py`, `recoverability.py`;
- learning and export: `src/mise/learning/`, `scripts/train_contact.py`,
  `scripts/export_contact.py`, `scripts/evaluate_contact_policy.py`;
- Intel benchmark: `bench/intel_bench.py`;
- demonstration builder: `scripts/build_submission_video.py`;
- final recording script: `docs/demo-guide.md`;
- frontend: `web/src/App.tsx` and `web/src/styles.css`.

## Safe continuation rules

- Inspect `git status`, recent commits, and existing tests before changing code.
- Preserve the observation/evaluator boundary and immutable run evidence.
- Never hide failed seeds, overwrite attempts, invent benchmark values, or claim
  Intel verification from the current Xeon development host.
- Do not replace the validated controller with a learned checkpoint unless a frozen
  closed-loop evaluation demonstrates that it is at least as reliable.
- Run relevant tests and the frontend production build, update documentation, then
  commit and push through the attached GitHub connector.

## Copy/paste prompt for a new Codex account

> Continue the SeoulStack MISE project from the repository
> https://github.com/arslanabdulghaffar/mise-soul-stacks. First clone/open the
> repository with submodules and read README.md, docs/HANDOFF.md,
> docs/implementation-status.md, docs/submission-checklist.md, and the challenge PDF.
> Inspect git status and the latest GitHub Actions result before editing. Treat the
> repository documentation and retained artifacts as authoritative; do not infer
> capabilities from chat history. Preserve the camera-observation/evaluator boundary,
> immutable evidence, fixed seeds, physical-contact constraints, and honest claim
> limits. Summarize the verified current state and remaining blockers before making
> changes. Then continue only the remaining work, starting with Intel Core Ultra
> OpenVINO validation and paired quality measurements. Do not redesign completed
> architecture or replace the validated full-task controller without better frozen
> closed-loop evidence.

## Hosted verification completed

[GitHub Actions run 34852939210](https://github.com/arslanabdulghaffar/mise-soul-stacks/actions/runs/34852939210)
passed all three jobs: 81 Python tests, frontend production build, and actual Docker
image build plus a live physical mug smoke run. The container run `bb10ee996c98`
completed autonomously in 66.38 wall seconds and verified seven artifact checksums.
The tested GitHub code revision is `701170de23a7226273cb3a2b6a7b02c5ce0c1426`
(local equivalent `2d267d7`). The newer camera presentation changes have separate
local regression and browser checks; see the camera validation record below.
Docker packaging validation is complete.

## Historical camera presentation validation

The presentation update passes 83 Python tests and the production frontend build.
Browser checks passed for desktop/mobile layout, all five live MJPEG streams,
archived-camera availability, and preserving the replay playhead when switching views.
The original ten-seed archive still verifies all 70 artifact checksums.

Balanced recording `20ab2f249d42` (seed 1001) completed all seven steps without
assistance, with one spoon recovery and zero forbidden collisions. It retained five
384×384 videos and nine verified artifact hashes. The original 138.3 simulation-second
trajectory took 395.93 wall seconds to render on this development host.
This is a single presentation regression, not a replacement ten-seed or Intel result.
See `docs/evidence/camera-validation.json`. Detail-mode rendering at 720×720 also
passes the camera/physics isolation test; no full Detail-mode episode is claimed.

## Historical combined camera view validation

The console now shows one large camera with the other four below it. Clicking a
small view promotes it. Recorded views follow the main view's play/pause, seek, and
playback rate; old archives show missing angles as unavailable. The layout reuses
existing frames and does not add simulation render passes. Fast live (`economy`) is
the new browser default because Balanced's five-camera capture took 395.93 seconds
on this host. Higher quality remains selectable for the next run.

Validation: production build and browser checks passed for five simultaneous live
streams, synchronized replay play/pause/seek/rate, thumbnail promotion, and mobile
layout. Fast-live seed-1001 run `33147dc3081c` completed the full task autonomously
with one recovery and zero collisions in 110.77 wall seconds (138.3 simulation
seconds), versus 395.93 wall seconds for prior Balanced run `20ab2f249d42`. All nine
artifact hashes verify. This compares two individual development-host runs, not a
controlled benchmark or Intel measurement. See `docs/evidence/fast-live-validation.json`.
