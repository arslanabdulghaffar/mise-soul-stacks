# Soul Stacks project handoff

Use this document when continuing the project in a new Codex or ChatGPT account.
The repository is the durable source of context; chat history and account memory are
not expected to transfer.

## Project identity

- Team: **Soul Stacks**
- Project: **MISE — Multi-modal Instruction to Skill Execution**
- Repository: <https://github.com/arslanabdulghaffar/mise-soul-stacks>
- Challenge: Intel Physical AI Online Challenge, dinner-table option
- Runtime: Python 3.12, MuJoCo, FastAPI, React/TypeScript/Vite, OpenVINO
- Robot: two simulated SO-101 arms using the upstream `vendor/SO-ARM100` submodule

The authoritative challenge text is `Online_Physical_AI_Challenge_Online (1).pdf`.
The revised design is `MISE_project_description.docx`; its searchable text is
`docs/project-brief.txt`.

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

- 66 Python tests pass, including actual MuJoCo contact and evaluator regressions;
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
expert. The retained pilot passed 3/10 test seeds; a newer checkpoint failed its
first closed-loop validation seed despite passing numerical export parity.

Physical SO-101 hardware is not required by the online challenge. Final MuJoCo and
OpenVINO execution on Intel Core Ultra Series 2 or 3 is required for the 20-point
Intel category and has not yet been performed.

## Remaining work

1. On an Intel Core Ultra Series 2/3 machine, install `requirements-learning.txt`,
   generate or copy the exported ACT model, and run `bench/intel_bench.py` with the
   correct `--intel-core-ultra-series` value.
2. Run reference and proposed optimized configurations on identical frozen seeds.
   Only mark quality preservation verified after reviewing paired task outcomes.
3. Verify the Docker image on a clean machine with submodules initialized.
4. Record the narrated pitch using `docs/demo-guide.md`; its script and shot list are
   prepared, but the Intel segment must wait for measured target results.
5. A public interactive runtime is optional. GitHub Pages cannot host FastAPI or
   MuJoCo; any public site must be a labeled read-only replay unless backed by a
   suitable server.

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

1. `README.md`
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

> Continue the Soul Stacks MISE project from the repository
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
