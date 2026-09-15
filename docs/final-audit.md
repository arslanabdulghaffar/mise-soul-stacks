# Final validation work — SeoulStack

The full-task expert and console are implemented. **The project is not yet proven
ready for maximum judging points.** Learned-policy reliability, broader shape
robustness, and the final Intel demonstration remain open. This audit supersedes
earlier statements suggesting that only deployment remained.

## Camera and runtime changes

Every new recording captures overhead, both wrists, and both arm-side views at the
same simulation instant and frame rate. The large view and four thumbnails remain
selectable, with synchronized replay controls. Historical videos keep their actual
original rates and cannot gain missing captured frames.

| Quality | Overhead | Each close-up | All camera rates |
| --- | --- | --- | --- |
| Fast live | 256×256 | 160×160 | 2 FPS |
| Balanced | 384×384 | 256×256 | 3 FPS |
| Detail | 720×720 | 384×384 | 2 FPS |

These rates use simulation time. Browser delivery and wall-clock speed depend on
the host. Five cameras require additional work. Display rendering uses one context
with smaller inset viewports. Mesa rasterization and each video encoder use bounded
threads; videos use fast encoding. Controller RGB calibration and physics remain
unchanged by these display settings.

The camera regression checks every profile, image dimensions, actual rendered
content, bitwise-identical controller observations, joints and physics state.
Browser checks cover five visible streams, thumbnail promotion, synchronized
replay play/pause/seek/rate, and mobile layout. The current regression suite has
**96 passing Python tests**, including policy-bundle integrity, ACT console contracts,
full-component utensil measurement, actual MuJoCo contact, and evaluator checks. The
production frontend builds, and all 70 original recorded-evidence checksums still
verify.

The complete synchronized recording `2f710b64aab8` passes all seven task steps,
with one recovery, zero collisions, and nine verified artifact hashes. Each camera
contains 277 frames at 2 FPS. The 138.3-second simulation took **204.00 wall seconds**
on this host. This is slower than the historical 110.77-second run that captured
close-ups less frequently; it is not a controlled performance comparison. See
[the recording verification](evidence/synchronized-camera-validation.json).

## Reliability findings

The original ten-seed full-task demonstrations remain retained and checksummed.
They establish performance inside their declared nominal envelope.

The first additional stress matrix passed **18/50**: lighting 5/10, background
6/10, changed primitives 0/10, physical properties 7/10, combined changes 0/10.
An intermediate handle/plate threshold change improved that matrix to **24/50**,
but regressed the nominal suite to 9/10. It was therefore not accepted as the final
perception fix. An earlier tighter spoon mask also regressed nominal performance
and was reverted.

The current plate correction preserves the original threshold for normal plate
regions and tightens it only when a highlight produces a region larger than the
calibrated plate envelope. This avoids shifting nominal grasp targets. The yellow
handle mask rejects brown tabletop pixels. The accepted controller passes **10/10 nominal seeds**, with zero forbidden
collisions. Results are recorded separately from the intermediate experiments.

A fresh 50-case matrix (seeds 7001–7010, declared before execution) after the fork
recovery and RGB fixes passed **29/50**: lighting 9/10, background 10/10, physical
properties 10/10, shape 0/10, and combined 0/10. The earlier 5/10 pilot is retained;
the newer complete matrix supersedes it for this source version. Shape and combined
failures remain visible because the changed spoon bowl can lose the receiver's grasp
during handoff. This is a real robustness boundary, not a relaxed evaluator.

Changing the spoon bowl geometry can lose the receiver's grasp during handoff.
Correct final object locations alone do not satisfy the physical-handoff evaluator.
Failures remain failures; evaluator thresholds, contact requirements, and collision
rules were not weakened. Evaluation reports now identify missing predicates even
when the controller finished its programmed sequence.

## Learned policy and Intel scope

Earlier ACT baselines passed 2/10 frozen test seeds, a refined checkpoint passed 1/5
development seeds, and a temporal-ensemble pilot passed 0/5. Those retained failures
are not overwritten. A new action-context candidate was initialized from the refined
policy, trained using actual preceding commanded action plus elapsed time, exported to
OpenVINO, and evaluated on ten fresh held-out seeds 50000–50009. It passed **10/10**
with independent grasp, lift, upright release, and stable-placement checks. Its model,
scene, runtime-source hashes, and evaluation hashes are registered before the console
allows selection. A fresh recorded console run (seed 50010) also passed. The policy
is limited to nominal arm-B mug pick/place and is not evidence of learned full-table
control. Intel pair validation rejects comparisons that change validated settings.

[The experiment summary](evidence/final-audit-experiments.json) retains all declared
outcomes and hashes of local source manifests/results. Detailed traces, original
and refined checkpoints, and frozen XML variants remain under `artifacts/` locally.
`scripts/package_contact_policy.py` creates a verified transfer ZIP from the
registered policy and its ten-seed evaluation; large weights are not committed to
GitHub.

The Intel target demonstration is deferred at the user's request. Use
[intel-validation.md](intel-validation.md) after confirming the actual target SKU.
Device ownership, export parity, and inference latency alone do not establish
preserved closed-loop task quality.

## Next work

1. Improve handoff retention under altered spoon shapes and validate on a new,
   predeclared stress matrix. Do not alter the evaluator to hide failures.
2. Run paired OpenVINO measurements and closed-loop evaluation on the required Intel
   hardware, then capture the final narrated demonstration.

GitHub contains source and recorded evidence. GitHub Pages cannot execute FastAPI
or MuJoCo. The interactive console runs on the machine hosting the API at port 8000;
sharing an interactive link requires a running server or forwarded development port.
