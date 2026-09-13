# Learning from recovery outcomes

The recovery-memory component is implemented in `mise.recovery_memory`. It
persists visual outcomes in SQLite and updates the success and duration estimates
used by the existing constrained recovery supervisor. It does **not** update ACT
weights. The full-task runtime now uses it to rank two registered physical spoon
regrasp profiles and records each camera-verified outcome.

For example, two failed front-grasp corrections can make a previously validated
side-grasp correction preferable on the next matching task. Within an episode, a
failed correction is excluded for the same observable state. If no tested
alternative meets the constraints, the supervisor stops.

Each record contains an episode ID, unique attempt ID, candidate and arm, elapsed
time, visual outcome, and a reference to the saved observation trace. The context
includes the skill, object, phase, observed failure category, an observable state
bucket, scene version and policy version. Exact matching deliberately avoids
transferring experience between incompatible scenes or checkpoints. Duplicate
delivery cannot count an attempt twice. Occluded or otherwise unknown outcomes
remain in the log but do not affect estimates.

For a candidate with validation estimate `p`, the updated estimate is
`(2*p + observed_successes) / (2 + observed_attempts)`. Duration uses the same
two-observation prior. This is an empirical ranking estimate, not a calibrated
confidence or a diagnosis of the failure's physical cause. No outcome can
register a new action, make an infeasible action feasible, change an explicit arm
assignment, undo a completed goal, or increase the attempt/time budgets.

`FullTaskController` calls `decide_with_memory`, retains the selected candidate,
consumes the bounded attempt, executes from the current state, and calls `record`
after RGB verification. It can choose a direct or cautious arm-B table regrasp.
The mug and standalone drawer experts do not invoke recovery. Simulator resets
remain new episodes.

The recovery-memory tests check persistent adaptation,
same-episode exclusion, unknown outcomes, version isolation, duplicate events,
summary evidence, and preservation of constraints. The full-task physics test also
checks that a successful physical correction is persisted.

The frozen matched comparison retains all outcomes for seeds 1001–1010. No recovery
and blind retry each pass 9/10. Monitored recovery passes 10/10 and uses one physical
correction. Memory writes are disabled during that comparison so test outcomes do
not leak into later decisions. Normal live runs update the persistent store. This
small nominal suite demonstrates benefit for one observed failure and does not
establish broad recovery generalization. Motor-policy retraining from corrective
demonstrations remains a separate experiment.
