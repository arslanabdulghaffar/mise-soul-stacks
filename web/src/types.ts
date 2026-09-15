export type RunStatus = 'queued' | 'running' | 'paused' | 'completed' | 'stopped' | 'failed'
export type Controller = 'contact_expert' | 'scripted_drawer' | 'learned_act'
export type RecoveryMode = 'none' | 'blind_retry' | 'adaptive'
export type Camera = 'top' | 'wrist_a' | 'wrist_b' | 'side_a' | 'side_b'
export type View = 'Overview' | 'Live run' | 'Evidence' | 'Recovery' | 'Benchmarks' | 'Method'
export type Hardware = { hostname?: string; cpu?: string; platform?: string; device?: string; [key: string]: unknown }
export type PlanStep = { id: number; skill: string; arm: string; object: string; target?: string; needs?: number[] }
export type Run = {
  id: string; status: RunStatus; command: string; seed: number; preset: string; controller: string;
  recovery_mode?: RecoveryMode;
  view_quality?: string;
  camera_configuration?: { camera_dimensions?: Partial<Record<Camera, {width: number; height: number}>>; width: number; height: number; camera_capture_hz: Partial<Record<Camera, number>> };
  created_at: string; hardware: Hardware; scope?: string; plan?: { steps: PlanStep[] }; config_hash: string;
  summary?: { full_task_success: boolean | null; scope?: string; reason?: string; wall_seconds?: number;
    assisted?: boolean; autonomous_contact_skill_success?: boolean; completed_steps?: number[]; fixture_success?: boolean; contact_skill_success?: boolean; task_success?: boolean; contact_evidence?: Record<string, unknown>; [key: string]: unknown } | null;
  artifacts: { trace?: string; summary?: string; manifest?: string; video?: string; [key: string]: string | undefined };
  last_sequence?: number; last_observation_at?: string; phase?: string; active_step?: PlanStep | null;
}
export type RunEvent = {
  run_id: string; sequence: number; timestamp: string; monotonic_time?: number; simulation_time?: number;
  type: string; active_skill?: string | null; arm_state?: Record<string, unknown> | string;
  monitor?: Record<string, unknown> | string; config_hash?: string; payload: Record<string, unknown>;
}
export type Health = { status: string; hardware: Hardware; capabilities?: { live_control?: boolean; controller?: string; controllers?: string[]; supported_commands?: string[] | Record<string, string[]>; [key: string]: unknown } }
export type Evaluation = { required_seeds: number[]; runs: Run[]; summary: { successes: number; total: number; wilson95: number[] | null } }
export type Benchmark = { records: Record<string, unknown>[]; hardware?: Hardware; csv?: string; [key: string]: unknown }
export type RecoveryComparison = { available: boolean; paired_success_gain_over_blind_retry?: number; note?: string;
  variants: Record<string, { label: string; successes: number; total: number; recovery_attempts: number; wilson95?: number[] }> }
