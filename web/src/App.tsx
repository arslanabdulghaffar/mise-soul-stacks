import { useCallback, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, BookOpen, Box, Braces, Check,
  CheckCheck, CheckCircle2, ChevronDown, ChevronRight, Circle, CircleDot, Clock3,
  Cpu, Download, Fingerprint, Gauge, GitBranch, Grip, Layers3,
  LayoutDashboard, LoaderCircle, Maximize2, Monitor, MousePointer2, MoveUpRight,
  Pause, Play, Radio, RotateCcw, ScanLine, Settings2, ShieldCheck, Square,
  Terminal, Video, X,
} from 'lucide-react'
import { eventSocket, request } from './api'
import type { Benchmark, Camera, Controller, Evaluation, Health, PlanStep, RecoveryComparison, RecoveryMode, Run, RunEvent, View } from './types'

const NAV: { name: View; icon: typeof Activity; detail: string }[] = [
  { name: 'Overview', icon: LayoutDashboard, detail: 'The project at a glance' },
  { name: 'Live run', icon: Radio, detail: 'Observe. Execute. Verify.' },
  { name: 'Evidence', icon: Layers3, detail: 'Every seed. Every outcome.' },
  { name: 'Recovery', icon: GitBranch, detail: 'Understand each intervention' },
  { name: 'Benchmarks', icon: Gauge, detail: 'Measured on the actual machine' },
  { name: 'Method', icon: BookOpen, detail: 'Inside the execution loop' },
]
const CAMERAS: { id: Camera; label: string; short: string }[] = [
  { id: 'top', label: 'Overhead', short: 'CAM 01' },
  { id: 'wrist_a', label: 'Arm A · wrist', short: 'CAM 02' },
  { id: 'wrist_b', label: 'Arm B · wrist', short: 'CAM 03' },
]
const ACTIVE_STATUSES = new Set(['queued', 'running', 'paused'])
const DEFAULT_COMMAND = 'Set the table.'
const FIXTURE_COMMAND = 'Open the drawer with arm A.'
const controllerName = (controller?: string) => controller === 'contact_expert' ? 'Contact expert' : controller === 'scripted_drawer' ? 'Drawer fixture (legacy)' : readable(controller)
const PHASE_LABELS: Record<string, string> = { locate_mug: 'Locate mug in overhead image', approach_mug: 'Approach mug', descend_to_grasp: 'Align gripper around mug', close_gripper: 'Close gripper', lift_mug: 'Lift mug', carry_to_goal: 'Carry to upper-right', lower_mug: 'Lower mug', release_mug: 'Release mug', retract_gripper: 'Withdraw gripper', clear_camera: 'Move arm clear of view', settle_and_verify: 'Check stable placement', completed: 'Control sequence finished', failed: 'Execution failed', approach: 'Approach object', pregrasp: 'Align the gripper', grasp: 'Acquire contact', close: 'Close gripper', lift: 'Lift object', transport: 'Move to target', place: 'Place object', release: 'Release object', retreat: 'Withdraw arm', verify: 'Verify outcome', settle: 'Check stable placement', done: 'Skill finished', stopped: 'Stopped by operator' }
const phaseLabel = (phase?: string | null) => phase ? PHASE_LABELS[phase] || readable(phase) : 'Standing by'
const readable = (value: string | undefined | null) => value ? value.replaceAll('_', ' ') : 'Unavailable'
const shortId = (id?: string) => id ? id.slice(0, 12) : 'No run selected'
const time = (value?: string) => value ? new Date(value).toLocaleTimeString([], { hour12: false }) : '—'
const duration = (seconds?: number | null) => typeof seconds === 'number' ? `${seconds.toFixed(2)} s` : '—'
const payloadText = (event: RunEvent) => {
  const p = event.payload || {}
  const candidate = p.message ?? p.reason ?? p.detail ?? p.description ?? (typeof p.phase === 'string' ? phaseLabel(p.phase) : undefined) ?? p.state
  return typeof candidate === 'string' ? candidate : event.active_skill ? readable(event.active_skill) : 'Recorded execution event'
}
const monitorText = (event?: RunEvent) => {
  const monitor = event?.monitor
  if (typeof monitor === 'string') return monitor
  if (monitor && typeof monitor.state === 'string') return monitor.state
  if (monitor && typeof monitor.status === 'string') return monitor.status
  return null
}

function Pill({ children, tone = '', dot = false }: { children: ReactNode; tone?: string; dot?: boolean }) {
  return <span className={`pill ${tone}`}>{dot && <span className="status-dot" />}{children}</span>
}
function Empty({ icon: Icon = Layers3, title, children }: { icon?: typeof Activity; title: string; children: ReactNode }) {
  return <div className="empty"><div className="empty-icon"><Icon size={25} strokeWidth={1.5} /></div><h3>{title}</h3><p>{children}</p></div>
}
function ArtifactLinks({ run }: { run: Run }) {
  const links = [['trace', 'Trace JSONL'], ['summary', 'Summary'], ['manifest', 'Manifest']] as const
  return <div className="artifact-links">{links.map(([key, label]) => run.artifacts?.[key] && <a key={key} href={run.artifacts[key]} download><ArrowDownToLine size={14} />{label}</a>)}</div>
}

export default function App() {
  const [view, setView] = useState<View>('Live run')
  const [health, setHealth] = useState<Health | null>(null)
  const [connected, setConnected] = useState(false)
  const [runs, setRuns] = useState<Run[]>([])
  const [run, setRun] = useState<Run | null>(null)
  const [events, setEvents] = useState<RunEvent[]>([])
  const [evaluations, setEvaluations] = useState<Evaluation | null>(null)
  const [benchmarks, setBenchmarks] = useState<Benchmark | null>(null)
  const [recoveryComparison, setRecoveryComparison] = useState<RecoveryComparison | null>(null)
  const [command, setCommand] = useState(DEFAULT_COMMAND)
  const [controller, setController] = useState<Controller>('contact_expert')
  const [planPreview, setPlanPreview] = useState<PlanStep[]>([])
  const [previewError, setPreviewError] = useState('')
  const [previewExecutable, setPreviewExecutable] = useState<boolean | null>(null)
  const initializedController = useRef(false)
  const [seed, setSeed] = useState('1001')
  const [preset, setPreset] = useState('nominal')
  const [recoveryMode, setRecoveryMode] = useState<RecoveryMode>('adaptive')
  const [camera, setCamera] = useState<Camera>('top')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [streamConnected, setStreamConnected] = useState(false)
  const [replay, setReplay] = useState(false)
  const [replayTime, setReplayTime] = useState<number | null>(null)
  const [now, setNow] = useState(Date.now())
  const [expanded, setExpanded] = useState(false)
  const [benchmarkComponent, setBenchmarkComponent] = useState('all')
  const [traceLoading, setTraceLoading] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)

  const refresh = useCallback(async () => {
    const result = await Promise.allSettled([
      request<Health>('/api/health'), request<{ runs: Run[] }>('/api/runs'),
      request<Evaluation>('/api/evaluations'), request<Benchmark>('/api/benchmarks'),
      request<RecoveryComparison>('/api/recovery-comparison'),
    ])
    if (result[0].status === 'fulfilled') { setHealth(result[0].value); setConnected(true) } else setConnected(false)
    if (result[1].status === 'fulfilled') setRuns(result[1].value.runs || [])
    if (result[2].status === 'fulfilled') setEvaluations(result[2].value)
    if (result[3].status === 'fulfilled') setBenchmarks(result[3].value)
    if (result[4].status === 'fulfilled') setRecoveryComparison(result[4].value)
  }, [])

  useEffect(() => { void refresh(); const interval = window.setInterval(() => void refresh(), 5000); return () => clearInterval(interval) }, [refresh])
  useEffect(() => { const interval = window.setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(interval) }, [])
  const active = !!run && ACTIVE_STATUSES.has(run.status)
  const operatorEnabled = connected && health?.capabilities?.live_control !== false
  const contactAvailable = health?.capabilities?.controller === 'contact_expert' || health?.capabilities?.controllers?.includes('contact_expert') === true
  const controllerAvailable = controller === 'scripted_drawer' || contactAvailable
  const commandExecutable = controller === 'scripted_drawer' || previewExecutable === true
  const supportedCommands = health?.capabilities?.supported_commands
  const commandExamples = controller === 'scripted_drawer' ? [FIXTURE_COMMAND] : Array.isArray(supportedCommands) ? supportedCommands : supportedCommands?.contact_expert || []

  useEffect(() => {
    if (!health || initializedController.current) return
    initializedController.current = true
    if (!contactAvailable) { setController('scripted_drawer'); setCommand(FIXTURE_COMMAND) }
  }, [health, contactAvailable])

  useEffect(() => {
    if (!connected || active) return
    let cancelled = false
    setPreviewExecutable(null); setPreviewError('')
    const timeout = window.setTimeout(() => {
      void request<{ plan: { steps: PlanStep[] }; executable?: boolean; reason?: string; note?: string }>('/api/plans', { method: 'POST', body: JSON.stringify({ command }) })
        .then(data => { if (!cancelled) { setPlanPreview(data.plan.steps); setPreviewExecutable(data.executable === true); setPreviewError(data.executable === false ? data.reason || data.note || 'This command is not available for the contact expert.' : '') } })
        .catch(error => { if (!cancelled) { setPlanPreview([]); setPreviewExecutable(false); setPreviewError(error instanceof Error ? error.message : 'Command preview unavailable') } })
    }, 350)
    return () => { cancelled = true; clearTimeout(timeout) }
  }, [command, controller, connected, active])
  const currentId = run?.id

  useEffect(() => {
    if (!currentId || replay) { setStreamConnected(false); return }
    let cancelled = false
    let socket: WebSocket | undefined
    let retry: number | undefined
    const connect = () => {
      if (cancelled) return
      socket = eventSocket(currentId)
      socket.onopen = () => setStreamConnected(true)
      socket.onmessage = (message: MessageEvent<string>) => {
        try {
          const data = JSON.parse(message.data)
          if (data.type === 'snapshot') {
            setRun(data.run)
            void request<RunEvent[]>(`/api/runs/${currentId}/trace`).then(trace => setEvents(previous => [...new Map([...trace.filter(event => event.type !== 'action'), ...previous].map(event => [event.sequence, event])).values()].sort((a, b) => a.sequence - b.sequence))).catch(() => {})
          } else if (typeof data.sequence === 'number' && data.type !== 'action') {
            setEvents(previous => previous.some(e => e.sequence === data.sequence) ? previous : [...previous, data].sort((a, b) => a.sequence - b.sequence).slice(-3000))
            if (data.type === 'observation') setRun(previous => previous ? { ...previous, last_sequence: data.sequence, last_observation_at: data.payload.frame_timestamp || previous.last_observation_at } : previous)
            if (data.payload?.phase || data.payload?.step) setRun(previous => previous ? { ...previous, phase: data.payload.phase || previous.phase, active_step: data.payload.step || previous.active_step } : previous)
            if (['run_completed', 'run_failed', 'run_stopped', 'completed', 'stopped', 'run.finished', 'run_status', 'run_finished'].includes(data.type)) {
              void request<Run>(`/api/runs/${currentId}`).then(latest => { setRun(latest); if (!ACTIVE_STATUSES.has(latest.status)) setReplay(true) }).catch(() => {})
              void refresh()
            }
          }
        } catch { setError('Received an unreadable event. Reconnect to refresh the run.') }
      }
      socket.onerror = () => socket?.close()
      socket.onclose = () => { setStreamConnected(false); if (!cancelled) retry = window.setTimeout(connect, 2500) }
    }
    connect()
    return () => { cancelled = true; clearTimeout(retry); socket?.close() }
  }, [currentId, replay, refresh])

  useEffect(() => {
    if (!currentId || !active || replay) return
    const interval = window.setInterval(() => { void request<Run>(`/api/runs/${currentId}`).then(setRun).catch(() => {}) }, 2000)
    return () => clearInterval(interval)
  }, [currentId, active, replay])

  useEffect(() => {
    if (!expanded) return
    const listener = (event: KeyboardEvent) => { if (event.key === 'Escape') setExpanded(false) }
    window.addEventListener('keydown', listener)
    return () => window.removeEventListener('keydown', listener)
  }, [expanded])

  const selectRun = async (selected: Run, nextView: View = 'Live run') => {
    setError(''); setEvents([]); setRun(selected); setView(nextView)
    setReplay(!ACTIVE_STATUSES.has(selected.status) || !operatorEnabled); setReplayTime(null); setTraceLoading(true)
    try {
      const [latest, trace] = await Promise.all([
        request<Run>(`/api/runs/${selected.id}`), request<RunEvent[]>(`/api/runs/${selected.id}/trace`),
      ])
      setRun(latest); setEvents(Array.isArray(trace) ? trace.filter(event => event.type !== 'action') : [])
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not load this run.') }
    finally { setTraceLoading(false) }
  }
  const startRun = async () => {
    if (!operatorEnabled || !controllerAvailable || !commandExecutable || busy || active) return
    setError(''); setBusy(true)
    try {
      const next = await request<Run>('/api/runs', { method: 'POST', body: JSON.stringify({ command, seed: Number(seed), preset, controller, recovery_mode: recoveryMode }) })
      setRun(next); setEvents([]); setReplay(false); setReplayTime(null); setCamera('top'); void refresh()
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not start the run.') }
    finally { setBusy(false) }
  }
  const control = async (action: 'pause' | 'resume' | 'stop') => {
    if (!run || !connected || busy) return
    setError(''); setBusy(true)
    try { await request(`/api/runs/${run.id}/control`, { method: 'POST', body: JSON.stringify({ action }) }); setRun(await request<Run>(`/api/runs/${run.id}`)); void refresh() }
    catch (e) { setError(e instanceof Error ? e.message : 'The control request failed.') }
    finally { setBusy(false) }
  }

  useEffect(() => { if (run && !ACTIVE_STATUSES.has(run.status) && run.artifacts?.video) setReplay(true) }, [run])

  const displayEvents = replayTime !== null && replay ? events.filter(e => (e.simulation_time ?? 0) <= replayTime) : events
  const latestEvent = displayEvents.at(-1)
  const latestPhaseEvent = [...displayEvents].reverse().find(e => typeof e.payload?.phase === 'string')
  const currentPhase = replay && replayTime !== null ? latestPhaseEvent?.payload.phase as string | undefined : run?.phase || latestPhaseEvent?.payload.phase as string | undefined
  const currentStepId = latestPhaseEvent?.payload.step_id ?? run?.active_step?.id
  const shownController = run?.controller || controller
  const isContactRun = shownController === 'contact_expert'
  const isFullRun = run?.scope === 'full_task' || run?.summary?.scope === 'full_task'
  const shownSteps = run?.plan?.steps || planPreview
  const latestMonitor = [...displayEvents].reverse().find(e => monitorText(e))
  const observedAt = run?.last_observation_at
  const ageSeconds = observedAt ? Math.max(0, (now - Date.parse(observedAt)) / 1000) : null
  const stale = active && ageSeconds !== null && ageSeconds > 5
  const mode = replay || health?.capabilities?.live_control === false ? 'RECORDED REPLAY' : connected && (!active || streamConnected) ? 'LIVE' : 'DISCONNECTED'
  const hardware = replay && run ? run.hardware : health?.hardware
  const completedRuns = runs.filter(r => !ACTIVE_STATUSES.has(r.status))
  const fullRuns = evaluations?.runs.filter(r => r.summary?.scope === 'full_task' && typeof r.summary?.full_task_success === 'boolean') ?? []
  const metadata = NAV.find(item => item.name === view)!
  const replayEvents = replayTime === null ? events : events.filter(e => (e.simulation_time ?? 0) <= replayTime)
  const visibleEvents = (replay ? replayEvents : events).filter(e => !['observation', 'frame', 'heartbeat', 'action'].includes(e.type))
  const recoveryEvents = events.filter(e => /retry|recovery|replan|cancel|stop|monitor|intervention/i.test(e.type))
  const recoveryVariants = ['none', 'blind_retry', 'adaptive'].map(key => recoveryComparison?.variants?.[key]).filter(Boolean)
  const benchmarkRecords = (benchmarks?.records || []).filter(r => benchmarkComponent === 'all' || r.component === benchmarkComponent)
  const allComponents = [...new Set((benchmarks?.records || []).map(r => String(r.component || 'system')))]
  const contactRuns = runs.filter(item => item.summary?.scope === 'contact_skill')
  const contactSuccesses = contactRuns.filter(item => !item.summary?.assisted && (item.summary?.contact_skill_success === true || item.summary?.task_success === true)).length
  const runOutcome = (item: Run) => {
    if (item.summary?.scope === 'contact_skill') {
      const success = item.summary.contact_skill_success ?? item.summary.task_success
      return success === true ? item.summary.assisted ? 'Skill passed · assisted' : 'Contact skill passed' : success === false ? item.summary.assisted ? 'Skill failed · assisted' : 'Contact skill failed' : readable(item.status)
    }
    if (item.summary?.scope === 'drawer_fixture') return 'Fixture only'
    return item.summary?.full_task_success === true ? 'Full task passed' : item.summary?.full_task_success === false ? 'Full task failed' : readable(item.status)
  }

  const timeline = (items: RunEvent[], recovery = false) => <div className="timeline">
    {items.length ? items.slice(-30).map(event => <div className={`timeline-event ${/retry|replan|recovery/i.test(event.type) ? 'intervention' : ''}`} key={event.sequence}>
      <div className="event-time">{typeof event.simulation_time === 'number' ? `${event.simulation_time.toFixed(1)}s` : time(event.timestamp)}</div>
      <div className="event-line"><span /></div><div className="event-content"><div><strong>{event.type === 'phase_changed' && typeof event.payload.phase === 'string' ? phaseLabel(event.payload.phase) : readable(event.type)}</strong><span className="event-seq">#{String(event.sequence).padStart(3, '0')}</span></div><p>{payloadText(event)}</p></div>
    </div>) : <div className="timeline-idle"><CircleDot size={16} /><span>{traceLoading ? 'Loading recorded events…' : recovery ? 'No recovery interventions have been recorded for this run.' : 'Execution events will appear here when a run starts.'}</span></div>}
  </div>

  const cameraStage = <div className={`camera-card ${expanded ? 'expanded' : ''}`}>
    <div className="camera-toolbar"><div className="camera-title"><Video size={15} /><span>Simulation view</span><span className="subtle">/ {CAMERAS.find(c => c.id === camera)?.short}</span></div><div className="camera-toolbar-right"><span className="render-label">MuJoCo</span><button className="icon-button" aria-label={expanded ? 'Close expanded camera' : 'Expand camera'} onClick={() => setExpanded(!expanded)}>{expanded ? <X size={17} /> : <Maximize2 size={16} />}</button></div></div>
    <div className="camera-viewport">
      {run && replay && run.artifacts?.video ? <video ref={videoRef} key={`${run.id}-${camera}-video`} src={run.artifacts[`video_${camera}`] || run.artifacts.video} controls playsInline onTimeUpdate={() => setReplayTime(videoRef.current?.currentTime ?? null)} onSeeked={() => setReplayTime(videoRef.current?.currentTime ?? null)} aria-label={`Recorded ${camera} camera for run ${run.id}`} /> : run && !replay ? <img key={`${run.id}-${camera}`} src={active && connected ? `/api/runs/${run.id}/camera/${camera}` : `/api/runs/${run.id}/frame/${camera}`} alt={`${CAMERAS.find(c => c.id === camera)?.label} camera from run ${run.id}`} onError={event => { event.currentTarget.style.opacity = '0'; event.currentTarget.parentElement?.classList.add('camera-unavailable') }} onLoad={event => { event.currentTarget.style.opacity = '1'; event.currentTarget.parentElement?.classList.remove('camera-unavailable') }} /> : null}

      <div className={`viewport-empty ${run && (!replay || run.artifacts?.video) ? 'hidden-unless-error' : ''} ${!run && connected ? 'preview-overlay' : ''}`}><div className="focus-frame"><ScanLine size={34} strokeWidth={1.1} /></div><h3>{run && replay ? 'Camera recording unavailable' : run ? 'Waiting for camera frames' : connected ? operatorEnabled ? 'No simulation running' : 'Recorded evidence is available' : 'Waiting for the local runtime'}</h3><p>{run && replay ? 'The recorded trace and run artifacts are available below.' : run ? 'Frames appear as the simulation worker renders them.' : connected ? operatorEnabled ? 'Run task starts a new simulation from your command and seed, with live camera frames.' : 'Select a run from Evidence to inspect its original cameras and trace.' : 'Connect the API to receive real camera frames and execution events.'}</p>{!run && <span className="viewport-label">{connected ? 'NO ACTIVE RUN · AWAITING FRAMES' : 'NO CAMERA SIGNAL'}</span>}</div>
      {run && !(replay && run.artifacts?.video) && <div className="viewport-badges"><Pill tone={replay ? 'violet' : active ? 'cyan' : 'neutral'} dot>{replay ? 'RECORDED FRAME' : active ? stale ? 'STALE FRAME' : 'CAMERA STREAM' : 'FINAL FRAME'}</Pill><span>{CAMERAS.find(c => c.id === camera)?.label}</span></div>}
      <div className="viewport-corners"><i /><i /><i /><i /></div>
      {!replay && <div className="viewport-footer"><span><span className="arm-dot cyan-dot" /> Arm A</span><span><span className="arm-dot violet-dot" /> Arm B</span><span className="viewport-time">{run ? `Observation ${time(observedAt)}` : 'SO-101 × 2 · Table-setting scene'}</span></div>}
    </div>
    <div className="camera-bottom"><div className="camera-tabs" role="tablist" aria-label="Camera angle">{CAMERAS.map(cam => <button key={cam.id} role="tab" aria-selected={camera === cam.id} className={camera === cam.id ? 'selected' : ''} disabled={replay && cam.id !== 'top' && !run?.artifacts?.[`video_${cam.id}`]} onClick={() => setCamera(cam.id)}><span className={`camera-indicator ${cam.id === 'wrist_b' ? 'violet-dot' : ''}`} />{cam.label}</button>)}</div><span className="camera-size">{replay ? 'Original recording' : 'RGB stream'}</span></div>
  </div>

  const planPanel = <aside className="panel plan-panel"><div className="panel-heading"><div><GitBranch size={16} /><h2>Execution plan</h2></div><Pill>{shownSteps.length} steps</Pill></div>
    <div className="plan-summary"><span className="eyebrow">{run ? 'ACCEPTED INSTRUCTION' : 'COMMAND PREVIEW'}</span><p>{run?.command || command}</p></div>
    <div className="plan-steps">{shownSteps.length ? shownSteps.map((step, index) => {
      const done = isContactRun || (run && !ACTIVE_STATUSES.has(run.status))
        ? run?.summary?.completed_steps?.includes(step.id) === true
        : displayEvents.some(e => /step_completed|step_done/.test(e.type) && e.payload.step_id === step.id)
      const current = !done && (active || replayTime !== null) && (currentStepId !== undefined ? currentStepId === step.id : latestEvent?.active_skill === step.skill)
      return <div className={`plan-step ${done ? 'done' : current ? 'current' : ''}`} key={step.id}><div className="step-number">{done ? <Check size={15} /> : current ? <LoaderCircle size={15} className="spin" /> : String(index + 1).padStart(2, '0')}</div><div className="step-body"><strong>{readable(step.skill)}</strong><p>{readable(step.object)}{step.target ? ` → ${readable(step.target)}` : ''}</p><span className={`arm-label ${String(step.arm).toLowerCase().includes('b') ? 'arm-b' : ''}`}>Arm {String(step.arm).replace(/^arm[_ ]?/i, '').toUpperCase()}</span>{current && currentPhase && <span className="step-phase">{phaseLabel(currentPhase)}</span>}</div></div>
    }) : <div className="plan-empty">{previewError || 'The runtime will validate the command and its arm, object, and target before execution.'}</div>}</div>
    <div className="monitor-box" aria-live="polite"><div><ShieldCheck size={17} /><span>{isContactRun ? 'Contact execution' : 'Execution monitor'}</span></div><strong>{currentPhase ? phaseLabel(currentPhase) : monitorText(latestMonitor) || (run?.status === 'completed' ? 'Run finished' : active ? 'Awaiting observation' : 'Standing by')}</strong><p>{run?.summary?.reason || (latestPhaseEvent ? payloadText(latestPhaseEvent) : latestMonitor ? payloadText(latestMonitor) : 'Start a live run to observe the active skill and its measured outcome.')}</p></div>
    <div className="plan-disclosure"><span className="tiny-dot" />{isContactRun ? 'Deterministic camera-guided IK/contact control.' : 'Legacy fixture · directly driven drawer.'}<br />{isFullRun ? 'Full-task outcome is scored by an isolated physics evaluator.' : isContactRun ? 'This skill remains separate from full-task scoring.' : 'No robot grasp or full-task result.'}</div>
  </aside>

  return <div className="app-shell">
    <aside className="sidebar"><a className="brand" href="#" onClick={e => { e.preventDefault(); setView('Overview') }} aria-label="MISE overview"><span className="brand-mark"><svg viewBox="0 0 32 32" fill="none"><path d="M5 24V8L16 19L27 8V24" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /></svg></span><span>mise<span className="brand-period">.</span></span></a><div className="workspace-label">ROBOTICS WORKSPACE</div><nav aria-label="Main navigation">{NAV.map(({ name, icon: Icon }) => <button key={name} className={`nav-item ${view === name ? 'active' : ''}`} onClick={() => { setView(name); setError('') }}><Icon size={18} strokeWidth={1.7} /><span>{name}</span>{name === 'Live run' && <span className={`nav-dot ${active ? 'is-running' : ''}`} />}</button>)}</nav>
      <div className="sidebar-bottom"><div className="build-card"><span className="build-icon"><Box size={19} /></span><div><strong>Research build</strong><span>{!connected ? 'Runtime disconnected' : !operatorEnabled ? 'Recorded evidence only' : contactAvailable ? 'Full physical task available' : 'Drawer fixture available'}</span></div><span className="tiny-dot" /></div><div className="sidebar-footer"><span>SOUL STACKS</span><span>2026 <MoveUpRight size={12} /></span></div></div>
    </aside>
    <div className="main-shell"><header className="topbar"><div className="breadcrumbs"><span>Workspace</span><ChevronRight size={14} /><strong>{view}</strong></div><div className="topbar-right"><span className="local-label"><Monitor size={14} />Local runtime</span><Pill dot tone={mode === 'LIVE' ? 'green' : mode === 'RECORDED REPLAY' ? 'violet' : 'muted'}>{mode}</Pill><span className="avatar" aria-label="Soul Stacks">S</span></div></header>
      <main><div className="page-heading"><div><div className="eyebrow"><span className="heading-line" />MULTI-MODAL INSTRUCTION TO SKILL EXECUTION</div><h1>{view === 'Live run' ? 'The execution console' : view === 'Overview' ? 'A place for every action.' : view === 'Evidence' ? 'Show the whole story.' : view === 'Recovery' ? 'Every correction, accounted for.' : view === 'Benchmarks' ? 'Performance, with proof.' : 'From instruction to outcome.'}</h1><p>{view === 'Live run' ? isContactRun ? 'Run the complete camera-guided table setting task or a validated physical skill.' : 'Observe the legacy drawer fixture and its recorded outcome.' : metadata.detail}</p></div><div className="heading-aside">{view === 'Live run' ? <><span className="round-icon"><Activity size={18} /></span><div><strong>{active ? readable(run?.status) : replay ? 'Recorded execution' : operatorEnabled ? 'Ready for live execution' : connected ? 'Replay mode' : 'Runtime disconnected'}</strong><span>{run ? `Seed ${run.seed} · ${shortId(run.id)}` : controllerName(controller)}</span></div></> : <Pill>RESEARCH BUILD · 01</Pill>}</div></div>
        {error && <div className="error-banner" role="alert"><span>{error}</span><button className="icon-button" onClick={() => setError('')} aria-label="Dismiss error"><X size={16} /></button></div>}
        {stale && !replay && <div className="warning-banner"><Clock3 size={16} />Observation is stale. Last received {time(observedAt)} ({Math.floor(ageSeconds!)} s ago). Controls require a runtime connection.</div>}
        {view === 'Live run' && <>
          <form className="command-panel" onSubmit={event => { event.preventDefault(); void startRun() }}><div className="command-main"><Terminal size={20} /><div><label htmlFor="command">WHAT SHOULD THE ROBOT DO?</label><input id="command" value={command} maxLength={500} onChange={event => setCommand(event.target.value)} disabled={active || !operatorEnabled} placeholder={DEFAULT_COMMAND} required /></div></div><div className="command-controls"><label className="seed-control" htmlFor="seed"><span>SEED</span><input id="seed" type="number" min="0" max="2147483647" step="1" value={seed} onChange={event => setSeed(event.target.value)} disabled={active || !operatorEnabled} required /></label><button className="primary-button run-button" type="submit" disabled={busy || active || !operatorEnabled || !controllerAvailable || !commandExecutable}>{busy ? <LoaderCircle size={16} className="spin" /> : <Play size={15} fill="currentColor" />}Run task<ArrowRight size={16} /></button></div></form>
          {controller === 'contact_expert' && previewExecutable === false && previewError && <div className="command-support-note" role="status"><CircleDot size={14} /><span>{previewError}</span></div>}
          <div className="command-examples"><span>TRY A SUPPORTED COMMAND</span>{commandExamples.map(example => <button key={example} disabled={active || !operatorEnabled} className={command === example ? 'selected' : ''} onClick={() => { setCommand(example); setError('') }}>{example}<ArrowUpRight size={11} /></button>)}{!commandExamples.length && <span>Waiting for the runtime command registry</span>}</div>
          <div className="run-options"><div><span className="muted">Controller</span><label className="select-wrap controller-select"><Cpu size={13} /><select value={controller} disabled={active || !operatorEnabled} aria-label="Execution controller" onChange={event => { const value = event.target.value as Controller; setController(value); setCommand(value === 'scripted_drawer' ? FIXTURE_COMMAND : DEFAULT_COMMAND); setError('') }}><option value="contact_expert" disabled={!contactAvailable}>Contact expert{!contactAvailable ? ' · unavailable' : ''}</option><option value="scripted_drawer">Drawer fixture (legacy)</option></select><ChevronDown size={13} /></label><span className="options-separator" /><span className="muted">Scenario</span><label className="select-wrap"><Settings2 size={13} /><select value={preset} onChange={e => setPreset(e.target.value)} disabled={active || !operatorEnabled} aria-label="Reproducible perturbation preset"><option value="nominal">Nominal scene</option><option value="low_friction">Reduced friction · stress preset</option><option value="displaced_objects">Displaced objects · stress preset</option></select><ChevronDown size={13} /></label><span className="options-separator" /><span className="muted">Recovery</span><label className="select-wrap"><GitBranch size={13} /><select value={recoveryMode} onChange={e => setRecoveryMode(e.target.value as RecoveryMode)} disabled={active || !operatorEnabled || controller !== 'contact_expert'} aria-label="Recovery policy"><option value="adaptive">Monitored · cost selected</option><option value="blind_retry">Bounded blind retry</option><option value="none">No recovery</option></select><ChevronDown size={13} /></label></div><span className="run-id"><Fingerprint size={13} />{shortId(run?.id)}</span></div>
          <div className="console-grid">{cameraStage}{planPanel}</div>
          <div className="execution-bar"><div><Pill dot tone={run?.status === 'completed' ? 'green' : active ? 'cyan' : 'neutral'}>{run ? readable(run.status).toUpperCase() : 'NO ACTIVE RUN'}</Pill><span>{run?.summary?.reason || (replay ? 'Viewing original recorded evidence' : active ? `${controllerName(run?.controller)} · ${phaseLabel(currentPhase)}` : 'Run a command to start a fresh simulation with live cameras.')}</span></div><div className="execution-controls">{active && !replay ? <><button className="secondary-button compact" disabled={!operatorEnabled || busy} onClick={() => void control(run?.status === 'paused' ? 'resume' : 'pause')}>{run?.status === 'paused' ? <Play size={14} /> : <Pause size={14} />}{run?.status === 'paused' ? 'Resume' : 'Pause'}</button><button className="stop-button" disabled={!operatorEnabled || busy} onClick={() => void control('stop')}><Square size={12} fill="currentColor" />Stop run</button></> : run && !replay ? <button className="secondary-button compact" onClick={() => void selectRun(run)}><RotateCcw size={14} />Open replay</button> : null}</div></div>
          {run?.summary?.scope === 'contact_skill' && <ContactEvidence run={run} />}
          {run?.summary?.scope === 'full_task' && <FullTaskEvidence run={run} />}
          <div className="lower-grid"><section className="panel timeline-panel"><div className="panel-heading"><div><Activity size={16} /><h2>Event timeline</h2><Pill>{visibleEvents.length}</Pill></div><span className="panel-caption">{replay ? 'Recorded trace' : 'Sequenced telemetry'}</span></div>{timeline(visibleEvents)}{run && <div className="timeline-artifacts"><ArtifactLinks run={run} /></div>}</section><section className="panel runtime-panel"><div className="panel-heading"><div><Cpu size={16} /><h2>Runtime identity</h2></div><span className={`connection-indicator ${connected ? 'online' : ''}`} /></div><dl className="runtime-details"><div><dt>Host</dt><dd title={hardware?.hostname}>{hardware?.hostname || 'Not connected'}</dd></div><div><dt>Compute</dt><dd title={hardware?.cpu}>{hardware?.cpu || hardware?.device || 'Unavailable'}</dd></div><div><dt>Controller</dt><dd>{controllerName(shownController)}</dd></div><div><dt>Observed at</dt><dd className="mono">{time(observedAt)}</dd></div><div><dt>Wall time</dt><dd className="mono">{duration(run?.summary?.wall_seconds)}</dd></div></dl><div className="runtime-footnote"><ShieldCheck size={14} /><span>Hardware and outcomes come from the runtime.</span></div></section></div>
        </>}
        {view === 'Overview' && <><section className="overview-hero"><div><Pill tone="cyan">THE MISE PROJECT</Pill><h2>Understand the task.<br />Coordinate the arms.<br /><span>Verify the outcome.</span></h2><p>MISE executes bimanual table setting with two simulated SO-101 arms, a dependency graph, camera-grounded contact skills, and independent outcome verification.</p><div className="hero-actions"><button className="primary-button" onClick={() => setView('Live run')}><Radio size={16} />Open execution console<ArrowRight size={16} /></button><button className="text-button" onClick={() => setView('Method')}>Explore the method<ArrowUpRight size={15} /></button></div></div><div className="overview-process"><div className="process-number">01 — 04</div>{[{icon: Braces, title: 'Plan', detail: 'Preserve objects, arms, and order'}, {icon: MousePointer2, title: 'Execute', detail: 'Select a registered motor skill'}, {icon: ScanLine, title: 'Verify', detail: 'Check an observable outcome'}, {icon: GitBranch, title: 'Recover', detail: 'Retry, replan, or stop within limits'}].map(({icon: Icon, title, detail}, i) => <div className="process-row" key={title}><span className={i === 3 ? 'violet-text' : ''}><Icon size={20} /></span><div><strong>{title}</strong><p>{detail}</p></div><span className="process-index">0{i + 1}</span></div>)}<span className="process-disclosure">VERIFIED EXECUTION · ADAPTIVE RECOVERY</span></div></section><div className="section-title"><div><span className="eyebrow">THE CURRENT BUILD</span><h2>Evidence starts with a real run.</h2></div><button className="text-button" onClick={() => setView('Evidence')}>View archive<ArrowUpRight size={15} /></button></div><div className="overview-cards"><section className="panel feature-card"><span className="feature-icon"><Video size={23} /></span><h3>Recorded demonstration</h3><p>{completedRuns.length ? `${completedRuns.length} recorded ${completedRuns.length === 1 ? 'run is' : 'runs are'} available. Inspect the original trace and recording.` : 'Start Set the table to create a complete physical run with three camera recordings.'}</p><button className="text-button" disabled={!completedRuns.length} onClick={() => completedRuns[0] && void selectRun(completedRuns[0])}>Open latest replay<ArrowRight size={15} /></button></section><section className="panel feature-card"><span className="feature-icon violet-text"><CheckCheck size={23} /></span><h3>All ten seeds. All outcomes.</h3><p>Predeclared seeds keep the evaluation honest. Contact-skill and fixture outcomes remain separate from full table-setting success.</p><button className="text-button" onClick={() => setView('Evidence')}>Inspect evidence<ArrowRight size={15} /></button></section><section className="panel feature-card"><span className="feature-icon"><Cpu size={23} /></span><h3>Local execution, measured.</h3><p>Simulation, the API, and this interface run locally. Intel optimization claims require hardware records and measured results.</p><button className="text-button" onClick={() => setView('Benchmarks')}>See measurements<ArrowRight size={15} /></button></section></div><div className="scope-banner"><ShieldCheck size={20} /><div><strong>{contactAvailable ? 'Current scope: verified full-task execution' : 'Current scope: a directly driven drawer fixture'}</strong><p>{contactAvailable ? 'The seven-step expert opens the drawer, sets four objects, and transfers the spoon from arm A to arm B through physical contact. Camera segmentation guides control and an isolated evaluator scores the outcome.' : 'The legacy fixture validates the execution and evidence pipeline. Its drawer actuator is driven directly.'}</p></div><button className="text-button" onClick={() => setView('Method')}>Build details<ArrowUpRight size={16} /></button></div></>}
        {view === 'Evidence' && <><div className="metric-grid"><Metric label="FULL-TASK SUCCESS" value={evaluations?.summary?.total ? `${evaluations.summary.successes} / ${evaluations.summary.total}` : '—'} note="Recorded full-task runs only" icon={CheckCircle2} /><Metric label="95% WILSON INTERVAL" value={evaluations?.summary?.wilson95 ? evaluations.summary.wilson95.map(n => `${(n * 100).toFixed(1)}%`).join(' – ') : '—'} note="Unavailable until evaluation" icon={ShieldCheck} /><Metric label="RECORDED RUNS" value={String(runs.length)} note={`${contactRuns.length} contact-skill runs · ${runs.filter(r => r.summary?.scope === 'drawer_fixture').length} legacy fixtures`} icon={Layers3} /></div><div className="scope-banner"><MousePointer2 size={21} /><div><strong>Autonomous contact-skill results: {contactRuns.length ? `${contactSuccesses} / ${contactRuns.length} passed` : 'not measured yet'}</strong><p>These outcomes cover only the requested contact manipulation. Operator-assisted runs are labeled and excluded from autonomous passes. Full-task runs are counted separately above.</p></div><Pill tone="cyan">SEPARATE SCOPE</Pill></div><section className="panel evidence-panel"><div className="panel-heading"><div><Grip size={16} /><h2>The fixed-seed suite</h2></div><Pill>{fullRuns.length ? `${new Set(fullRuns.map(r => r.seed)).size} seeds recorded` : 'AWAITING FULL-TASK RUNS'}</Pill></div><p className="section-description">Every attempt is retained. Contact-skill and fixture results do not count as complete place settings.</p><div className="seed-grid">{(evaluations?.required_seeds || Array.from({ length: 10 }, (_, i) => i)).map((value, index) => {
          const match = fullRuns.filter(r => r.seed === value).at(-1)
          return <button key={value} className={`seed-card ${match?.summary?.full_task_success === true ? 'seed-success' : match ? 'seed-failed' : ''}`} disabled={!match} onClick={() => match && void selectRun(match)}><span className="seed-index">{String(index + 1).padStart(2, '0')}</span>{match ? match.summary?.full_task_success ? <CheckCircle2 size={22} /> : <X size={22} /> : <Circle size={21} strokeWidth={1} />}<strong>Seed {value}</strong><span>{match ? runOutcome(match) : 'Not run'}</span></button>
        })}</div><div className="evidence-note"><CircleDot size={13} />Ten fixed seeds are a demonstration target; they do not establish broad generalization.</div></section><section className="panel archive-panel"><div className="panel-heading"><div><Layers3 size={16} /><h2>Run archive</h2></div><button className="text-button" onClick={() => void refresh()}><RotateCcw size={13} />Refresh</button></div>{runs.length ? <div className="table-scroll"><table><thead><tr><th>Run / instruction</th><th>Seed</th><th>Controller / scenario</th><th>Outcome</th><th>Wall time</th><th>Evidence</th></tr></thead><tbody>{runs.map(item => <tr key={item.id}><td><strong className="mono">{shortId(item.id)}</strong><span className="table-detail">{item.command}</span></td><td className="mono">{item.seed}</td><td>{controllerName(item.controller)}<span className="table-detail">{readable(item.preset)}</span></td><td><Pill tone={item.summary?.scope === 'drawer_fixture' ? 'neutral' : item.summary?.contact_skill_success || item.summary?.task_success || item.summary?.full_task_success ? 'green' : ''}>{runOutcome(item)}</Pill></td><td className="mono">{duration(item.summary?.wall_seconds)}</td><td><button className="text-button" onClick={() => void selectRun(item)}>{ACTIVE_STATUSES.has(item.status) ? 'Open live' : 'Replay'}<ArrowUpRight size={14} /></button></td></tr>)}</tbody></table></div> : <Empty title="Your evidence archive starts here">Start a supported command from the live console. Its configuration, trace, contact evidence, and outcome will be saved here.</Empty>}</section></>}
        {view === 'Recovery' && <><div className="scope-banner"><GitBranch size={21} /><div><strong>Recovery continues from the current state.</strong><p>The visual verifier categorizes a missed spoon placement. The supervisor ranks two validated arm-B regrasp profiles by measured prior success and cost, preserves completed goals, and stores the outcome for later matching contexts.</p></div><Pill tone="violet">ADAPTIVE RECOVERY</Pill></div><div className="recovery-layout"><section className="panel"><div className="panel-heading"><div><RotateCcw size={16} /><h2>Recovery replay</h2></div><select aria-label="Select recovery run" value={run?.id || ''} onChange={event => { const match = runs.find(r => r.id === event.target.value); if (match) void selectRun(match, 'Recovery') }}><option value="">Select a recorded run</option>{completedRuns.map(r => <option key={r.id} value={r.id}>Seed {r.seed} · {shortId(r.id)}</option>)}</select></div>{run ? <><div className="replay-meta"><Pill tone="violet">RECORDED TRACE</Pill><span>{shortId(run.id)} · seed {run.seed} · {readable(run.recovery_mode)}</span></div>{timeline(recoveryEvents, true)}<div className="timeline-artifacts"><ArtifactLinks run={run} /></div></> : <Empty icon={GitBranch} title="A correction should leave a trace.">Select a recorded run to inspect monitor decisions, action cancellations, and interventions.</Empty>}</section><section className="panel recovery-contract"><div className="panel-heading"><div><ShieldCheck size={16} /><h2>Supervisor contract</h2></div></div>{[['CONTINUE', 'Progress is plausible; execute the next validated segment.'], ['DONE', 'Terminal predicates remain satisfied.'], ['RETRY', 'A reachable object and tested corrective action remain.'], ['REPLAN', 'A registered alternative preserves instruction constraints.'], ['STOP', 'Evidence is ambiguous, limits are exceeded, or recovery is infeasible.']].map(([state, detail]) => <div className="contract-row" key={state}><span className={`contract-state ${state === 'RETRY' || state === 'REPLAN' ? 'violet-text' : ''}`}>{state}</span><p>{detail}</p></div>)}<div className="contract-footer">Validated limit: 2 recovery attempts / skill · 180 s / episode</div></section></div><section className="panel"><div className="panel-heading"><div><GitBranch size={16} /><h2>Matched comparison</h2></div><span className="panel-caption">Identical seed, scene, verifier, and budget</span></div><div className="comparison-grid">{recoveryVariants.length ? recoveryVariants.map((variant, index) => <div className="comparison-card" key={variant!.label}><span className="eyebrow">CONTROLLER 0{index + 1}</span><h3>{variant!.label}</h3><strong>{variant!.successes} / {variant!.total}</strong><p>{variant!.recovery_attempts} total recovery attempts</p></div>) : ['No recovery', 'Bounded blind retry', 'Monitored recovery'].map((label, index) => <div className="comparison-card" key={label}><span className="eyebrow">CONTROLLER 0{index + 1}</span><h3>{label}</h3><strong>—</strong><p>{recoveryComparison?.note || 'Run make eval-recovery to create evidence'}</p></div>)}</div></section></>}
        {view === 'Benchmarks' && <><div className="benchmark-heading"><div className="hardware-banner"><Cpu size={23} /><div><span className="eyebrow">REPORTED HARDWARE</span><strong>{hardware?.cpu || benchmarks?.hardware?.cpu || 'Hardware identity unavailable'}</strong><span>{hardware?.hostname || 'Connect the local runtime'} · Intel target verification pending</span></div></div><label className="select-wrap"><select value={benchmarkComponent} onChange={event => setBenchmarkComponent(event.target.value)} aria-label="Benchmark component"><option value="all">All components</option>{allComponents.map(c => <option key={c} value={c}>{readable(c)}</option>)}</select><ChevronDown size={14} /></label></div><div className="metric-grid"><Metric label="POLICY LATENCY · P95" value="—" note="No policy measurements published" icon={Clock3} /><Metric label="CLOSED-LOOP PRESERVATION" value="—" note="Requires paired reference / optimized runs" icon={ShieldCheck} /><Metric label="MEASURED CONFIGURATIONS" value={String(benchmarks?.records?.length || 0)} note="Hardware and model identities required" icon={Cpu} /></div><section className="panel"><div className="panel-heading"><div><Gauge size={16} /><h2>Measured performance</h2></div>{benchmarks?.csv && <a className="text-button" href={benchmarks.csv} download><Download size={14} />Raw CSV</a>}</div>{benchmarkRecords.length ? <div className="table-scroll"><table><thead><tr><th>Component</th><th>Device</th><th>Precision</th><th>Model / configuration</th><th>Median</th><th>p95</th></tr></thead><tbody>{benchmarkRecords.map((record, index) => <tr key={index}><td>{String(record.component ?? 'System')}</td><td>{String(record.device ?? 'Unavailable')}</td><td>{String(record.precision ?? 'Unavailable')}</td><td className="mono">{String(record.model_hash ?? record.config_hash ?? 'Unavailable').slice(0, 16)}</td><td>{typeof record.median_ms === 'number' ? `${record.median_ms.toFixed(2)} ms` : '—'}</td><td>{typeof record.p95_ms === 'number' ? `${record.p95_ms.toFixed(2)} ms` : '—'}</td></tr>)}</tbody></table></div> : <Empty icon={Gauge} title="Measure first. Publish second.">Benchmark records will appear after a measured run. Targets, simulated values, and unverified device support are never shown as results.</Empty>}</section><div className="benchmark-principles">{[{ icon: Cpu, title: 'Exact deployment identity', description: 'Record the host, device, driver, model hash, and precision for each measurement.' }, { icon: Clock3, title: 'The complete loop', description: 'Measure rendering, inference, telemetry, and recording together, with cold and warm timings.' }, { icon: ShieldCheck, title: 'Preserved task quality', description: 'Compare reference and optimized policies on the same frozen seeds before promoting a precision.' }].map(({icon: Icon, title, description}) => <div key={title}><Icon size={20} /><h3>{title}</h3><p>{description}</p></div>)}</div></>}
        {view === 'Method' && <><section className="panel method-architecture"><div className="panel-heading"><div><GitBranch size={16} /><h2>The execution loop</h2></div><Pill>IMPLEMENTED PIPELINE</Pill></div><div className="architecture-flow">{[{ icon: Video, label: 'Observe', detail: 'RGB + robot joints' }, { icon: Braces, label: 'Plan', detail: 'Language + constraints' }, { icon: MousePointer2, label: 'Execute', detail: 'Registered contact skills' }, { icon: ScanLine, label: 'Verify', detail: 'Calibrated visual checks' }, { icon: GitBranch, label: 'Supervise', detail: 'Continue, retry, replan, stop' }].map(({icon: Icon, label, detail}, index) => <div className="architecture-item" key={label}><div><span className="architecture-icon"><Icon size={22} /></span><strong>{label}</strong><p>{detail}</p></div>{index < 4 && <ArrowRight size={17} />}</div>)}</div><div className="observation-boundary"><ShieldCheck size={17} /><p><strong>Observation boundary.</strong> Deployment controllers receive images, robot joints, commands, and action history. True object poses and contacts belong to the isolated evaluator and training data.</p></div></section><div className="method-grid"><section className="panel method-card"><span className="eyebrow">CURRENT CONTROLLERS</span><h2>Deterministic contact execution</h2><p>The complete expert opens the drawer, places the plate, retrieves both utensils, transfers the spoon from arm A to arm B, and places the mug upright. A supported command selects the validated task graph and its IK/contact actions. It runs the simulation on demand and records each phase. It is an engineering expert for developing manipulation, not a trained ACT or vision-language policy.</p><div className="code-snippet"><Terminal size={15} /><code>{DEFAULT_COMMAND}</code></div><ul className="check-list"><li><Check size={15} />Reproducible seed and scenario configuration</li><li><Check size={15} />Live cameras and sequenced execution events</li><li><Check size={15} />Run controls and downloadable evidence</li></ul><p className="method-note">Target estimates use RGB color segmentation and a calibrated overhead projection. IK uses the known robot model and joints; physical contact evidence is evaluator output. The visual verifier is deterministic rather than learned. The contact scene uses finite frictional finger pads and a solid cylindrical mug approximation; hardware calibration is pending. The separate legacy drawer fixture directly drives the drawer actuator and does not demonstrate a robot grasp.</p></section><section className="panel method-card"><span className="eyebrow">SOFTWARE STATUS</span><h2>Adaptive recovery is connected</h2><p>The task graph exposes independent work and the scheduler enforces arm, object, zone, and dependency locks. For the goal-level command, arm A opens the drawer while arm B places the mug in one validated parallel control group. On an observed spoon miss, the supervisor ranks two validated regrasp profiles by success-adjusted cost, enforces the attempt and episode budgets, and persists the visual outcome for future matching contexts.</p><div className="roadmap-list">{[['01', 'Command planning', 'Bounded natural-language grammar produces validated task graphs.'], ['02', 'Full task expert', 'Verified plate, utensils, upright mug, and physical A → B spoon handoff.'], ['03', 'Adaptive recovery', 'Visual failure detection, cost-selected corrections, and persistent outcome memory.'], ['04', 'Intel evaluation', 'Final OpenVINO and hardware validation will be completed on the target.']].map(([number, title, detail]) => <div key={number}><span>{number}</span><div><strong>{title}</strong><p>{detail}</p></div></div>)}</div></section><section className="panel method-card"><span className="eyebrow">RUN IT LOCALLY</span><h2>A reproducible workspace</h2><p>The API serves the compiled interface and records simulation evidence locally. Runtime instructions and dependencies are in the repository README.</p><div className="terminal-block"><span>Setup dependencies</span><code>make setup</code><span>Start the local demonstration</span><code>make console</code><span>Evaluate ten full-task seeds</span><code>make eval</code><span>Compare recovery modes</span><code>make eval-recovery</code></div></section><section className="panel method-card"><span className="eyebrow">INTERPRETING THE EVIDENCE</span><h2>Explicit limits. Useful results.</h2><ul className="plain-list"><li><strong>No invented measurements.</strong> Missing results stay unavailable.</li><li><strong>Skill outcomes are separate.</strong> Only independently evaluated full-task runs establish complete bimanual success.</li><li><strong>Recovery preserves the episode.</strong> A reset or operator action is recorded explicitly.</li><li><strong>Unsupported requests are rejected.</strong> Pouring and unregistered skills are outside the current command grammar.</li><li><strong>Recordings keep their identity.</strong> Replay retains the original seed, configuration, and host.</li></ul></section></div></>}
        <footer className="main-footer"><span><span className="footer-mark">M</span>MISE <span className="footer-divider">/</span> Bimanual table setting with verified recovery</span><button onClick={() => setView('Method')}>Built for observable robotics<ArrowUpRight size={12} /></button></footer>
      </main>
    </div>
  </div>
}

function Metric({ label, value, note, icon: Icon }: { label: string; value: string; note: string; icon: typeof Activity }) {
  return <section className="panel metric-card"><div><span className="eyebrow">{label}</span><Icon size={17} /></div><strong>{value}</strong><p>{note}</p></section>
}

function FullTaskEvidence({ run }: { run: Run }) {
  const summary = run.summary!
  const evaluation = summary.evaluation && typeof summary.evaluation === 'object'
    ? summary.evaluation as Record<string, unknown> : null
  const goals = evaluation?.objects_in_goal && typeof evaluation.objects_in_goal === 'object'
    ? evaluation.objects_in_goal as Record<string, boolean> : {}
  const checks: [string, boolean][] = [
    ['Drawer opened by arm A', evaluation?.drawer_open === true],
    ['Plate stable in center', goals.plate === true],
    ['Spoon transferred A → B', evaluation?.handoff_complete === true],
    ['Spoon stable on right', goals.spoon === true],
    ['Fork stable on left', goals.fork === true],
    ['Mug upright in upper-right', goals.mug === true],
    ['No forbidden collision', summary.collision_count === 0],
  ]
  return <section className="panel contact-evidence"><div className="panel-heading"><div><CheckCheck size={16} /><h2>Full-task evidence</h2></div><Pill tone={summary.full_task_success ? 'green' : ''}>{summary.full_task_success ? summary.assisted ? 'FULL TASK PASSED · ASSISTED' : 'FULL TASK PASSED' : 'FULL TASK FAILED'}</Pill></div><p className="section-description">{summary.reason || 'Inspect the complete recorded outcome.'} The evaluator observes physics but never supplies targets to the controller.</p>
    <dl className="contact-measurements">{checks.map(([label, passed]) => <div key={label}><dt>{label}</dt><dd className={passed ? 'verified-measurement' : ''}>{passed ? <CheckCircle2 size={14} /> : <Circle size={14} />}{passed ? 'Verified' : 'Not verified'}</dd></div>)}<div><dt>Bounded recovery attempts</dt><dd>{typeof summary.recovery_attempts === 'number' ? String(summary.recovery_attempts) : 'Unavailable'}</dd></div></dl>
    <div className="contact-source"><ShieldCheck size={14} /><span>Independent checks cover contact provenance, stable placement, utensil retrieval, handoff, and collision geometry in the complete shared scene.</span></div>
  </section>
}

function ContactEvidence({ run }: { run: Run }) {
  const summary = run.summary!
  const success = summary.contact_skill_success ?? summary.task_success
  const evidence = summary.contact_evidence
  const rawPhysics = evidence?.physics_verification
  const physics = rawPhysics && typeof rawPhysics === 'object' ? rawPhysics as Record<string, unknown> : null
  const drawer = run.plan?.steps.some(step => step.skill === 'open_drawer')
  const checks = drawer ? [['bilateral_grasp_observed', 'Bilateral knob grasp'], ['opening_with_bilateral_contact', 'Pull with gripper contact'], ['stable_released_opening', 'Stable released opening']] : [['bilateral_grasp_observed', 'Bilateral grasp'], ['sustained_lift_observed', 'Sustained lift'], ['stable_released_placement', 'Stable released placement']]
  const metric = (value: unknown, unit: string, factor = 1) => typeof value === 'number' && Number.isFinite(value) ? `${(value * factor).toFixed(2)} ${unit}` : 'Unavailable'
  return <section className="panel contact-evidence"><div className="panel-heading"><div><MousePointer2 size={16} /><h2>Contact-skill evidence</h2></div><Pill tone={success === true ? 'green' : ''}>{success === true ? summary.assisted ? 'SKILL PASSED · ASSISTED' : 'SKILL PASSED' : success === false ? 'SKILL FAILED' : 'OUTCOME UNAVAILABLE'}</Pill></div><p className="section-description">{summary.reason || 'Inspect the recorded contact outcome.'} {summary.assisted ? 'Operator assistance was recorded. ' : ''}Full table-setting success has not been evaluated.</p>
    <dl className="contact-measurements">{checks.map(([key, label]) => <div key={key}><dt>{label}</dt><dd className={physics?.[key] === true ? 'verified-measurement' : ''}>{physics?.[key] === true ? <CheckCircle2 size={14} /> : <Circle size={14} />}{physics?.[key] === true ? 'Observed' : physics?.[key] === false ? 'Not observed' : 'Unavailable'}</dd></div>)}{drawer ? <><div><dt>Drawer travel · physics</dt><dd>{metric(physics?.drawer_travel_m, 'cm', 100)}</dd></div><div><dt>Drawer travel · camera</dt><dd>{metric(evidence?.visual_drawer_travel_m, 'cm', 100)}</dd></div></> : <><div><dt>Mug center error · physics</dt><dd>{metric(physics?.goal_error_m, 'mm', 1000)}</dd></div><div><dt>Mug center error · camera</dt><dd>{metric(evidence?.camera_goal_error_m, 'mm', 1000)}</dd></div><div><dt>Maximum mug center height</dt><dd>{metric(physics?.maximum_center_height_m, 'cm', 100)}</dd></div></>}</dl>
    <div className="contact-source"><ShieldCheck size={14} /><span>Contact checks use an independent physics verifier. {typeof physics?.geometry === 'string' ? physics.geometry : 'Grasp geometry and hardware calibration are part of the current research scope.'}</span></div>
  </section>
}
