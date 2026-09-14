import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import type { Camera, Run } from './types'

export const CAMERAS: { id: Camera; label: string; short: string }[] = [
  { id: 'top', label: 'Overhead', short: 'CAM 01' },
  { id: 'wrist_a', label: 'Arm A · wrist', short: 'CAM 02' },
  { id: 'wrist_b', label: 'Arm B · wrist', short: 'CAM 03' },
  { id: 'side_a', label: 'Arm A · side', short: 'CAM 04' },
  { id: 'side_b', label: 'Arm B · side', short: 'CAM 05' },
]

export function CameraPreviews({ run, camera, replay, live, master, onSelect }: {
  run: Run | null; camera: Camera; replay: boolean; live: boolean;
  master: RefObject<HTMLVideoElement | null>; onSelect: (camera: Camera) => void;
}) {
  const videos = useRef(new Map<Camera, HTMLVideoElement>())
  const synchronize = () => {
    const main = master.current
    if (!main || !replay) return
    for (const video of videos.current.values()) {
      if (video.readyState < 1) continue
      const target = Math.min(main.currentTime, video.duration)
      if (Math.abs(video.currentTime - target) > .3) video.currentTime = target
      video.playbackRate = main.playbackRate
      if (main.paused || main.ended) video.pause()
      else if (video.paused && !video.ended) void video.play().catch(() => {})
    }
  }
  useEffect(() => {
    if (!replay) return
    const main = master.current
    if (!main) return
    const events = ['play', 'pause', 'seeking', 'seeked', 'timeupdate', 'ratechange', 'loadedmetadata']
    events.forEach(event => main.addEventListener(event, synchronize))
    synchronize()
    return () => events.forEach(event => main.removeEventListener(event, synchronize))
  }, [run?.id, camera, replay, master])

  return <div className="camera-previews" aria-label="Other camera views">
    {CAMERAS.filter(view => view.id !== camera).map(view => {
      const video = run?.artifacts?.[`video_${view.id}`] || (view.id === 'top' ? run?.artifacts?.video : undefined)
      const available = !!run && (!replay || !!video)
      return <button key={`${run?.id}-${view.id}`} className="camera-preview" disabled={!available}
        aria-label={`Enlarge ${view.label}`} onClick={() => onSelect(view.id)}>
        <div className="camera-preview-image">
          {available && replay ? <video muted playsInline preload="metadata" src={video} aria-hidden="true"
            ref={element => { if (element) videos.current.set(view.id, element); else videos.current.delete(view.id) }}
            onLoadedMetadata={synchronize} /> : available ? <img
            src={`/api/runs/${run!.id}/${live ? 'camera' : 'frame'}/${view.id}`} alt=""
            onError={event => event.currentTarget.classList.add('unavailable')}
            onLoad={event => event.currentTarget.classList.remove('unavailable')} /> :
            <span>{run ? 'Not recorded' : 'Awaiting run'}</span>}
        </div>
        <span className="camera-preview-label">{view.label}<span>{available ? 'Enlarge ↗' : '—'}</span></span>
      </button>
    })}
  </div>
}
