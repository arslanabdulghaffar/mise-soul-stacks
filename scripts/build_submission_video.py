"""Validate and concatenate the ten predeclared full-task demonstration runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(path: Path) -> dict[str, Any] | None:
    try:
        run = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    summary = run.get("summary")
    if not isinstance(summary, dict):
        sibling = path.with_name("summary.json")
        try:
            summary = json.loads(sibling.read_text())
        except (OSError, json.JSONDecodeError):
            summary = {}
    run["summary"] = summary
    run["directory"] = str(path.parent)
    return run


def select_runs(root: Path, required_seeds: list[int]) -> tuple[list[dict[str, Any]], dict[int, int]]:
    grouped: dict[int, list[dict[str, Any]]] = {seed: [] for seed in required_seeds}
    attempt_counts = {seed: 0 for seed in required_seeds}
    for path in root.rglob("run.json"):
        run = _load_run(path)
        if not run or run.get("seed") not in grouped:
            continue
        seed = int(run["seed"])
        attempt_counts[seed] += 1
        summary = run["summary"]
        video = path.parent / "top.mp4"
        if (
            run.get("status") == "completed"
            and (run.get("scope") == "full_task" or summary.get("scope") == "full_task")
            and summary.get("full_task_success") is True
            and summary.get("assisted") is not True
            and video.is_file()
        ):
            run["video"] = str(video)
            grouped[seed].append(run)
    selected = []
    missing = []
    for seed in required_seeds:
        candidates = grouped[seed]
        if not candidates:
            missing.append(seed)
            continue
        # A dedicated recording archive should contain one attempt per seed. If it
        # contains retries, choosing the latest is disclosed in the manifest.
        candidates.sort(key=lambda row: (str(row.get("created_at", "")), str(row.get("id", ""))))
        selected.append(candidates[-1])
    if missing:
        raise ValueError(f"Missing successful, unassisted full-task videos for seeds: {missing}")
    return selected, attempt_counts


def verify_run_files(run: dict[str, Any]) -> int:
    """Check the worker's original evidence before using a recording in a demo."""
    directory = Path(run['directory'])
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('run_id') != run.get('id') or manifest.get('scope') != 'full_task':
        raise ValueError('Run manifest identity or scope does not match the recording.')
    checksums = manifest.get('files_sha256', {})
    required = {'run.json', 'summary.json', 'trace.jsonl', 'timing.csv',
                'top.mp4', 'wrist_a.mp4', 'wrist_b.mp4'}
    if not required.issubset(checksums):
        raise ValueError('Run manifest is missing required evidence checksums.')
    for name, expected in checksums.items():
        path = directory / name
        if path.resolve().parent != directory.resolve() or not path.is_file():
            raise ValueError(f'Invalid or missing manifest file: {name}')
        if sha256(path) != expected:
            raise ValueError(f'Run checksum mismatch: {name}')
    return len(checksums)


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
    raise RuntimeError("ffmpeg is unavailable. Install imageio-ffmpeg or add ffmpeg to PATH.")


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr[-4000:] if completed.stderr else completed.stdout[-4000:]
        raise RuntimeError(f"ffmpeg failed:\n{detail}")


def build_video(selected: list[dict[str, Any]], output: Path, *, speed: float = 6.0, fps: int = 24) -> None:
    from PIL import Image, ImageDraw, ImageFont
    import imageio_ffmpeg

    ffmpeg = _ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mise-demo-") as temporary:
        temp = Path(temporary)
        segments = []
        for index, run in enumerate(selected, start=1):
            segment = temp / f"seed-{int(run['seed'])}.mp4"
            preset = str(run.get("preset", "nominal")).replace("_", " ")
            label = f"SEED {run['seed']}  |  {preset.upper()}"
            badge = temp / f"seed-{int(run['seed'])}.png"
            reader = imageio_ffmpeg.read_frames(str(run["video"]))
            metadata = next(reader)
            reader.close()
            width, height = metadata["size"]
            badge_height = max(48, round(height * .22))
            canvas = Image.new("RGBA", (width, badge_height), (10, 16, 26, 225))
            painter = ImageDraw.Draw(canvas)
            font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            font_size = max(10, round(width / 22))
            font = ImageFont.truetype(str(font_path), font_size) if font_path.is_file() else ImageFont.load_default()
            painter.text((8, 5), "SeoulStack  |  SET THE TABLE", fill=(115, 212, 228, 255), font=font)
            painter.text((8, badge_height // 2 + 2), label, fill=(230, 237, 245, 255), font=font)
            canvas.save(badge)
            overlay = (
                f"[0:v]setpts=PTS/{speed:g},minterpolate=fps={fps}[fast];"
                "[fast][1:v]overlay=0:0:shortest=1"
            )
            _run([
                ffmpeg, "-y", "-loglevel", "error", "-i", str(run["video"]),
                "-loop", "1", "-i", str(badge), "-filter_complex", overlay,
                "-shortest", "-an", "-c:v", "libx264", "-preset", "medium",
                "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(segment),
            ])
            segments.append(segment)
        concat = temp / "segments.ffconcat"
        concat.write_text("ffconcat version 1.0\n" + "".join(
            f"file '{str(path).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n"
            for path in segments
        ))
        _run([
            ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
            "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(output),
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("artifacts/submission-runs"))
    parser.add_argument("--seeds", type=Path, default=Path("eval/seeds.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/submission/ten-seed-demo.mp4"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/submission/ten-seed-demo.json"))
    parser.add_argument("--speed", type=float, default=6.0, help="Playback acceleration for each complete run")
    parser.add_argument("--fps", type=int, default=24, help="Interpolated output frame rate")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0 or args.fps < 1:
        parser.error("speed and fps must be positive")
    required = [int(seed) for seed in yaml.safe_load(args.seeds.read_text())["required"]]
    if len(required) != 10 or len(set(required)) != 10:
        raise SystemExit("The submission suite must contain exactly ten distinct predeclared seeds.")
    try:
        selected, attempt_counts = select_runs(args.runs_root, required)
        if any(count != 1 for count in attempt_counts.values()):
            raise ValueError('Use a dedicated archive with exactly one declared attempt per seed; '
                             'retain retries separately rather than selecting only successes.')
        verified_files = sum(verify_run_files(run) for run in selected)
        if not args.check_only:
            build_video(selected, args.output, speed=args.speed, fps=args.fps)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    payload = {
        "schema_version": "mise.ten-seed-video.v1",
        "command": "Set the table.",
        "required_seeds": required,
        "selection_policy": "exactly one checksum-verified unassisted full-task attempt per predeclared seed",
        "verified_artifact_files": verified_files,
        "playback_speed": args.speed,
        "output_fps": args.fps,
        "attempt_counts": attempt_counts,
        "runs": [
            {
                "seed": row["seed"], "run_id": row.get("id"), "preset": row.get("preset"),
                "source": row["video"], "source_sha256": sha256(Path(row["video"])),
            }
            for row in selected
        ],
        "video": None if args.check_only else str(args.output),
        "video_sha256": None if args.check_only else sha256(args.output),
    }
    # A check must never replace the published video's binding with video=null.
    if not args.check_only:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
