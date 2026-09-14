"""Check a fresh deployed console and complete one recorded physical mug task.

Uses only Python's standard library so the host needs no project dependencies.
Run against an isolated writable runtime; this creates an immutable smoke run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def wait_for_health(get_json, timeout: float):
    """Allow the published container port to reset connections during startup."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return get_json('/api/health')
        except (URLError, TimeoutError, ConnectionError):
            if time.monotonic() >= deadline:
                raise RuntimeError('Runtime did not become healthy before timeout') from None
            time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300,
                        help="Maximum seconds to wait for startup and task completion, separately")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    base = args.url.rstrip("/")

    def request(path: str, payload: dict | None = None) -> tuple[bytes, str]:
        if not path.startswith("/") or urlsplit(path).netloc:
            raise RuntimeError(f"Unexpected artifact URL: {path}")
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(base + path, data=data,
                      headers={"Content-Type": "application/json"} if data is not None else {})
        with urlopen(req, timeout=30) as response:
            return response.read(), response.headers.get_content_type()

    def get_json(path: str, payload: dict | None = None):
        return json.loads(request(path, payload)[0])

    health = wait_for_health(get_json, args.timeout)
    assert health["status"] == "ok" and health["capabilities"]["live_control"], health

    body, content_type = request("/")
    assert content_type == "text/html", "Packaged frontend is missing"
    assets = re.findall(r'(?:src|href)="(/assets/[^"\s]+\.(?:js|css))"', body.decode())
    assert assets, "Frontend has no production JS/CSS assets"
    for asset in assets:
        data, _ = request(asset)
        assert data, f"Empty frontend asset: {asset}"
    plan = get_json("/api/plans", {"command": "Set the table."})
    assert plan["executable"] and len(plan["plan"]["steps"]) == 7, plan

    run = get_json("/api/runs", {"command": "Place the mug in the upper-right with arm B.",
                                 "seed": 1001, "controller": "contact_expert",
                                 "preset": "nominal", "recovery_mode": "adaptive"})
    run_path = f"/api/runs/{run['id']}"
    print(f"Started isolated physical smoke run {run['id']}", flush=True)
    deadline = time.monotonic() + args.timeout
    while run["status"] not in {"completed", "failed", "stopped"}:
        if time.monotonic() >= deadline:
            get_json(run_path + "/control", {"action": "stop"})
            raise RuntimeError(f"Physical smoke run timed out: {run['id']}")
        time.sleep(1)
        run = get_json(run_path)
    summary = run["summary"]
    assert run["status"] == "completed", summary
    assert summary["autonomous_contact_skill_success"] and not summary["assisted"], summary

    # Finishing the run publishes its manifest immediately after terminal status.
    deadline = time.monotonic() + 10
    while True:
        try:
            manifest = get_json(run_path + "/artifacts/manifest.json")
            break
        except URLError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.2)
    files = manifest["files_sha256"]
    assert {"top.mp4", "wrist_a.mp4", "wrist_b.mp4", "summary.json", "timing.csv", "trace.jsonl", "run.json"} <= files.keys(), files
    for filename, expected in files.items():
        content, _ = request(run_path + "/artifacts/" + filename)
        assert content and hashlib.sha256(content).hexdigest() == expected, filename
    print(json.dumps({"status": "passed", "run_id": run["id"],
                      "verified_artifacts": len(files),
                      "wall_seconds": summary["wall_seconds"],
                      "autonomous_contact_skill_success": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
