"""Package the accepted local model and its ten-seed evidence for another machine."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from mise.learned_assets import accepted_policy, model_directory
from mise.sim import ROOT
from mise.telemetry import file_hash
from scripts.validate_intel_pair import read_evaluation


def package(output: Path, evaluation: Path) -> dict:
    accepted = accepted_policy()
    model = model_directory()
    read_evaluation(evaluation)
    if (file_hash(evaluation / 'manifest.json') != accepted['evaluation_manifest_sha256']
            or file_hash(evaluation / 'results.json') != accepted['evaluation_results_sha256']):
        raise ValueError('Evaluation does not match accepted policy')
    checkpoint = model / 'checkpoint.pt'
    if file_hash(checkpoint) != accepted['checkpoint_sha256']:
        raise ValueError('Checkpoint does not match accepted policy')
    files = [checkpoint, model / 'accepted_policy.json',
             *(model / 'openvino' / name for name in accepted['model_files_sha256']),
             Path(accepted['scene_path']), *sorted(evaluation.glob('*.json'))]
    entries = {str(path.resolve().relative_to(ROOT)): path for path in files}
    hashes = {name: file_hash(path) for name, path in entries.items()}
    manifest = dict(schema_version='mise.policy-transfer.v1', files_sha256=hashes,
                    checkpoint_sha256=accepted['checkpoint_sha256'],
                    scope='learned_mug_contact_skill', intel_target_verified=False,
                    instructions='Extract into the matching repository checkout, preserving paths. Install requirements-learning.txt. Start make console; ACT supports nominal mug pick/place only.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', compression=ZIP_DEFLATED) as archive:
        for name, path in entries.items():
            archive.write(path, name)
        archive.writestr('policy-transfer-manifest.json', json.dumps(manifest, indent=2) + '\n')
    with ZipFile(output) as archive:
        import hashlib
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Archive verification failed: {name}')
    return dict(path=str(output), sha256=file_hash(output), files=len(entries), bytes=output.stat().st_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=Path('artifacts/learned_checks/context-heldout-50000'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/submission/seoulstack-act-policy.zip'))
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.evaluation), indent=2))
