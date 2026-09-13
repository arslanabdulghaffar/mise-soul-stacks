"""Atomic scene publication for simultaneous API, test and collector readers."""
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_scene(tree, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, prefix=output.stem + '-', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            tree.write(stream, encoding='utf-8', xml_declaration=True)
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
