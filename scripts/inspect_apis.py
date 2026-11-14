"""Record installed signatures and source hashes; run with uv run python scripts/inspect_apis.py."""

import inspect
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))


def main() -> None:
    from anomalib.data import Folder, ImageBatch, ImageItem
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    from brickcell.util import save_json, sha256

    entries = {}
    for value in (
        Folder,
        ImageItem,
        ImageBatch,
        Patchcore,
        Patchcore.configure_pre_processor,
        Engine.fit,
        Engine.export,
        Patchcore.to_openvino,
    ):
        source = Path(inspect.getfile(value))
        entries[value.__qualname__] = {
            "signature": str(inspect.signature(value)),
            "source": str(source),
            "source_sha256": sha256(source),
        }
    save_json(Path("artifacts/api_signatures.json"), entries)
    print("Saved artifacts/api_signatures.json")


if __name__ == "__main__":
    main()
