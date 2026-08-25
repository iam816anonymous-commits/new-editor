from pathlib import Path
from typing import Dict, Any
import json

def export_render_manifest(hash_dir: Path, manifest_data: Dict[str, Any]) -> Path:
    manifest_path = hash_dir / "render_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    return manifest_path
