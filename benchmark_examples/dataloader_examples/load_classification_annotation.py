"""Load classification preview annotation JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from benchmark_utils import get_dataset_root


def load_classification_annotation(sample_dir: Path | str) -> Tuple[dict, dict]:
    """Load annotation.json for a bundled classification preview sample.

    Returns
    -------
    annotation : parsed JSON dict
    meta       : summary fields for quick inspection
    """
    sample_dir = Path(sample_dir)
    ann_path = sample_dir / "annotation.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotation.json in {sample_dir}")

    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    segments = annotation.get("segments", [])
    meta = {
        "preview_id": annotation.get("preview_id"),
        "recording_id": annotation.get("recording_id"),
        "action_code": annotation.get("action_code"),
        "class_id": annotation.get("class_id"),
        "class_name": annotation.get("class_name"),
        "preview_duration_sec": annotation.get("preview_duration"),
        "segment_count": len(segments),
        "modalities": list(annotation.get("modalities", {}).keys()),
    }
    return annotation, meta


if __name__ == "__main__":
    root = get_dataset_root()
    sample = root / "samples/classification_previews/A01_forehand_net"
    ann, meta = load_classification_annotation(sample)
    print(
        f"preview_id={meta['preview_id']} recording_id={meta['recording_id']} "
        f"class={meta['class_name']} segments={meta['segment_count']} "
        f"modalities={meta['modalities']}"
    )
