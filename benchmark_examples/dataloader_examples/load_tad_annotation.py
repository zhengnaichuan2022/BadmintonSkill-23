"""Load untrimmed TAD annotation JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from benchmark_utils import get_dataset_root


def load_tad_annotation(sample_dir: Path | str) -> Tuple[dict, List[dict], dict]:
    """Load tad_annotation.json for an untrimmed continuous sequence.

    Returns
    -------
    annotation : full JSON dict
    segments   : list of segment dicts with label_id, action_code, label, segment [start, end]
    meta       : summary (video_id, duration, segment_count)
    """
    sample_dir = Path(sample_dir)
    ann_path = sample_dir / "tad_annotation.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing tad_annotation.json in {sample_dir}")

    annotation = json.loads(ann_path.read_text(encoding="utf-8"))
    segments = annotation.get("annotations", [])
    meta = {
        "video_id": annotation.get("video_id"),
        "action_code": annotation.get("action_code"),
        "duration_sec": annotation.get("duration"),
        "clip_start_sec": annotation.get("clip_start_sec"),
        "clip_end_sec": annotation.get("clip_end_sec"),
        "segment_count": len(segments),
    }
    return annotation, segments, meta


if __name__ == "__main__":
    root = get_dataset_root()
    sample = root / "samples/untrimmed_tad/V000045"
    ann, segments, meta = load_tad_annotation(sample)
    print(f"video_id={meta['video_id']} segments={meta['segment_count']} duration={meta['duration_sec']}s")
