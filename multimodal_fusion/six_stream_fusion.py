#!/usr/bin/env python3
"""：Skeleton  +  RGB +  IMU（ logits/）。

：
  - Skeleton: outputs/{proto}/{model}_{stream}/result.pkl（）
  - IMU:      training/outputs/{proto}_ispl_inception_full_w256/test_scores.pkl
  - Video:    video/outputs/{proto}/r2plus1d_badminton_r34_kinetics/test_scores.pkl

 IMU/Video ，:
  python3 export_test_scores.py --protocol xsub

Usage:
  python3 .py
  python3 .py --protocol xacq
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fusion_utils import (
    eval_scores,
    fuse_average,
    load_labels,
    load_scores,
    load_skeleton_scores,
    scores_path,
)

OUT_MD = ROOT / ".md"
OUT_JSON = ROOT / "multimodal/fusion_6stream_results.json"

IMU_MODEL = "ispl_inception"
VIDEO_TAG = "r2plus1d_badminton_r34_kinetics"
SK_MODEL = "protogcn"
STREAMS = ["j", "b", "jm", "bm"]


def pct(v: float) -> str:
    return f"{100 * v:.2f}%"


def run_protocol(protocol: str) -> dict:
    _, labels = load_labels(protocol)

    imu_p = scores_path("imu", protocol, f"{IMU_MODEL}_full_w256")
    vid_p = scores_path("video", protocol, VIDEO_TAG)
    if not imu_p.is_file():
        raise FileNotFoundError(f" IMU : {imu_p}\n: python3 export_test_scores.py --protocol {protocol} --modality imu")
    if not vid_p.is_file():
        raise FileNotFoundError(f" Video : {vid_p}\n: python3 export_test_scores.py --protocol {protocol} --modality video")

    imu_s, imu_y, imu_m = load_scores(imu_p)
    vid_s, vid_y, vid_m = load_scores(vid_p)

    sk_parts = []
    sk_meta = {}
    for stream in STREAMS:
        sk_s = load_skeleton_scores(protocol, SK_MODEL, stream)
        sk_parts.append(sk_s)
        acc, mca = eval_scores(sk_s, labels[: len(sk_s)])
        sk_meta[stream] = {"acc": acc, "macro_f1": mca}

    sk4 = fuse_average(sk_parts)
    parts6 = sk_parts + [vid_s, imu_s]
    fused6 = fuse_average(parts6)

    # 
    sk4_acc, sk4_mca = eval_scores(sk4, labels)
    f6_acc, f6_mca = eval_scores(fused6, labels)

    return {
        "protocol": protocol,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "fusion": "score_average",
        "streams": {
            "skeleton_model": SK_MODEL,
            "skeleton_streams": STREAMS,
            "video_model": VIDEO_TAG,
            "imu_model": IMU_MODEL,
        },
        "single_stream": {
            "imu": {"acc": float(imu_m["accuracy"]), "macro_f1": float(imu_m["macro_f1"])},
            "video": {"acc": float(vid_m["accuracy"]), "macro_f1": float(vid_m["macro_f1"])},
            **{f"sk_{k}": v for k, v in sk_meta.items()},
        },
        "skeleton_4stream": {"acc": sk4_acc, "macro_f1": sk4_mca},
        "fusion_6stream": {"acc": f6_acc, "macro_f1": f6_mca, "n_test": int(len(labels))},
        "sources": {
            "imu_scores": str(imu_p),
            "video_scores": str(vid_p),
            "skeleton_scores": [
                str(ROOT / "skeleton/outputs" / protocol / (SK_MODEL if s == "j" else f"{SK_MODEL}_{s}") / "result.pkl")
                for s in STREAMS
            ],
        },
    }


def build_markdown(results: list[dict]) -> str:
    lines = [
        "# ",
        "",
        f"> : `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## ",
        "",
        "- **Skeleton **: protogcn · j / b / jm / bm（`result.pkl` logits）",
        "- **RGB**: r2plus1d ·  + Kinetics（`test_scores.pkl`）",
        "- **IMU**: ispl_inception · full_w256（`test_scores.pkl`）",
        "- ****:  6  ****",
        "",
        "## ",
        "",
        "|  | Skeleton  | +RGB | +IMU（） |",
        "|------|--------------|------|-------------|",
    ]
    for r in results:
        p = r["protocol"].upper()
        s4 = r["skeleton_4stream"]
        imu = r["single_stream"]["imu"]
        vid = r["single_stream"]["video"]
        f6 = r["fusion_6stream"]
        lines.append(
            f"| {p} | {pct(s4['acc'])} | {pct(vid['acc'])} | **{pct(f6['acc'])}** |"
        )

    lines += ["", "### ", ""]
    for r in results:
        p = r["protocol"].upper()
        f6 = r["fusion_6stream"]
        s4 = r["skeleton_4stream"]
        lines += [
            f"#### {p}",
            "",
            f"- Skeleton  (protogcn): Acc {pct(s4['acc'])}, MF1 {pct(s4['macro_f1'])}",
            f"- Video (r2plus1d): Acc {pct(r['single_stream']['video']['acc'])}, MF1 {pct(r['single_stream']['video']['macro_f1'])}",
            f"- IMU (ispl_inception): Acc {pct(r['single_stream']['imu']['acc'])}, MF1 {pct(r['single_stream']['imu']['macro_f1'])}",
            f"- ****: Acc **{pct(f6['acc'])}**, MF1 **{pct(f6['macro_f1'])}**",
            "",
        ]
    lines += [
        "---",
        "",
        ":",
        "```bash",
        "python3 export_test_scores.py --protocol xsub   #  IMU/Video ",
        "python3 export_test_scores.py --protocol xacq",
        "python3 .py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="6-stream fusion from score files")
    parser.add_argument("--protocol", choices=["xsub", "xacq", "both"], default="both")
    args = parser.parse_args()
    protos = ["xsub", "xacq"] if args.protocol == "both" else [args.protocol]

    results = [run_protocol(p) for p in protos]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(build_markdown(results), encoding="utf-8")

    for r in results:
        f6 = r["fusion_6stream"]
        print(f"{r['protocol']}: 6-stream Acc {pct(f6['acc'])} MF1 {pct(f6['macro_f1'])}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
