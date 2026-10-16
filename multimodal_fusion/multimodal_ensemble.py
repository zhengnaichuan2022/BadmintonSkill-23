#!/usr/bin/env python3
""" test ensemble： Video +  IMU +  Skeleton /。

 Skeleton （8 ）：
  -  (j/b/jm/bm)： =  + r2plus1d + ispl_inception
  -  (posec3d_joint j/b)： = b+j + r2plus1d + ispl_inception

 Skeleton  b+j / 。

Usage:
  python3 export_test_scores.py --protocol xsub --modality both
  python3 export_test_scores.py --protocol xacq --modality both
  python3 multimodal_ensemble.py
  python3 multimodal_ensemble.py --protocol xsub
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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

OUT_JSON = ROOT / "multimodal/ensemble_results.json"
OUT_MD = ROOT / ".md"
FULL_SUMMARY_MD = ROOT / ".md"

IMU_MODEL = "ispl_inception"
VIDEO_TAG = "r2plus1d_badminton_r34_kinetics"

SK_COMPLEXITY = {
    "stgcnpp": ("1.38M", "0.32G"),
    "stgcn": ("3.08M", "1.04G"),
    "ctrgcn": ("1.42M", "0.37G"),
    "msg3d": ("2.96M", "2.45G"),
    "aagcn": ("3.75M", "1.03G"),
    "posec3d_joint": ("2.00M", "15.80G"),
    "protogcn": ("4.21M", "0.55G"),
    "blockgcn": ("1.27M", "0.28G"),
}

SK_MODELS = [
    "stgcnpp",
    "stgcn",
    "ctrgcn",
    "msg3d",
    "aagcn",
    "posec3d_joint",
    "protogcn",
    "blockgcn",
]

FOUR_STREAMS = ["j", "b", "jm", "bm"]
TWO_STREAMS = ["j", "b"]


def pct(v: float) -> str:
    return f"{100 * v:.2f}%"


def _metric(scores: np.ndarray, labels: np.ndarray) -> dict:
    acc, mca = eval_scores(scores, labels)
    return {"acc": acc, "macro_f1": mca}


def _load_sk_streams(protocol: str, model: str, streams: List[str]) -> List[np.ndarray]:
    return [load_skeleton_scores(protocol, model, s) for s in streams]


def _require_scores(protocol: str) -> tuple:
    imu_p = scores_path("imu", protocol, f"{IMU_MODEL}_full_w256")
    vid_p = scores_path("video", protocol, VIDEO_TAG)
    missing = [p for p in (imu_p, vid_p) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            ":\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\n: python3 export_test_scores.py --protocol {protocol} --modality both"
        )
    imu_s, _, imu_m = load_scores(imu_p)
    vid_s, _, vid_m = load_scores(vid_p)
    return imu_s, vid_s, imu_m, vid_m, imu_p, vid_p


def run_model(protocol: str, model: str, labels: np.ndarray, imu_s: np.ndarray, vid_s: np.ndarray) -> dict:
    has_four = model != "posec3d_joint"
    streams = FOUR_STREAMS if has_four else TWO_STREAMS
    parts = _load_sk_streams(protocol, model, streams)

    n = len(labels)
    for i, p in enumerate(parts):
        if len(p) != n:
            raise ValueError(f"{model} stream {streams[i]} len {len(p)} != {n}")

    bj = fuse_average(parts[:2])
    sk_fused = fuse_average(parts) if has_four else bj

    if has_four:
        multimodal = fuse_average(parts + [vid_s, imu_s])
        fusion_kind = "6stream"
        n_streams = 6
    else:
        multimodal = fuse_average(parts + [vid_s, imu_s])
        fusion_kind = "4stream"
        n_streams = 4

    sk3 = fuse_average([sk_fused, vid_s, imu_s])

    return {
        "model": model,
        "skeleton_streams": streams,
        "fusion_kind": fusion_kind,
        "n_streams": n_streams,
        "skeleton_bj": _metric(bj, labels),
        "skeleton_fused": _metric(sk_fused, labels),
        "multimodal_3way": _metric(sk3, labels),
        "multimodal": _metric(multimodal, labels),
    }


def run_protocol(protocol: str) -> dict:
    _, labels = load_labels(protocol)
    imu_s, vid_s, imu_m, vid_m, imu_p, vid_p = _require_scores(protocol)

    per_model = []
    for model in SK_MODELS:
        try:
            per_model.append(run_model(protocol, model, labels, imu_s, vid_s))
        except FileNotFoundError as e:
            per_model.append({"model": model, "error": str(e)})

    best = max(
        (m for m in per_model if "multimodal" in m),
        key=lambda m: m["multimodal"]["acc"],
    )

    return {
        "protocol": protocol,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "fusion": "score_average",
        "video_model": VIDEO_TAG,
        "imu_model": IMU_MODEL,
        "n_test": int(len(labels)),
        "single_modality": {
            "video": {"acc": float(vid_m["accuracy"]), "macro_f1": float(vid_m["macro_f1"])},
            "imu": {"acc": float(imu_m["accuracy"]), "macro_f1": float(imu_m["macro_f1"])},
        },
        "sources": {"video_scores": str(vid_p), "imu_scores": str(imu_p)},
        "best_multimodal": {
            "model": best["model"],
            "acc": best["multimodal"]["acc"],
            "macro_f1": best["multimodal"]["macro_f1"],
            "fusion_kind": best["fusion_kind"],
        },
        "models": per_model,
    }


def build_markdown(results: list[dict]) -> str:
    lines = [
        "# ",
        "",
        f"> : `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## ",
        "",
        "- **Video**: r2plus1d ·  + Kinetics（test ）",
        "- **IMU**: ispl_inception · full_w256（test ）",
        "- **Skeleton**:  j/b/jm/bm ；PoseC3D  j/b ",
        "- ****: Skeleton  + Video + IMU ****",
        "  -  → 6 ；PoseC3D → 4  (j+b+Video+IMU)",
        "",
    ]

    for r in results:
        p = r["protocol"].upper()
        vid = r["single_modality"]["video"]
        imu = r["single_modality"]["imu"]
        best = r["best_multimodal"]
        lines += [
            f"## {p}",
            "",
            f"- Video: Acc {pct(vid['acc'])}, MF1 {pct(vid['macro_f1'])}",
            f"- IMU: Acc {pct(imu['acc'])}, MF1 {pct(imu['macro_f1'])}",
            f"- ****: {best['model']} ({best['fusion_kind']}) — Acc **{pct(best['acc'])}**, MF1 **{pct(best['macro_f1'])}**",
            "",
            "| Skeleton  | Skeleton / | b+j | 3-way (Sk+Vid+IMU) |  (+Vid+IMU) |",
            "|--------------|------------------|-----|--------------------|-----------------------|",
        ]
        for m in r["models"]:
            if "error" in m:
                lines.append(f"| {m['model']} | — | — | — | : {m['error'][:40]}… |")
                continue
            sk = m["skeleton_fused"]
            bj = m["skeleton_bj"]
            m3 = m["multimodal_3way"]
            mm = m["multimodal"]
            sk_label = "4s" if m["fusion_kind"] == "6stream" else "bj"
            lines.append(
                f"| {m['model']} | {sk_label} {pct(sk['acc'])} | {pct(bj['acc'])} | "
                f"{pct(m3['acc'])} | **{pct(mm['acc'])}** |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        ":",
        "```bash",
        "python3 export_test_scores.py --protocol xsub --modality both",
        "python3 export_test_scores.py --protocol xacq --modality both",
        "python3 multimodal_ensemble.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def build_full_summary_section(results: list[dict]) -> str:
    """ .md 。"""
    lines = [
        "## （Skeleton  +  RGB +  IMU）",
        "",
        "> ： Skeleton  **** j/b/jm/bm  ** Video**（r2plus1d ·  Kinetics）",
        ">  ** IMU**（ispl_inception · full_w256） test logits ****（score average）。",
        "> PoseC3D  jm/bm， j+b （ 4 ：j+b+Video+IMU）。",
        "> ：`python3 export_test_scores.py --protocol xsub --modality video --reuse-dump`  `python3 multimodal_ensemble.py`",
        "",
        "**（ test ）**",
        "",
    ]
    for r in results:
        p = r["protocol"].upper()
        vid = r["single_modality"]["video"]
        imu = r["single_modality"]["imu"]
        lines.append(
            f"- **{p}** Video r2plus1d: Acc {pct(vid['acc'])}, MF1 {pct(vid['macro_f1'])} · "
            f"IMU ispl_inception: Acc {pct(imu['acc'])}, MF1 {pct(imu['macro_f1'])}"
        )
    lines.append("")

    for r in results:
        p = r["protocol"].upper()
        best = r["best_multimodal"]
        lines += [
            f"### {p}",
            "",
            "|  | Params | FLOPs | Sk /bj |  Acc |  MF1 |  |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        rows = []
        for m in r["models"]:
            if "error" in m:
                params, flops = SK_COMPLEXITY.get(m["model"], ("—", "—"))
                rows.append((m["model"], params, flops, "—", "—", "—", "—"))
                continue
            params, flops = SK_COMPLEXITY.get(m["model"], ("—", "—"))
            sk = m["skeleton_fused"]
            mm = m["multimodal"]
            sk_label = "4s " + pct(sk["acc"]) if m["fusion_kind"] == "6stream" else "bj " + pct(sk["acc"])
            n = "6" if m["fusion_kind"] == "6stream" else "4"
            rows.append((m["model"], params, flops, sk_label, pct(mm["acc"]), pct(mm["macro_f1"]), n))
        rows.sort(key=lambda x: float(x[4].rstrip("%")) if x[4] != "—" else -1, reverse=True)
        for row in rows:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | **{row[4]}** | **{row[5]}** | {row[6]} |")
        lines += [
            "",
            f"- ****: {best['model']} — Acc **{pct(best['acc'])}**, MF1 **{pct(best['macro_f1'])}** "
            f"({best['fusion_kind']})",
            "",
        ]
    return "\n".join(lines)


def patch_full_summary_md(results: list[dict]) -> None:
    if not FULL_SUMMARY_MD.is_file():
        return
    import re

    text = FULL_SUMMARY_MD.read_text(encoding="utf-8")
    section = build_full_summary_section(results)
    start = "## （Skeleton  +  RGB +  IMU）"
    end = "## （test）"

    # ，
    while text.count(start) > 1:
        first = text.index(start)
        second = text.index(start, first + 1)
        mid_end = text.index(end, second)
        text = text[:second] + text[mid_end:]

    if start in text and end in text:
        pre = text.split(start)[0].rstrip()
        post = text.split(end, 1)[1]
        text = pre + "\n\n" + section.rstrip() + "\n\n---\n\n" + end + post
    else:
        marker = "---\n\n## （test）"
        if marker in text:
            text = text.replace(marker, "---\n\n" + section.rstrip() + "\n\n---\n\n## （test）")

    for r in results:
        p = r["protocol"].upper()
        best = r["best_multimodal"]
        bullet = (
            f"- **（Sk + RGB+IMU）** {best['model']}: "
            f"Acc **{pct(best['acc'])}**, MF1 **{pct(best['macro_f1'])}**"
        )
        header = f"### {p}\n"
        if header not in text:
            continue
        chunk = text.split(header, 1)[1]
        nxt = chunk.find("\n### ")
        body = chunk if nxt < 0 else chunk[:nxt]
        rest = "" if nxt < 0 else chunk[nxt:]
        lines = [ln for ln in body.splitlines() if "（Sk + RGB+IMU）" not in ln]
        if lines and lines[-1].strip() == "":
            lines.pop()
        lines.append(bullet)
        lines.append("")
        text = text.split(header, 1)[0] + header + "\n".join(lines) + rest

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "" not in text:
        text = text.replace("> :", f"> : `{ts}` · :", 1)
    else:
        text = re.sub(r"> : `[^`]+`", f"> : `{ts}`", text, count=1)

    FULL_SUMMARY_MD.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal ensemble on test set")
    parser.add_argument("--protocol", choices=["xsub", "xacq", "both"], default="both")
    args = parser.parse_args()
    protos = ["xsub", "xacq"] if args.protocol == "both" else [args.protocol]

    results = [run_protocol(p) for p in protos]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(build_markdown(results), encoding="utf-8")
    patch_full_summary_md(results)

    for r in results:
        b = r["best_multimodal"]
        print(
            f"{r['protocol']}: best {b['model']} ({b['fusion_kind']}) "
            f"Acc {pct(b['acc'])} MF1 {pct(b['macro_f1'])}"
        )
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    if FULL_SUMMARY_MD.is_file():
        print(f"Updated {FULL_SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
