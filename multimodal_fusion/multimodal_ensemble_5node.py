#!/usr/bin/env python3
"""5-node IMU + RGB + Skeleton  score-average ensemble。

 1-node ：
  - IMU  training/outputs/5node/{protocol}_{model}_full_w256_* 
  -  training/cache/{protocol}_full_w256_5node.npz (N,256,30)
  - Video / Skeleton  multimodal_ensemble.py 

Usage:
  python3 multimodal_ensemble_5node.py                  #  + 
  python3 multimodal_ensemble_5node.py --skip-export    #  test_scores.pkl
  python3 multimodal_ensemble_5node.py --protocol xsub
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fusion_utils import (  # noqa: E402
    eval_scores,
    fuse_average,
    load_labels,
    load_scores,
    load_skeleton_scores,
    save_scores,
    scores_path,
)

OUT_JSON = ROOT / "multimodal/ensemble_results_5node.json"
OUT_MD = ROOT / "_5node.md"

VIDEO_TAG = "r2plus1d_badminton_r34_kinetics"

# 5-node （ broken  r_tsf_net）
IMU_5NODE_MODELS = [
    "ispl_inception",
    "resnet_se",
    "residual_bilstm",
    "cnn_lstm",
    "stacked_lstm",
    "mobile_hart_xs",
    "bilstm",
    "vanilla_lstm",
    "cnn",
    "deep_conv_lstm",
    "icg_net",
    "mch_cnn_gru",
]

#  IMU ；
DEFAULT_IMU = {
    "xsub": "resnet_se",
    "xacq": "residual_bilstm",
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
    return {"acc": float(acc), "macro_f1": float(mca)}


def latest_5node_dir(protocol: str, model: str) -> Path:
    pat = str(ROOT / "training/outputs/5node" / f"{protocol}_{model}_full_w256_*")
    dirs = [
        p
        for p in sorted(glob.glob(pat), key=os.path.getmtime)
        if Path(p).is_dir() and (Path(p) / "results.json").is_file()
    ]
    if not dirs:
        raise FileNotFoundError(f"no 5node output: {pat}")
    return Path(dirs[-1])


def imu_scores_path(protocol: str, model: str) -> Path:
    """Canonical path for 5-node scores (stable, no timestamp)."""
    return ROOT / "training/outputs/5node" / f"{protocol}_{model}_full_w256" / "test_scores.pkl"


def _keras_custom_objects(model_name: str) -> dict:
    """Collect Layer subclasses / callables needed to deserialize .keras."""
    sys.path.insert(0, str(ROOT / "training"))
    from models.registry import load_model_module  # type: ignore

    mod = load_model_module(model_name)
    objs = {}
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and name[0].isupper():
            objs[name] = obj
    # also common lowercase layer aliases used in some HAR ports
    for name in (
        "SensorWiseMHA",
        "ClassToken",
        "DropPath",
        "GatedLinearUnit",
        "PatchEncoder",
        "Prompts",
        "SensorPatches",
        "mixAccGyro",
        "liteFormer",
        "Multihead3dRotation",
    ):
        if hasattr(mod, name):
            objs[name] = getattr(mod, name)
    return objs


def export_imu_5node(protocol: str, model: str, force: bool = False) -> Path:
    import tensorflow as tf

    out = imu_scores_path(protocol, model)
    if out.is_file() and not force:
        print(f"[skip] {out}")
        return out

    src = latest_5node_dir(protocol, model)
    cache = np.load(ROOT / "training/cache" / f"{protocol}_full_w256_5node.npz", allow_pickle=True)
    X_test = cache["X_test"].astype(np.float32)
    y_test = cache["y_test"].astype(np.int64)

    mean = np.load(src / "norm_mean.npy")
    std = np.load(src / "norm_std.npy")
    Xn = (X_test - mean) / std

    model_path = src / "best.keras"
    if not model_path.is_file():
        model_path = src / "final.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"no keras weights in {src}")

    custom = _keras_custom_objects(model)
    try:
        keras_model = tf.keras.models.load_model(
            str(model_path), compile=False, custom_objects=custom
        )
    except Exception as e:
        raise RuntimeError(f"load failed {protocol}/{model}: {e}") from e
    probs = keras_model.predict(Xn, batch_size=64, verbose=0)
    acc, mca = eval_scores(probs, y_test)

    save_scores(
        out,
        probs.astype(np.float32),
        y_test,
        {
            "modality": "imu_5node",
            "protocol": protocol,
            "model": model,
            "source_dir": str(src),
            "accuracy": acc,
            "macro_f1": mca,
            "n_channels": 30,
            "window": 256,
        },
    )
    # also drop a copy next to the timestamped run
    save_scores(
        src / "test_scores.pkl",
        probs.astype(np.float32),
        y_test,
        {
            "modality": "imu_5node",
            "protocol": protocol,
            "model": model,
            "source_dir": str(src),
            "accuracy": acc,
            "macro_f1": mca,
        },
    )
    print(f"IMU5 {protocol}/{model}: acc={acc:.4f} mca={mca:.4f} -> {out}")
    return out


def export_all_imu(protocols: List[str], models: List[str], force: bool = False) -> None:
    for proto in protocols:
        for model in models:
            try:
                export_imu_5node(proto, model, force=force)
            except (FileNotFoundError, RuntimeError, ValueError) as e:
                print(f"[warn] skip {proto}/{model}: {e}")


def _load_sk_streams(protocol: str, model: str, streams: List[str]) -> List[np.ndarray]:
    return [load_skeleton_scores(protocol, model, s) for s in streams]


def run_sk_model(
    protocol: str,
    sk_model: str,
    labels: np.ndarray,
    imu_s: np.ndarray,
    vid_s: np.ndarray,
) -> dict:
    has_four = sk_model != "posec3d_joint"
    streams = FOUR_STREAMS if has_four else TWO_STREAMS
    parts = _load_sk_streams(protocol, sk_model, streams)
    n = len(labels)
    for i, p in enumerate(parts):
        if len(p) != n:
            raise ValueError(f"{sk_model} stream {streams[i]} len {len(p)} != {n}")

    bj = fuse_average(parts[:2])
    sk_fused = fuse_average(parts) if has_four else bj
    multimodal = fuse_average(parts + [vid_s, imu_s])
    sk3 = fuse_average([sk_fused, vid_s, imu_s])
    # also: skeleton + 5node IMU only (no RGB)
    sk_imu = fuse_average(parts + [imu_s])
    # RGB + 5node IMU only
    vid_imu = fuse_average([vid_s, imu_s])

    return {
        "model": sk_model,
        "skeleton_streams": streams,
        "fusion_kind": "6stream" if has_four else "4stream",
        "n_streams": 6 if has_four else 4,
        "skeleton_bj": _metric(bj, labels),
        "skeleton_fused": _metric(sk_fused, labels),
        "multimodal_3way": _metric(sk3, labels),
        "multimodal": _metric(multimodal, labels),
        "skeleton_imu": _metric(sk_imu, labels),
        "video_imu": _metric(vid_imu, labels),
    }


def load_video(protocol: str) -> Tuple[np.ndarray, dict, Path]:
    vid_p = scores_path("video", protocol, VIDEO_TAG)
    if not vid_p.is_file():
        raise FileNotFoundError(vid_p)
    vid_s, _, vid_m = load_scores(vid_p)
    return vid_s, vid_m, vid_p


def load_imu5(protocol: str, model: str) -> Tuple[np.ndarray, dict, Path]:
    p = imu_scores_path(protocol, model)
    if not p.is_file():
        raise FileNotFoundError(p)
    s, _, m = load_scores(p)
    return s, m, p


def load_imu1(protocol: str) -> Optional[Tuple[np.ndarray, dict, Path]]:
    p = scores_path("imu", protocol, "ispl_inception_full_w256")
    if not p.is_file():
        return None
    s, _, m = load_scores(p)
    return s, m, p


def run_protocol_with_imu(
    protocol: str,
    imu_model: str,
    labels: np.ndarray,
    vid_s: np.ndarray,
    vid_m: dict,
    vid_p: Path,
) -> dict:
    imu_s, imu_m, imu_p = load_imu5(protocol, imu_model)
    if len(imu_s) != len(labels) or len(vid_s) != len(labels):
        raise ValueError(
            f"{protocol}: len mismatch labels={len(labels)} imu={len(imu_s)} vid={len(vid_s)}"
        )

    per_model = []
    for sk in SK_MODELS:
        try:
            per_model.append(run_sk_model(protocol, sk, labels, imu_s, vid_s))
        except FileNotFoundError as e:
            per_model.append({"model": sk, "error": str(e)})

    ok = [m for m in per_model if "multimodal" in m]
    best = max(ok, key=lambda m: m["multimodal"]["acc"]) if ok else None

    # pairwise / modality mixes (independent of skeleton model choice for vid+imu)
    vid_imu = _metric(fuse_average([vid_s, imu_s]), labels)

    return {
        "protocol": protocol,
        "imu_model": imu_model,
        "imu_nodes": 5,
        "video_model": VIDEO_TAG,
        "n_test": int(len(labels)),
        "single_modality": {
            "video": {"acc": float(vid_m["accuracy"]), "macro_f1": float(vid_m["macro_f1"])},
            "imu": {"acc": float(imu_m["accuracy"]), "macro_f1": float(imu_m["macro_f1"])},
        },
        "video_imu": vid_imu,
        "sources": {"video_scores": str(vid_p), "imu_scores": str(imu_p)},
        "best_multimodal": {
            "model": best["model"],
            "acc": best["multimodal"]["acc"],
            "macro_f1": best["multimodal"]["macro_f1"],
            "fusion_kind": best["fusion_kind"],
        }
        if best
        else None,
        "models": per_model,
    }


def imu_model_scan(protocol: str, labels: np.ndarray, vid_s: np.ndarray) -> List[dict]:
    """Sweep 5-node IMU models with fixed best skeleton (protogcn 6-stream + poseC3D)."""
    rows = []
    for imu_model in IMU_5NODE_MODELS:
        try:
            imu_s, imu_m, _ = load_imu5(protocol, imu_model)
        except FileNotFoundError:
            continue
        if len(imu_s) != len(labels):
            continue
        row = {
            "imu_model": imu_model,
            "imu_acc": float(imu_m["accuracy"]),
            "imu_mf1": float(imu_m["macro_f1"]),
            "video_imu": _metric(fuse_average([vid_s, imu_s]), labels),
        }
        for sk in ("protogcn", "posec3d_joint", "ctrgcn"):
            try:
                r = run_sk_model(protocol, sk, labels, imu_s, vid_s)
                row[f"mm_{sk}"] = r["multimodal"]
                row[f"sk_imu_{sk}"] = r["skeleton_imu"]
            except FileNotFoundError:
                pass
        rows.append(row)
    rows.sort(key=lambda r: r.get("mm_protogcn", {}).get("acc", 0.0), reverse=True)
    return rows


def imu_ensemble_mix(
    protocol: str,
    labels: np.ndarray,
    vid_s: np.ndarray,
    top_k: int = 3,
) -> List[dict]:
    """Ensemble top-k 5-node IMU models, optionally with video/skeleton."""
    scored = []
    for m in IMU_5NODE_MODELS:
        try:
            s, meta, _ = load_imu5(protocol, m)
        except FileNotFoundError:
            continue
        if len(s) != len(labels):
            continue
        scored.append((float(meta["accuracy"]), m, s))
    scored.sort(reverse=True)
    if not scored:
        return []

    combos = []
    for k in range(1, min(top_k, len(scored)) + 1):
        names = [x[1] for x in scored[:k]]
        imu_fuse = fuse_average([x[2] for x in scored[:k]])
        entry = {
            "imu_models": names,
            "k": k,
            "imu_ensemble": _metric(imu_fuse, labels),
            "video_imu": _metric(fuse_average([vid_s, imu_fuse]), labels),
        }
        try:
            parts = _load_sk_streams(protocol, "protogcn", FOUR_STREAMS)
            entry["mm_protogcn"] = _metric(fuse_average(parts + [vid_s, imu_fuse]), labels)
            entry["sk_imu_protogcn"] = _metric(fuse_average(parts + [imu_fuse]), labels)
        except FileNotFoundError:
            pass
        try:
            parts = _load_sk_streams(protocol, "posec3d_joint", TWO_STREAMS)
            entry["mm_posec3d"] = _metric(fuse_average(parts + [vid_s, imu_fuse]), labels)
        except FileNotFoundError:
            pass
        combos.append(entry)
    return combos


def compare_1node(protocol: str, labels: np.ndarray, vid_s: np.ndarray, imu5_model: str) -> dict:
    """Side-by-side 1-node vs 5-node with same skeleton/video."""
    out: Dict = {"protocol": protocol, "imu5_model": imu5_model}
    imu1 = load_imu1(protocol)
    imu5_s, imu5_m, _ = load_imu5(protocol, imu5_model)
    out["imu5"] = {"acc": float(imu5_m["accuracy"]), "macro_f1": float(imu5_m["macro_f1"])}

    for sk in ("protogcn", "posec3d_joint"):
        try:
            r5 = run_sk_model(protocol, sk, labels, imu5_s, vid_s)
            out[f"mm5_{sk}"] = r5["multimodal"]
        except FileNotFoundError:
            pass
        if imu1 is not None:
            s1, m1, _ = imu1
            out["imu1"] = {"acc": float(m1["accuracy"]), "macro_f1": float(m1["macro_f1"])}
            try:
                r1 = run_sk_model(protocol, sk, labels, s1, vid_s)
                out[f"mm1_{sk}"] = r1["multimodal"]
            except FileNotFoundError:
                pass
    return out


def build_markdown(payload: dict) -> str:
    lines = [
        "# （5-node IMU）",
        "",
        f"> : `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## ",
        "",
        "- **Video**: r2plus1d ·  + Kinetics",
        "- **IMU**: **5-node**（30-ch, full_w256） Keras  test softmax",
        "- **Skeleton**:  j/b/jm/bm ；PoseC3D  j/b",
        "- ****: score average（）",
        "  -  → 6 （j+b+jm+bm+Vid+IMU5）",
        "  - PoseC3D → 4 （j+b+Vid+IMU5）",
        "- ：Video+IMU5、Skeleton+IMU5、 IMU  ensemble",
        "",
    ]

    for r in payload["main"]:
        p = r["protocol"].upper()
        vid = r["single_modality"]["video"]
        imu = r["single_modality"]["imu"]
        best = r["best_multimodal"]
        lines += [
            f"## {p} · （IMU5 = `{r['imu_model']}`）",
            "",
            f"- Video: Acc {pct(vid['acc'])}, MF1 {pct(vid['macro_f1'])}",
            f"- IMU 5-node `{r['imu_model']}`: Acc {pct(imu['acc'])}, MF1 {pct(imu['macro_f1'])}",
            f"- Video+IMU5: Acc {pct(r['video_imu']['acc'])}, MF1 {pct(r['video_imu']['macro_f1'])}",
            f"- ****: {best['model']} ({best['fusion_kind']}) — "
            f"Acc **{pct(best['acc'])}**, MF1 **{pct(best['macro_f1'])}**",
            "",
            "| Skeleton | Sk  | b+j | Sk+IMU5 | 3-way | +Vid+IMU5 |",
            "|---------|--------|-----|---------|-------|---------------|",
        ]
        for m in r["models"]:
            if "error" in m:
                lines.append(f"| {m['model']} | — | — | — | — |  |")
                continue
            sk = m["skeleton_fused"]
            bj = m["skeleton_bj"]
            si = m["skeleton_imu"]
            m3 = m["multimodal_3way"]
            mm = m["multimodal"]
            sk_label = "4s" if m["fusion_kind"] == "6stream" else "bj"
            lines.append(
                f"| {m['model']} | {sk_label} {pct(sk['acc'])} | {pct(bj['acc'])} | "
                f"{pct(si['acc'])} | {pct(m3['acc'])} | **{pct(mm['acc'])}** |"
            )
        lines.append("")

    # 1-node vs 5-node
    lines += ["## 1-node vs 5-node IMU（ Video / Skeleton）", ""]
    lines += [
        "| Protocol | IMU1 Acc | IMU5 Acc (model) | mm1 protogcn | mm5 protogcn | mm1 poseC3D | mm5 poseC3D |",
        "|----------|----------|------------------|--------------|--------------|-------------|-------------|",
    ]
    for c in payload.get("compare_1node", []):
        p = c["protocol"]
        i1 = c.get("imu1", {})
        i5 = c.get("imu5", {})
        lines.append(
            f"| {p} | {pct(i1.get('acc', 0)) if i1 else '—'} | "
            f"{pct(i5.get('acc', 0))} (`{c['imu5_model']}`) | "
            f"{pct(c['mm1_protogcn']['acc']) if 'mm1_protogcn' in c else '—'} | "
            f"{pct(c['mm5_protogcn']['acc']) if 'mm5_protogcn' in c else '—'} | "
            f"{pct(c['mm1_posec3d_joint']['acc']) if 'mm1_posec3d_joint' in c else '—'} | "
            f"{pct(c['mm5_posec3d_joint']['acc']) if 'mm5_posec3d_joint' in c else '—'} |"
        )
    lines.append("")

    # IMU model scan
    for proto, rows in payload.get("imu_scan", {}).items():
        lines += [
            f"## {proto.upper()} · 5-node IMU （ Skeleton）",
            "",
            "| IMU5  | IMU Acc | Vid+IMU | mm protogcn | mm poseC3D | mm ctrgcn | Sk+IMU protogcn |",
            "|-----------|---------|---------|-------------|------------|-----------|-----------------|",
        ]
        for row in rows:
            lines.append(
                f"| {row['imu_model']} | {pct(row['imu_acc'])} | "
                f"{pct(row['video_imu']['acc'])} | "
                f"{pct(row['mm_protogcn']['acc']) if 'mm_protogcn' in row else '—'} | "
                f"{pct(row['mm_posec3d_joint']['acc']) if 'mm_posec3d_joint' in row else '—'} | "
                f"{pct(row['mm_ctrgcn']['acc']) if 'mm_ctrgcn' in row else '—'} | "
                f"{pct(row['sk_imu_protogcn']['acc']) if 'sk_imu_protogcn' in row else '—'} |"
            )
        lines.append("")

    # Multi-IMU ensemble
    for proto, combos in payload.get("imu_ensemble", {}).items():
        lines += [
            f"## {proto.upper()} · Top-k 5-node IMU ensemble + ",
            "",
            "| k | IMU models | IMU ens | Vid+IMU | Sk+IMU protogcn | mm protogcn | mm poseC3D |",
            "|---|------------|---------|---------|-----------------|-------------|------------|",
        ]
        for c in combos:
            lines.append(
                f"| {c['k']} | {', '.join(c['imu_models'])} | "
                f"{pct(c['imu_ensemble']['acc'])} | {pct(c['video_imu']['acc'])} | "
                f"{pct(c['sk_imu_protogcn']['acc']) if 'sk_imu_protogcn' in c else '—'} | "
                f"{pct(c['mm_protogcn']['acc']) if 'mm_protogcn' in c else '—'} | "
                f"{pct(c['mm_posec3d']['acc']) if 'mm_posec3d' in c else '—'} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        ":",
        "```bash",
        "python3 multimodal_ensemble_5node.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="5-node multimodal ensemble")
    parser.add_argument("--protocol", choices=["xsub", "xacq", "both"], default="both")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument(
        "--imu-models",
        nargs="*",
        default=None,
        help="5-node models to export/scan (default: all ready)",
    )
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    protos = ["xsub", "xacq"] if args.protocol == "both" else [args.protocol]
    models = args.imu_models or IMU_5NODE_MODELS

    if not args.skip_export:
        print("=== Export 5-node IMU test scores ===")
        export_all_imu(protos, models, force=args.force_export)

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "fusion": "score_average",
        "imu_nodes": 5,
        "main": [],
        "compare_1node": [],
        "imu_scan": {},
        "imu_ensemble": {},
    }

    for proto in protos:
        _, labels = load_labels(proto)
        vid_s, vid_m, vid_p = load_video(proto)
        imu_model = DEFAULT_IMU[proto]
        print(f"=== Main table {proto} imu={imu_model} ===")
        main_r = run_protocol_with_imu(proto, imu_model, labels, vid_s, vid_m, vid_p)
        payload["main"].append(main_r)
        payload["compare_1node"].append(compare_1node(proto, labels, vid_s, imu_model))
        print(f"=== IMU scan {proto} ===")
        payload["imu_scan"][proto] = imu_model_scan(proto, labels, vid_s)
        print(f"=== IMU top-k ensemble {proto} ===")
        payload["imu_ensemble"][proto] = imu_ensemble_mix(proto, labels, vid_s, top_k=5)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(build_markdown(payload), encoding="utf-8")

    print("\n=== Summary ===")
    for r in payload["main"]:
        b = r["best_multimodal"]
        print(
            f"{r['protocol']} [IMU5={r['imu_model']}]: best {b['model']} "
            f"Acc {pct(b['acc'])} MF1 {pct(b['macro_f1'])} | "
            f"Vid+IMU {pct(r['video_imu']['acc'])}"
        )
    for c in payload["compare_1node"]:
        if "mm1_protogcn" in c and "mm5_protogcn" in c:
            d = c["mm5_protogcn"]["acc"] - c["mm1_protogcn"]["acc"]
            print(
                f"{c['protocol']} protogcn Δ(mm5-mm1) = {100*d:+.2f}pp "
                f"({pct(c['mm1_protogcn']['acc'])} → {pct(c['mm5_protogcn']['acc'])})"
            )
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
