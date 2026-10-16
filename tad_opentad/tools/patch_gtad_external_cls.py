#!/usr/bin/env python3
"""Point all GTAD benchmark configs to StandardClassifier JSON paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "configs" / "beiyu" / "benchmark"

OLD = """    external_cls=dict(
        type='UniformTopKClassifier',
        class_names='data/beiyu/annotations/category_idx.txt',
        topk=11,
    ),"""

for path in sorted(BENCH.glob("*_gtad.py")):
    stem = path.stem  # e.g. skel_j_xsub_gtad
    feat_protocol = stem[: -len("_gtad")]  # skel_j_xsub
    new = f"""    external_cls=dict(
        type='StandardClassifier',
        path='data/beiyu/classifiers/{feat_protocol}.json',
        topk=11,
        apply_softmax=False,
    ),"""
    text = path.read_text(encoding="utf-8")
    if OLD not in text:
        if "StandardClassifier" in text:
            continue
        raise SystemExit(f"unexpected content: {path}")
    path.write_text(text.replace(OLD, new), encoding="utf-8")
    print("patched", path.name)
