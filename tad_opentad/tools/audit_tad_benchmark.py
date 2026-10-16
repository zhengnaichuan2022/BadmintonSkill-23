#!/usr/bin/env python3
"""Audit TAD benchmark: missing runs, failures, suspicious metrics."""

from __future__ import annotations

import json
import re
from pathlib import Path

OPENTAD_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = OPENTAD_ROOT / "configs" / "beiyu" / "benchmark_manifest.json"
LOG_ROOT = OPENTAD_ROOT / "logs"

AVG_RE = re.compile(r"OK:(?P<name>[^:]+):(?P<map>[0-9.]+|-)")


def collect_results() -> dict[str, dict]:
    """Best result per experiment (highest mAP if multiple runs)."""
    best: dict[str, dict] = {}
    for tsv in sorted(LOG_ROOT.glob("tad_benchmark_*/results.tsv")):
        for line in tsv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OK:"):
                _, rest = line.split(":", 1)
                name, m = rest.rsplit(":", 1)
                if m == "-":
                    continue
                val = float(m)
                prev = best.get(name)
                if prev is None or prev.get("map") is None or val > prev["map"]:
                    best[name] = {"map": val, "log_dir": tsv.parent.name, "status": "ok"}
            elif line.startswith("FAIL:"):
                _, rest = line.split(":", 1)
                name, phase = rest.split(":", 1)
                if name not in best:
                    best[name] = {"map": None, "log_dir": tsv.parent.name, "status": f"fail_{phase}"}
    return best


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = collect_results()

    missing = []
    failed = []
    ok_rows = []
    suspicious = []

    for item in manifest:
        name = Path(item["config"]).stem
        feat = item["feature"]
        model = item["model"]
        proto = item["protocol"]
        rec = results.get(name)

        if rec is None:
            missing.append(name)
            continue
        if rec["status"] != "ok":
            failed.append((name, rec["status"], rec["log_dir"]))
            continue

        m = rec["map"]
        ok_rows.append((feat, model, proto, m, name, rec["log_dir"]))

        # TriDet on single-stream skeleton often underperforms; flag extreme gaps
        if model == "tridet" and feat in {"skel_j", "skel_b"} and m < 10:
            suspicious.append((name, m, "TriDet single-stream mAP < 10% (check tuning)"))

    # Cross-model gap on same feature/protocol
    by_fp: dict[tuple, list] = {}
    for feat, model, proto, m, name, logd in ok_rows:
        by_fp.setdefault((feat, proto), []).append((model, m))
    for (feat, proto), rows in by_fp.items():
        if len(rows) < 2:
            continue
        maps = [m for _, m in rows]
        if max(maps) - min(maps) > 20:
            suspicious.append(
                (f"{feat}_{proto}", min(maps), max(maps), f"spread {max(maps)-min(maps):.1f}% across models")
            )

    lines = [
        "# TAD Benchmark Audit",
        "",
        f"Manifest experiments: {len(manifest)}",
        f"With OK result: {len(ok_rows)}",
        f"Missing (never OK): {len(missing)}",
        f"Failed runs only: {len(failed)}",
        f"Suspicious: {len(suspicious)}",
        "",
        "## Missing / never completed",
        "",
    ]
    if missing:
        for n in sorted(missing):
            lines.append(f"- {n}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Failed (latest log)", ""])
    if failed:
        for n, st, logd in sorted(failed):
            lines.append(f"- {n}: {st} ({logd})")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Suspicious metrics", ""])
    if suspicious:
        for row in suspicious:
            lines.append(f"- {row}")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Completed results (by feature)", ""])
    for feat in sorted({r[0] for r in ok_rows}):
        lines.append(f"### {feat}")
        lines.append("| model | xsub | xacq |")
        lines.append("| --- | ---: | ---: |")
        models = sorted({r[1] for r in ok_rows if r[0] == feat})
        for model in models:
            xs = next((m for f, mo, p, m, _, _ in ok_rows if f == feat and mo == model and p == "xsub"), None)
            xa = next((m for f, mo, p, m, _, _ in ok_rows if f == feat and mo == model and p == "xacq"), None)
            lines.append(f"| {model} | {xs if xs is not None else '-'} | {xa if xa is not None else '-'} |")
        lines.append("")

    out = OPENTAD_ROOT / "TAD_benchmark_audit.md"
    text = "\n".join(lines)
    print(text)
    out.write_text(text, encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
