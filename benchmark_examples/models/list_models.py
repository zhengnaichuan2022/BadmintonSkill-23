#!/usr/bin/env python3
"""List bundled model code in the anonymous additional."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
paths = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)
print(f"Bundled Python modules: {len(paths)}")
for key in ["com_m2u", "badmintonskill23", "opentad_datasets", "imu/models"]:
    hits = [p.relative_to(ROOT) for p in paths if key in str(p)]
    print(f"  {key}: {len(hits)} files")
