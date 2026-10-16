# Restored from bytecode backup.
from __future__ import annotations
import marshal
from pathlib import Path
_PYC = Path(__file__).resolve().parents[2] / "_pyc_backup" / "tsf" / "feature.cpython-38.pyc"
exec(marshal.loads(_PYC.read_bytes()[16:]), globals())
