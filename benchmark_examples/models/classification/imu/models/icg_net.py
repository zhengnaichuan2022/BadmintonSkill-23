# Restored from bytecode backup (do not compile over backup).
from __future__ import annotations
import marshal
from pathlib import Path
_PYC = Path(__file__).resolve().parents[1] / "_pyc_backup" / "models" / "icg_net.cpython-38.pyc"
exec(marshal.loads(_PYC.read_bytes()[16:]), globals())
