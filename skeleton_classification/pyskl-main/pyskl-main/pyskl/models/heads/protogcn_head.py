"""ProtoGCN classification head (with class-specific contrastive loss)."""

from __future__ import annotations

import sys
from pathlib import Path

from ..builder import HEADS

_SKEL_ROOT = Path(__file__).resolve().parents[5]
_PROTO_ROOT = _SKEL_ROOT / "ProtoGCN-main"
if _PROTO_ROOT.is_dir() and str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

from protogcn.models.heads.simple_head import SimpleHead as _ProtoGCNHead  # noqa: E402


@HEADS.register_module()
class ProtoGCNHead(_ProtoGCNHead):
    pass
