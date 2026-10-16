"""Register ProtoGCN backbone from skeleton/ProtoGCN-main."""

from __future__ import annotations

import sys
from pathlib import Path

from ..builder import BACKBONES

_SKEL_ROOT = Path(__file__).resolve().parents[5]
_PROTO_ROOT = _SKEL_ROOT / "ProtoGCN-main"
if _PROTO_ROOT.is_dir() and str(_PROTO_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROTO_ROOT))

from protogcn.models.gcns.protogcn import ProtoGCN as _ProtoGCN  # noqa: E402


@BACKBONES.register_module()
class ProtoGCN(_ProtoGCN):
    """Thin wrapper so checkpoints trained with pyskl configs load cleanly."""
