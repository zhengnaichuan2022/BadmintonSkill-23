#!/usr/bin/env python3
"""r_tsf_net trainer with safe weight checkpointing.

Root cause of ~20% post-train accuracy: EarlyStopping(restore_best_weights=True)
+ ModelCheckpoint(.keras) corrupt custom layers (Multihead3dRotation etc.).
Training history val was ~70%; disk evaluate collapsed.

Fix:
  - ModelCheckpoint(save_weights_only=True) -> best.weights.h5
  - EarlyStopping(restore_best_weights=False)
  - After fit, load_weights(best.weights.h5) before evaluate
"""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.callbacks as kcb

_ORIG_FIT = tf.keras.Model.fit
_ORIG_ES = kcb.EarlyStopping
_ORIG_MC = kcb.ModelCheckpoint


class _FixedEarlyStopping(_ORIG_ES):
    def __init__(self, *args, **kwargs):
        kwargs["restore_best_weights"] = False
        super().__init__(*args, **kwargs)


class _FixedModelCheckpoint(_ORIG_MC):
    def __init__(self, filepath, *args, **kwargs):
        fp = str(filepath)
        if fp.endswith(".keras"):
            fp = fp[: -len(".keras")] + ".weights.h5"
        elif fp.endswith(".h5") and not fp.endswith(".weights.h5"):
            fp = fp[: -len(".h5")] + ".weights.h5"
        kwargs["save_weights_only"] = True
        super().__init__(fp, *args, **kwargs)


def _fit_then_load_best(self, *args, **kwargs):
    history = _ORIG_FIT(self, *args, **kwargs)
    callbacks = list(kwargs.get("callbacks") or [])
    weight_path = None
    for cb in callbacks:
        fp = getattr(cb, "filepath", None)
        if fp and str(fp).endswith((".weights.h5", ".h5")) and Path(fp).is_file():
            weight_path = str(fp)
            break
    if weight_path is None:
        # Fallback: newest best.weights.h5 under OUTPUT_DIR (same run dir as ckpt)
        for cb in callbacks:
            fp = getattr(cb, "filepath", None)
            if fp:
                cand = Path(fp)
                if cand.is_file():
                    weight_path = str(cand)
                    break
    if weight_path is None:
        print("WARNING: no best.weights.h5 found; evaluating last-epoch weights")
    else:
        print(f"Loading best weights from {weight_path}")
        self.load_weights(weight_path)
        # quick sanity on a tiny batch is left to evaluate()
    return history


# Patch BEFORE importing train (binds EarlyStopping / ModelCheckpoint at import).
kcb.EarlyStopping = _FixedEarlyStopping
kcb.ModelCheckpoint = _FixedModelCheckpoint
tf.keras.Model.fit = _fit_then_load_best

# Ensure training dir is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402  # loads from _pyc_backup via marshal


if __name__ == "__main__":
    # Default model if user forgot --model
    if "--model" not in sys.argv:
        sys.argv[1:1] = ["--model", "r_tsf_net"]
    train.main()
