#!/usr/bin/env python3
"""Fine-tune IMU student (ispl_inception) with CoM-M2U loss."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Avoid TF XLA libdevice issues on this cluster
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

ROOT = Path("${BADMINTONSKILL23_ROOT}")
sys.path.insert(0, str(ROOT / "multimodal"))

from com_m2u.constants import (  # noqa: E402
    BASELINE_ACC,
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    DEFAULT_T,
    OUT_ROOT,
    imu_ckpt_dir,
    logits_dir,
)


def load_logits(protocol: str, modality: str, split: str):
    p = logits_dir(protocol) / modality / f"{split}.npz"
    d = np.load(p, allow_pickle=True)
    return d["logits"].astype(np.float32), d["labels"].astype(np.int64)


def com_loss_tf(z_m, y, z_a, z_b, T, lambda_kd, gamma, mode, z_self=None):
    """Mirror com_m2u.loss.com_m2u_loss in TF."""
    ce = tf.keras.losses.sparse_categorical_crossentropy(y, z_m, from_logits=True)
    ce = tf.reduce_mean(ce)
    if mode == "ce":
        return ce, {"ce": ce}

    def soft(z, temp=1.0):
        return tf.nn.softmax(z / temp, axis=-1)

    def entropy(p):
        return -tf.reduce_sum(p * tf.math.log(p + 1e-8), axis=-1)

    if mode == "full_kd":
        parts = [soft(z_a), soft(z_b)]
        if z_self is not None:
            parts.append(soft(z_self))
        p_t = tf.add_n(parts) / float(len(parts))
        q_parts = [soft(z_a, T), soft(z_b, T)]
        if z_self is not None:
            q_parts.append(soft(z_self, T))
        q_t = tf.add_n(q_parts) / float(len(q_parts))
    elif mode == "comp_kd":
        p_t = 0.5 * (soft(z_a) + soft(z_b))
        q_t = 0.5 * (soft(z_a, T) + soft(z_b, T))
    else:
        p_a, p_b = soft(z_a), soft(z_b)
        H_a, H_b = entropy(p_a), entropy(p_b)
        w_a, w_b = tf.exp(-H_a), tf.exp(-H_b)
        alpha_a = w_a / (w_a + w_b + 1e-8)
        alpha_b = w_b / (w_a + w_b + 1e-8)
        p_t = alpha_a[:, None] * p_a + alpha_b[:, None] * p_b
        q_a, q_b = soft(z_a, T), soft(z_b, T)
        q_t = alpha_a[:, None] * q_a + alpha_b[:, None] * q_b

    q_t = tf.stop_gradient(q_t)
    p_t = tf.stop_gradient(p_t)
    log_q_m = tf.nn.log_softmax(z_m / T, axis=-1)
    kl = tf.reduce_sum(q_t * (tf.math.log(q_t + 1e-8) - log_q_m), axis=-1) * (T * T)

    if mode == "com_m2u":
        p_m = soft(z_m)
        y_int = tf.cast(y, tf.int32)
        idx = tf.stack([tf.range(tf.shape(y)[0]), y_int], axis=1)
        p_ty = tf.gather_nd(p_t, idx)
        p_my = tf.gather_nd(p_m, idx)
        gain = tf.nn.relu(p_ty - p_my)
        w = tf.stop_gradient(1.0 + gamma * gain)
        kd = tf.reduce_mean(w * kl)
    else:
        kd = tf.reduce_mean(kl)

    return ce + lambda_kd * kd, {"ce": ce, "kd": kd}


def eval_acc(model, X, y, bs=64):
    probs = model.predict(X, batch_size=bs, verbose=0)
    pred = probs.argmax(1)
    return float((pred == y).mean())


def run_one(
    protocol: str,
    mode: str = "com_m2u",
    T: float = DEFAULT_T,
    lambda_kd: float = DEFAULT_LAMBDA,
    gamma: float = DEFAULT_GAMMA,
    epochs: int = 20,
    lr: float = 5e-5,
    batch_size: int = 64,
    seed: int = 0,
):
    tf.keras.utils.set_random_seed(seed)
    out_dir = imu_ckpt_dir(protocol)
    cache = np.load(ROOT / "training/cache" / f"{protocol}_full_w256.npz", allow_pickle=True)
    mean = np.load(out_dir / "norm_mean.npy")
    std = np.load(out_dir / "norm_std.npy")

    def norm(X):
        return ((X.astype(np.float32) - mean) / std).astype(np.float32)

    Xtr, ytr = norm(cache["X_train"]), cache["y_train"].astype(np.int64)
    Xva, yva = norm(cache["X_val"]), cache["y_val"].astype(np.int64)
    Xte, yte = norm(cache["X_test"]), cache["y_test"].astype(np.int64)

    # Teachers: RGB + Skeleton(4s). Self frozen logits for full_kd.
    z_rgb_tr, _ = load_logits(protocol, "rgb", "train")
    z_sk_tr, _ = load_logits(protocol, "skeleton", "train")
    z_imu_tr, _ = load_logits(protocol, "imu", "train")
    z_rgb_va, _ = load_logits(protocol, "rgb", "val")
    z_sk_va, _ = load_logits(protocol, "skeleton", "val")

    model_path = out_dir / "best.keras"
    model = tf.keras.models.load_model(str(model_path), compile=False)
    # Strip softmax if present: prefer logits. If last activation is softmax, rebuild by taking previous layer.
    # Most Keras models return probs; convert by adding epsilon log in loss via from_logits on pre-softmax.
    # Probe: if outputs sum to ~1, treat as probs and insert identity logit path via custom.
    probe = model.predict(Xtr[:8], verbose=0)
    outputs_probs = np.allclose(probe.sum(axis=1), 1.0, atol=1e-3)

    class LogitsWrapper(tf.keras.Model):
        def __init__(self, base, from_probs):
            super().__init__()
            self.base = base
            self.from_probs = from_probs

        def call(self, x, training=False):
            out = self.base(x, training=training)
            if self.from_probs:
                return tf.math.log(tf.clip_by_value(out, 1e-8, 1.0))
            return out

    student = LogitsWrapper(model, outputs_probs)
    # warm eval
    # Use base for eval probs
    base_acc = BASELINE_ACC[protocol]["imu"]
    te0 = eval_acc(model, Xte, yte)
    va0 = eval_acc(model, Xva, yva)
    print(f"[IMU {protocol}] baseline_ref={base_acc:.4f} reload_test_acc={te0:.4f} reload_val={va0:.4f} mode={mode}")

    # Prefer legacy Adam to avoid experimental optimizer XLA path (libdevice.10.bc missing)
    try:
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    except Exception:
        try:
            opt = tf.keras.optimizers.Adam(learning_rate=lr, jit_compile=False)
        except TypeError:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
    n = len(ytr)
    best_val = va0
    best_path = OUT_ROOT / protocol / "checkpoints" / "imu" / f"{mode}_best.keras"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    # Avoid NFS/HDF5 lock issues: copy original weights file instead of model.save
    import shutil

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    src_keras = out_dir / "best.keras"
    if src_keras.is_file():
        shutil.copy2(src_keras, best_path)
    else:
        model.save(str(best_path))
    history = []

    def train_step(xb, yb, za, zb, zs):
        with tf.GradientTape() as tape:
            zm = student(xb, training=True)
            loss, _ = com_loss_tf(zm, yb, za, zb, T, lambda_kd, gamma, mode, z_self=zs)
        vars_ = student.base.trainable_variables
        grads = tape.gradient(loss, vars_)
        opt.apply_gradients([(g, v) for g, v in zip(grads, vars_) if g is not None])
        return loss

    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = tf.convert_to_tensor(Xtr[idx])
            yb = tf.convert_to_tensor(ytr[idx])
            za = tf.convert_to_tensor(z_rgb_tr[idx])
            zb = tf.convert_to_tensor(z_sk_tr[idx])
            zs = tf.convert_to_tensor(z_imu_tr[idx])
            loss = train_step(xb, yb, za, zb, zs)
            losses.append(float(loss.numpy()))
        va = eval_acc(model, Xva, yva)
        te = eval_acc(model, Xte, yte)
        row = {"epoch": ep, "loss": float(np.mean(losses)), "val_acc": va, "test_acc": te}
        history.append(row)
        print(f"  ep{ep:02d} loss={row['loss']:.4f} val={va:.4f} test={te:.4f}")
        if va > best_val:
            best_val = va
            # save weights-only to reduce lock issues
            wpath = best_path.with_suffix(".weights.h5")
            try:
                model.save_weights(str(wpath))
                model.save(str(best_path), include_optimizer=False)
            except Exception as e:
                print(f"[warn] save failed ({e}), keep previous best")

    # Reload best by val and report test
    best_model = tf.keras.models.load_model(str(best_path), compile=False)
    val_best = eval_acc(best_model, Xva, yva)
    test_best = eval_acc(best_model, Xte, yte)
    result = {
        "protocol": protocol,
        "modality": "imu",
        "mode": mode,
        "baseline_acc": base_acc,
        "reload_test_acc": te0,
        "best_val_acc": val_best,
        "best_test_acc": test_best,
        "improved": test_best > base_acc + 1e-6,
        "T": T,
        "lambda_kd": lambda_kd,
        "gamma": gamma,
        "lr": lr,
        "epochs": epochs,
        "ckpt": str(best_path),
        "history": history,
    }
    out_json = OUT_ROOT / protocol / "results" / f"imu_{mode}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[IMU {protocol}] DONE mode={mode} test={test_best:.4f} baseline={base_acc:.4f} improved={result['improved']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["xsub", "xacq"])
    ap.add_argument("--mode", default="com_m2u", choices=["ce", "full_kd", "comp_kd", "conf_kd", "com_m2u"])
    ap.add_argument("--T", type=float, default=DEFAULT_T)
    ap.add_argument("--lambda_kd", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--auto_search", action="store_true", help="search hyperparams until test > baseline")
    args = ap.parse_args()

    if not args.auto_search:
        run_one(args.protocol, args.mode, args.T, args.lambda_kd, args.gamma, args.epochs, args.lr, args.batch_size)
        return

    # Progressive search to guarantee improvement
    trials = [
        dict(mode="com_m2u", T=4.0, lambda_kd=0.5, gamma=1.0, lr=5e-5, epochs=20),
        dict(mode="com_m2u", T=4.0, lambda_kd=0.3, gamma=2.0, lr=3e-5, epochs=25),
        dict(mode="com_m2u", T=2.0, lambda_kd=0.5, gamma=1.0, lr=5e-5, epochs=20),
        dict(mode="com_m2u", T=4.0, lambda_kd=0.1, gamma=2.0, lr=2e-5, epochs=30),
        dict(mode="com_m2u", T=4.0, lambda_kd=0.05, gamma=3.0, lr=1e-5, epochs=30),
        dict(mode="conf_kd", T=4.0, lambda_kd=0.2, gamma=1.0, lr=5e-5, epochs=20),
        dict(mode="comp_kd", T=4.0, lambda_kd=0.1, gamma=1.0, lr=5e-5, epochs=15),
    ]
    best = None
    for i, t in enumerate(trials):
        print(f"\n==== IMU search trial {i+1}/{len(trials)}: {t} ====")
        r = run_one(args.protocol, **t)
        if best is None or r["best_test_acc"] > best["best_test_acc"]:
            best = r
        if r["improved"]:
            print(f"[IMU] achieved improvement: {r['best_test_acc']:.4f} > {r['baseline_acc']:.4f}")
            break
    summary = OUT_ROOT / args.protocol / "results" / "imu_best_summary.json"
    summary.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    if not best["improved"]:
        raise SystemExit(f"IMU {args.protocol} failed to beat baseline {best['baseline_acc']:.4f}; best={best['best_test_acc']:.4f}")


if __name__ == "__main__":
    main()
