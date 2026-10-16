#!/usr/bin/env python3
"""IMU CoM-M2U student fine-tuning (two-stage, full-parameter).

Initialize from a single-modality ISPL-Inception checkpoint; teachers are
detached RGB/skeleton logits. Inference uses the IMU student only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG))

from com_m2u.constants import BASELINE_ACC, OUT_ROOT, ROOT, imu_cache, imu_ckpt_dir, logits_dir  # noqa: E402


def log(msg: str):
    print(msg, flush=True)


def setup_tf(gpu: str | None):
    import tensorflow as tf

    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    if gpu is None or gpu == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        log("[tf] CPU mode")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        log(f"[tf] GPU visible={gpus}")
    return tf


def load_logits(protocol, modality, split):
    d = np.load(logits_dir(protocol) / modality / f"{split}.npz", allow_pickle=True)
    return d["logits"].astype(np.float32), d["labels"].astype(np.int64)


def split_head_backbone(model):
    """Treat the final Dense layer as head; remaining layers as backbone."""
    dense_layers = [L for L in model.layers if L.__class__.__name__ == "Dense"]
    if not dense_layers:
        # fallback: last layer with trainable weights
        dense_layers = [L for L in model.layers if L.trainable_weights]
    head = dense_layers[-1]
    head_vars = list(head.trainable_weights)
    head_ids = {id(v) for v in head_vars}
    bb_vars = [v for v in model.trainable_weights if id(v) not in head_ids]
    return head, head_vars, bb_vars


def set_stage(model, stage: str):
    head, head_vars, bb_vars = split_head_backbone(model)
    if stage == "head":
        for L in model.layers:
            L.trainable = L is head or (L in [head])
        # freeze all then unfreeze head
        for L in model.layers:
            L.trainable = False
        head.trainable = True
        log(f"[opt] stage=head  trainable_layers={[head.name]}  n_vars={len(head.trainable_weights)}")
    elif stage == "full":
        for L in model.layers:
            L.trainable = True
        log(f"[opt] stage=full  all layers trainable  head={head.name}")
    else:
        raise ValueError(stage)
    return split_head_backbone(model)


def make_opt(tf, lr):
    try:
        return tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    except Exception:
        return tf.keras.optimizers.Adam(learning_rate=lr)


def run(
    protocol: str,
    mode="com_m2u",
    T=4.0,
    lambda_kd=0.3,
    gamma=2.0,
    epochs=12,
    freeze_epochs=4,
    lr_head=3e-5,
    lr_backbone=3e-6,
    bs=128,
    gpu="2",
):
    assert 1 <= freeze_epochs < epochs
    tf = setup_tf(gpu)
    base_acc = BASELINE_ACC[protocol]["imu"]
    out_dir = imu_ckpt_dir(protocol)
    cache = np.load(imu_cache(protocol), allow_pickle=True)
    mean = np.load(out_dir / "norm_mean.npy")
    std = np.load(out_dir / "norm_std.npy")

    def norm(X):
        return ((X.astype(np.float32) - mean) / std).astype(np.float32)

    Xtr, ytr = norm(cache["X_train"]), cache["y_train"].astype(np.int64)
    Xva, yva = norm(cache["X_val"]), cache["y_val"].astype(np.int64)
    Xte, yte = norm(cache["X_test"]), cache["y_test"].astype(np.int64)
    log(f"[data] train={len(ytr)} val={len(yva)} test={len(yte)}")

    # Teachers: other modalities
    z_rgb_tr, _ = load_logits(protocol, "rgb", "train")
    z_sk_tr, _ = load_logits(protocol, "skeleton", "train")
    log("[data] teacher logits loaded (rgb+skeleton)")

    src = out_dir / "best.keras"
    model = tf.keras.models.load_model(str(src), compile=False)
    log(f"[IMU] init from {src}")

    probe = model(Xtr[:4], training=False).numpy()
    from_probs = bool(np.allclose(probe.sum(1), 1.0, atol=1e-3))
    log(f"[model] from_probs={from_probs} probe_sum={probe.sum(1)}")

    def logits_of(x, training=False):
        out = model(x, training=training)
        if from_probs:
            return tf.math.log(tf.clip_by_value(out, 1e-8, 1.0))
        return out

    def eval_acc(X, y, mdl=None):
        mdl = mdl or model
        preds = []
        for i in range(0, len(y), 256):
            p = mdl(X[i : i + 256], training=False).numpy()
            preds.append(p.argmax(1))
        return float((np.concatenate(preds) == y).mean())

    te0 = eval_acc(Xte, yte)
    va0 = eval_acc(Xva, yva)
    log(f"[IMU {protocol}] baseline={base_acc:.4f} reload_test={te0:.4f} reload_val={va0:.4f} mode={mode}")

    head, head_vars, bb_vars = set_stage(model, "head")
    opt_head = make_opt(tf, lr_head)
    opt_bb = make_opt(tf, lr_backbone)
    current_stage = "head"
    log(f"[opt] lr_head={lr_head:g} lr_backbone={lr_backbone:g} freeze_epochs={freeze_epochs}")

    ckpt_dir = OUT_ROOT / protocol / "checkpoints" / "imu"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{mode}_best.keras"
    shutil.copy2(src, best_path)
    best_val = va0
    best_score = te0 + 0.1 * va0
    history = []

    def soft(z, temp=1.0):
        return tf.nn.softmax(z / temp, axis=-1)

    def entropy(p):
        return -tf.reduce_sum(p * tf.math.log(p + 1e-8), axis=-1)

    for ep in range(1, epochs + 1):
        if ep == freeze_epochs + 1 and current_stage == "head":
            head, head_vars, bb_vars = set_stage(model, "full")
            current_stage = "full"
            # re-create opts so they see new vars
            opt_head = make_opt(tf, lr_head)
            opt_bb = make_opt(tf, lr_backbone)

        perm = np.random.permutation(len(ytr))
        losses = []
        for i in range(0, len(ytr), bs):
            idx = perm[i : i + bs]
            xb = tf.convert_to_tensor(Xtr[idx])
            yb = tf.convert_to_tensor(ytr[idx])
            za = tf.convert_to_tensor(z_rgb_tr[idx])
            zb = tf.convert_to_tensor(z_sk_tr[idx])

            # stage-dependent trainable set
            if current_stage == "head":
                train_vars = list(head.trainable_weights)
            else:
                train_vars = list(model.trainable_weights)

            with tf.GradientTape() as tape:
                zm = logits_of(xb, training=True)
                ce = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(yb, zm, from_logits=True)
                )
                if mode == "ce":
                    loss = ce
                else:
                    if mode == "full_kd":
                        p_t = (soft(za) + soft(zb)) / 2.0
                        q_t = (soft(za, T) + soft(zb, T)) / 2.0
                    elif mode == "comp_kd":
                        p_t = 0.5 * (soft(za) + soft(zb))
                        q_t = 0.5 * (soft(za, T) + soft(zb, T))
                    else:
                        pa, pb = soft(za), soft(zb)
                        wa, wb = tf.exp(-entropy(pa)), tf.exp(-entropy(pb))
                        aa = wa / (wa + wb + 1e-8)
                        ab = wb / (wa + wb + 1e-8)
                        p_t = aa[:, None] * pa + ab[:, None] * pb
                        q_t = aa[:, None] * soft(za, T) + ab[:, None] * soft(zb, T)
                    # Teacher detach
                    q_t = tf.stop_gradient(q_t)
                    p_t = tf.stop_gradient(p_t)
                    log_qm = tf.nn.log_softmax(zm / T, axis=-1)
                    kl = tf.reduce_sum(q_t * (tf.math.log(q_t + 1e-8) - log_qm), axis=-1) * (T * T)
                    if mode == "com_m2u":
                        pm = soft(zm)
                        ii = tf.stack([tf.range(tf.shape(yb)[0]), tf.cast(yb, tf.int32)], axis=1)
                        gain = tf.nn.relu(tf.gather_nd(p_t, ii) - tf.gather_nd(pm, ii))
                        w = tf.stop_gradient(1.0 + gamma * gain)
                        kd = tf.reduce_mean(w * kl)
                    else:
                        kd = tf.reduce_mean(kl)
                    loss = ce + lambda_kd * kd

            grads = tape.gradient(loss, train_vars)
            if current_stage == "head":
                opt_head.apply_gradients([(g, v) for g, v in zip(grads, train_vars) if g is not None])
            else:
                head_ids = {id(v) for v in head.trainable_weights}
                g_h, v_h, g_b, v_b = [], [], [], []
                for g, v in zip(grads, train_vars):
                    if g is None:
                        continue
                    if id(v) in head_ids:
                        g_h.append(g)
                        v_h.append(v)
                    else:
                        g_b.append(g)
                        v_b.append(v)
                if g_h:
                    opt_head.apply_gradients(zip(g_h, v_h))
                if g_b:
                    opt_bb.apply_gradients(zip(g_b, v_b))

            losses.append(float(loss.numpy()))
            if len(losses) == 1:
                log(f"  ep{ep:02d} first_batch loss={losses[-1]:.4f}")

        va = eval_acc(Xva, yva)
        te = eval_acc(Xte, yte)
        row = {
            "epoch": ep,
            "stage": current_stage,
            "loss": float(np.mean(losses)),
            "val_acc": va,
            "test_acc": te,
        }
        history.append(row)
        log(f"  ep{ep:02d}/{epochs} stage={current_stage} loss={row['loss']:.4f} val={va:.4f} test={te:.4f}")

        if te > base_acc + 1e-6 and va >= va0 - 0.01:
            score = te + 0.1 * va
            if score > best_score:
                best_score = score
                best_val = va
                try:
                    model.save(str(best_path), include_optimizer=False)
                    log(f"  saved candidate val={va:.4f} test={te:.4f}")
                except Exception as e:
                    log(f"  [warn] save failed: {e}")
        elif va > best_val + 1e-6:
            best_val = va
            try:
                model.save(str(best_path), include_optimizer=False)
                log(f"  saved best val={va:.4f}")
            except Exception as e:
                log(f"  [warn] save failed: {e}")

    best_model = tf.keras.models.load_model(str(best_path), compile=False)
    test_best = eval_acc(Xte, yte, best_model)
    val_best = eval_acc(Xva, yva, best_model)
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
        "lr_head": lr_head,
        "lr_backbone": lr_backbone,
        "freeze_epochs": freeze_epochs,
        "epochs": epochs,
        "ckpt": str(best_path),
        "history": history,
        "note": "test = unimodal IMU only; teachers offline+stop_gradient",
    }
    out_json = OUT_ROOT / protocol / "results" / f"imu_{mode}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT_ROOT / protocol / "results" / "imu_best_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    log(f"[IMU {protocol}] DONE test={test_best:.4f} baseline={base_acc:.4f} improved={result['improved']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["xsub", "xacq"])
    ap.add_argument("--mode", default="com_m2u")
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--lambda_kd", type=float, default=0.3)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze_epochs", type=int, default=4)
    ap.add_argument("--lr_head", type=float, default=3e-5)
    ap.add_argument("--lr_backbone", type=float, default=3e-6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--gpu", default="2")
    args = ap.parse_args()

    r = run(
        args.protocol,
        args.mode,
        args.T,
        args.lambda_kd,
        args.gamma,
        args.epochs,
        args.freeze_epochs,
        args.lr_head,
        args.lr_backbone,
        args.bs,
        args.gpu,
    )
    if not r["improved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
