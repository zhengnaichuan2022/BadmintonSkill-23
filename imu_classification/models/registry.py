""" HAR （ training/models/）。"""
from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Dict, List, Optional

from . import (
    bilstm,
    cnn,
    cnn_lstm,
    deep_conv_lstm,
    icg_net,
    ispl_inception,
    mch_cnn_gru,
    mobile_hart_xs,
    residual_bilstm,
    resnet_se,
    stacked_lstm,
    tsf,
    vanilla_lstm,
)


@dataclass
class ModelSpec:
    name: str
    module: ModuleType
    framework: str
    description: str


MODELS: Dict[str, ModelSpec] = {
    "deep_conv_lstm": ModelSpec(
        "deep_conv_lstm", deep_conv_lstm, "tensorflow", "DeepConvLSTM baseline"
    ),
    "ispl_inception": ModelSpec(
        "ispl_inception", ispl_inception, "tensorflow", "iSPL Inception (TimeSeries)"
    ),
    "cnn_lstm": ModelSpec(
        "cnn_lstm", cnn_lstm, "tensorflow", "CNN-LSTM (frame-wise CNN + LSTM)"
    ),
    "cnn": ModelSpec("cnn", cnn, "tensorflow", "1D-CNN + FC"),
    "vanilla_lstm": ModelSpec(
        "vanilla_lstm", vanilla_lstm, "tensorflow", "Single-layer LSTM"
    ),
    "bilstm": ModelSpec("bilstm", bilstm, "tensorflow", "Bidirectional LSTM"),
    "stacked_lstm": ModelSpec(
        "stacked_lstm", stacked_lstm, "tensorflow", "Stacked LSTM"
    ),
    "residual_bilstm": ModelSpec(
        "residual_bilstm", residual_bilstm, "tensorflow", "ResNet + BiLSTM"
    ),
    "resnet_se": ModelSpec(
        "resnet_se", resnet_se, "tensorflow", "ResNet-SE (channel attention)"
    ),
    "mch_cnn_gru": ModelSpec(
        "mch_cnn_gru", mch_cnn_gru, "tensorflow", "Multichannel CNN-GRU"
    ),
    "icg_net": ModelSpec(
        "icg_net", icg_net, "tensorflow", "Inception CNN-GRU hybrid"
    ),
    "mobile_hart_xs": ModelSpec(
        "mobile_hart_xs",
        mobile_hart_xs,
        "tensorflow",
        "MobileHART-XS (acc/gyro dual-stream)",
    ),
    "r_tsf_net": ModelSpec(
        "r_tsf_net", tsf, "tensorflow", "rTsfNet (multi-head 3D rotation + TSF)"
    ),
}


def list_models(framework: Optional[str] = None) -> List[str]:
    if framework is None:
        return list(MODELS.keys())
    return [n for n, s in MODELS.items() if s.framework == framework]


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODELS:
        avail = ", ".join(MODELS.keys())
        raise KeyError(f"Unknown model '{name}'. Available: {avail}")
    return MODELS[name]


def load_model_module(name: str) -> ModuleType:
    return get_model_spec(name).module
