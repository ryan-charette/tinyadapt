from __future__ import annotations

import time

import streamlit as st
import torch

from tinyadapt.models import MODEL_NAMES, create_model
from tinyadapt.models.bignet import BIGNET_DIM
from tinyadapt.utils.checkpoints import find_default_checkpoint
from tinyadapt.utils.stats import format_bytes, profile_model, storage_bytes


@st.cache_resource
def load_model(name: str, rank: int, group_size: int):
    checkpoint = find_default_checkpoint()
    return create_model(name, checkpoint, lora_rank=rank, group_size=group_size).eval()


st.set_page_config(page_title="TinyAdapt", layout="wide")
st.title("TinyAdapt")

with st.sidebar:
    variant = st.selectbox("Model", MODEL_NAMES)
    batch_size = st.slider("Batch size", 1, 64, 8)
    rank = st.slider("LoRA rank", 1, 32, 8)
    group_size = st.select_slider("4-bit group size", options=[8, 16, 32, 64], value=16)
    run = st.button("Run inference")

if run:
    baseline = load_model("fp32", rank, group_size)
    model = load_model(variant, rank, group_size)
    x = torch.randn(batch_size, BIGNET_DIM)
    with torch.no_grad():
        start = time.perf_counter()
        output = model(x)
        latency_ms = (time.perf_counter() - start) * 1000.0
        reference = baseline(x)
        drift = (output - reference).abs()

    baseline_bytes = storage_bytes(baseline)
    profile = profile_model(variant, model, fp32_storage_bytes=baseline_bytes)
    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{latency_ms:.2f} ms")
    col2.metric("Mean drift", f"{float(drift.mean().item()):.5f}")
    col3.metric("Storage", format_bytes(profile.storage_bytes), f"{profile.reduction_vs_fp32:.1f}% vs FP32")
    st.dataframe(
        {
            "metric": ["total parameters", "trainable parameters", "max drift"],
            "value": [
                f"{profile.total_parameters:,}",
                f"{profile.trainable_parameters:,}",
                f"{float(drift.max().item()):.5f}",
            ],
        },
        use_container_width=True,
    )
else:
    st.info("Choose a model and run inference.")
