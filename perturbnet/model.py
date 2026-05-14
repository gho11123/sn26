from __future__ import annotations

import os

import torch
from torchvision.models import EfficientNet_V2_M_Weights, efficientnet_v2_m

# cuDNN auto-tunes the fastest convolution algorithm for each input shape.
# Safe because our input shape is stable (always one 480x480 image after preprocess).
torch.backends.cudnn.benchmark = True

WEIGHTS = EfficientNet_V2_M_Weights.IMAGENET1K_V1
LABELS = [label.lower() for label in WEIGHTS.meta.get("categories", [])]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
PREPROCESS = WEIGHTS.transforms()

# Toggle bfloat16 autocast on CUDA forward passes (~2-3x faster, ~zero accuracy
# loss for adversarial pixel selection at L∞=1/255). Disable via env var if a
# downstream caller needs strictly fp32 numerics (e.g. matching a fp32-only
# validator's argmax at the decision boundary).
_BF16_ENABLED = os.getenv("PERTURB_BF16", "1") not in {"0", "false", "False"}

# torch.compile is opt-in (default off): on small GPUs like the A4000 the
# inductor backend can hang for minutes on first call and the eventual speedup
# is small. Enable via env var if you've confirmed it works on your hardware.
_COMPILE_ENABLED = os.getenv("PERTURB_TORCH_COMPILE", "0") not in {"0", "false", "False"}


def load_efficientnet_v2_m(device: torch.device) -> torch.nn.Module:
    try:
        model = efficientnet_v2_m(weights=WEIGHTS)
    except Exception:
        # Keep model family stable even if pretrained weights are unavailable.
        model = efficientnet_v2_m(weights=None)
    model = model.to(device).eval()
    if _COMPILE_ENABLED and device.type == "cuda" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
        except Exception:
            pass  # fall back to uncompiled model
    return model


def resolve_target_index(target_label: str) -> int | None:
    return LABEL_TO_INDEX.get(target_label.strip().lower())


def normalize_prediction_label(raw_label: str) -> str:
    return raw_label.strip().lower().replace("_", " ")


def _preprocess_for_efficientnet_v2_m(image_bchw: torch.Tensor) -> torch.Tensor:
    # Use torchvision's canonical transform pipeline for this exact weights variant.
    return PREPROCESS(image_bchw)


def _use_autocast(image: torch.Tensor) -> bool:
    return _BF16_ENABLED and image.is_cuda


def predict_index(model: torch.nn.Module, image_chw: torch.Tensor) -> int:
    with torch.no_grad():
        if _use_autocast(image_chw):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(_preprocess_for_efficientnet_v2_m(image_chw.unsqueeze(0)))
        else:
            logits = model(_preprocess_for_efficientnet_v2_m(image_chw.unsqueeze(0)))
        return int(logits.argmax(dim=1).item())


def logits_for_images(model: torch.nn.Module, image_bchw: torch.Tensor) -> torch.Tensor:
    if _use_autocast(image_bchw):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(_preprocess_for_efficientnet_v2_m(image_bchw))
    return model(_preprocess_for_efficientnet_v2_m(image_bchw))


def predict_label(model: torch.nn.Module, image_chw: torch.Tensor) -> str:
    idx = predict_index(model=model, image_chw=image_chw)
    if 0 <= idx < len(LABELS):
        return LABELS[idx]
    return str(idx)

