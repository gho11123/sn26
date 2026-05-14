#!/usr/bin/env python
"""Replay the validator's verify_and_score on a saved miner task.

Usage:
    python scripts/score_task.py <task_id_or_block>
    python scripts/score_task.py <task_id_or_block> --response-time-ms 500
    python scripts/score_task.py <task_id_or_block> --tasks-dir /path/to/tasks

The first argument can be either:
  - a task_id ("8182594-16998081064234601235"), resolved by scanning meta.json files
  - a block / directory name ("8182594") under the tasks dir

Scoring constants come from perturbnet.constants (env overridable, matches validator).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import perturbnet.constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import (
    load_efficientnet_v2_m,
    normalize_prediction_label,
    predict_label,
)


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    if x_clean.ndim != 3 or x_adv.ndim != 3 or x_clean.shape != x_adv.shape:
        return 0.0
    padding = kernel_size // 2
    x = x_clean.unsqueeze(0)
    y = x_adv.unsqueeze(0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / (denominator + 1e-12)).mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _resolve_task_dir(arg: str, tasks_root: Path) -> Path:
    direct = tasks_root / arg
    if direct.is_dir():
        return direct
    for meta_path in tasks_root.glob("*/meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if str(meta.get("task_id", "")) == arg:
            return meta_path.parent
    raise SystemExit(
        f"Could not resolve '{arg}' as a directory or a saved task_id under {tasks_root}"
    )


def _png_bytes_to_b64(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _hr(title: str) -> str:
    return f"\n{title}\n" + "-" * len(title)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", help="task_id or directory/block name under --tasks-dir")
    parser.add_argument(
        "--tasks-dir",
        default=os.getenv("PERTURB_MINER_SAVE_DIR", str(ROOT / "tasks")),
        help="Root directory where the miner saves tasks (default: ./tasks).",
    )
    parser.add_argument(
        "--response-time-ms",
        type=int,
        default=None,
        help="Override response time in ms. Defaults to meta.json's response_time_ms (or 0).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the EfficientNet inference and SSIM compute.",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_dir).resolve()
    task_dir = _resolve_task_dir(args.task, tasks_root)
    meta_path = task_dir / "meta.json"
    clean_path = task_dir / "clean.png"
    perturbed_path = task_dir / "perturbed.png"
    for required in (meta_path, clean_path, perturbed_path):
        if not required.exists():
            raise SystemExit(f"Missing required file: {required}")

    meta = json.loads(meta_path.read_text())
    epsilon = float(meta["epsilon"])
    norm_type = str(meta.get("norm_type", "Linf"))
    timeout_seconds = float(meta.get("timeout_seconds", 30))
    true_label = str(meta.get("true_label", ""))
    response_time_ms = args.response_time_ms
    if response_time_ms is None:
        response_time_ms = int(meta.get("response_time_ms", 0))

    device = torch.device(args.device)
    model = load_efficientnet_v2_m(device)

    x_clean = decode_image_b64(_png_bytes_to_b64(clean_path)).to(device)
    x_adv = decode_image_b64(_png_bytes_to_b64(perturbed_path)).to(device)

    print(_hr("Task"))
    print(f"  dir                : {task_dir}")
    print(f"  task_id            : {meta.get('task_id')}")
    print(f"  block              : {meta.get('block')}")
    print(f"  prompt             : {meta.get('prompt')}")
    print(f"  true_label         : {true_label}")
    print(f"  epsilon            : {epsilon}")
    print(f"  norm_type          : {norm_type}")
    print(f"  timeout_seconds    : {timeout_seconds}")
    print(f"  miner_status       : {meta.get('status')}")
    print(f"  miner_final_pred   : {meta.get('final_pred')}  (target_index={meta.get('target_index')})")
    print(f"  caller_hotkey      : {meta.get('caller_hotkey')}")

    print(_hr("Validator scoring constants"))
    print(f"  min_linf_delta            : {C.MIN_LINF_DELTA}")
    print(f"  max_linf_delta            : {C.MAX_LINF_DELTA}")
    print(f"  min_ssim                  : {C.MIN_SSIM}")
    print(f"  min_psnr_db               : {C.MIN_PSNR_DB}")
    print(f"  linf_component_weight     : {C.LINF_COMPONENT_WEIGHT}")
    print(f"  rmse_component_weight     : {C.RMSE_COMPONENT_WEIGHT}")
    print(f"  PERTURBATION_WEIGHT       : {C.PERTURBATION_WEIGHT}")
    print(f"  SPEED_WEIGHT              : {C.SPEED_WEIGHT}")

    print(_hr("Image checks"))
    print(f"  clean shape        : {tuple(x_clean.shape)}")
    print(f"  adv   shape        : {tuple(x_adv.shape)}")
    print(f"  adv   min/max      : {x_adv.min().item():.6f} / {x_adv.max().item():.6f}")
    if x_adv.shape != x_clean.shape:
        print("\nFINAL_SCORE: 0.0  (shape_mismatch)")
        return 0
    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        print("\nFINAL_SCORE: 0.0  (value_out_of_range)")
        return 0

    diff = x_adv - x_clean
    if norm_type == "Linf":
        norm = float(diff.abs().max().item())
    elif norm_type == "L2":
        norm = float(diff.norm(2).item())
    else:
        norm = float(diff.ne(0).sum().item())
    rmse = float(torch.sqrt(torch.mean(diff ** 2)).item())
    ssim = _compute_ssim(x_clean=x_clean, x_adv=x_adv)
    psnr_db = _compute_psnr_db(x_clean=x_clean, x_adv=x_adv)
    prediction = predict_label(model, x_adv)
    normalized_prediction = normalize_prediction_label(prediction)

    effective_max_delta = min(epsilon, float(C.MAX_LINF_DELTA))
    print(_hr("Distortion metrics"))
    print(f"  norm ({norm_type})         : {norm:.6f}")
    print(f"  rmse                : {rmse:.6f}")
    print(f"  ssim                : {ssim:.6f}")
    print(f"  psnr_db             : {psnr_db:.4f}")
    print(f"  effective_max_delta = min(epsilon, max_linf_delta) = {effective_max_delta:.6f}")

    print(_hr("Hard gates"))
    gates = []
    gates.append(("norm >= min_linf_delta", norm >= float(C.MIN_LINF_DELTA), f"{norm:.6f} >= {C.MIN_LINF_DELTA}"))
    gates.append(("norm <= effective_max_delta", norm <= effective_max_delta, f"{norm:.6f} <= {effective_max_delta:.6f}"))
    gates.append(("prediction != true_label", normalized_prediction != true_label, f"adv_pred='{normalized_prediction}' vs true='{true_label}'"))
    gates.append(("ssim >= min_ssim", ssim >= float(C.MIN_SSIM), f"{ssim:.6f} >= {C.MIN_SSIM}"))
    psnr_required = float(C.MIN_PSNR_DB) > 0.0
    gates.append((
        "psnr_db >= min_psnr_db" if psnr_required else "psnr_db (disabled)",
        (psnr_db >= float(C.MIN_PSNR_DB)) if psnr_required else True,
        f"{psnr_db:.4f} >= {C.MIN_PSNR_DB}" if psnr_required else "min_psnr_db=0 → gate disabled",
    ))
    failed = []
    for name, ok, detail in gates:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name:<28}  {detail}")
        if not ok:
            failed.append(name)

    fail_reason_map = {
        "norm >= min_linf_delta": "below_min_delta",
        "norm <= effective_max_delta": "above_max_delta",
        "prediction != true_label": "label_match_with_original",
        "ssim >= min_ssim": "below_min_ssim",
        "psnr_db >= min_psnr_db": "below_min_psnr_db",
    }

    print(_hr("Score components"))
    denom_linf = max(1e-12, effective_max_delta - float(C.MIN_LINF_DELTA))
    linf_ratio = min(max((norm - float(C.MIN_LINF_DELTA)) / denom_linf, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_ratio = min(max(rmse / max(1e-12, effective_max_delta), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2
    total_w = max(1e-12, float(C.LINF_COMPONENT_WEIGHT) + float(C.RMSE_COMPONENT_WEIGHT))
    perturbation_score = (
        float(C.LINF_COMPONENT_WEIGHT) * linf_score + float(C.RMSE_COMPONENT_WEIGHT) * rmse_score
    ) / total_w
    time_ratio = response_time_ms / max(1e-12, (timeout_seconds * 1000.0))
    speed_score = 1.0 - min(time_ratio, 1.0)
    print(f"  linf_ratio          : {linf_ratio:.6f}")
    print(f"  linf_score          : (1-linf_ratio)^2 = {linf_score:.6f}")
    print(f"  rmse_ratio          : {rmse_ratio:.6f}")
    print(f"  rmse_score          : (1-rmse_ratio)^2 = {rmse_score:.6f}")
    print(
        f"  perturbation_score  : ({C.LINF_COMPONENT_WEIGHT}*linf + {C.RMSE_COMPONENT_WEIGHT}*rmse)/sum_w "
        f"= {perturbation_score:.6f}"
    )
    print(f"  response_time_ms    : {response_time_ms}  (timeout {int(timeout_seconds)}s)")
    print(f"  speed_score         : 1 - min(t/timeout,1) = {speed_score:.6f}")

    print(_hr("Final"))
    if failed:
        primary = failed[0]
        reason = fail_reason_map.get(primary, "gate_failed")
        print(f"  reason              : {reason}  (failed_gate='{primary}')")
        print(f"  FINAL_SCORE         : 0.0")
        print(
            "\n  Note: score is 0 because a hard gate failed; the perturbation/speed numbers"
            "\n  above are shown for diagnostic purposes only."
        )
        return 0

    final = float(C.PERTURBATION_WEIGHT) * perturbation_score + float(C.SPEED_WEIGHT) * speed_score
    print(
        f"  formula             : {C.PERTURBATION_WEIGHT}*perturbation_score + "
        f"{C.SPEED_WEIGHT}*speed_score"
    )
    print(f"  FINAL_SCORE         : {final:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
