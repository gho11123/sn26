"""Sparse, uint8-aligned, runner-up-targeted adversarial attack for EfficientNet-V2-M.

The strategy:
  1. Predict argmax on the clean image; identify the runner-up class.
  2. Each iteration, compute gradient of (logit[true_label] - logit[runner_up])
     w.r.t. the input. Select the top-K pixel-channels by |gradient|. Step each
     by -sign(gradient) * (1/255) so the gap shrinks.
  3. Snap the image to the 8-bit grid each iteration -- this is what survives
     the PNG round-trip the validator does. Without snapping the float
     perturbation drifts when re-decoded.
  4. Early-exit as soon as argmax flips.
  5. Grow K (double it) every `iters_per_k` iterations if not yet flipped.

The mask drops pixels outside the centre min(H,W) square: the model's preprocess
resizes shortest-side to 480 then centre-crops to 480x480, so pixels outside
that square contribute essentially nothing to the prediction.
"""
from __future__ import annotations

import typing

import torch
import torch.nn.functional as F

from perturbnet.model import LABELS, logits_for_images, predict_index


def _label(idx: int) -> str:
    return LABELS[idx] if 0 <= idx < len(LABELS) else str(idx)


QUANTUM = 1.0 / 255.0


def _quantise(t: torch.Tensor) -> torch.Tensor:
    return (t.clamp(0.0, 1.0) * 255.0).round() / 255.0


def centre_crop_mask(shape: typing.Sequence[int]) -> torch.Tensor:
    if len(shape) != 3:
        raise ValueError(f"expected (C, H, W), got {tuple(shape)}")
    c, h, w = shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    mask = torch.zeros((c, h, w))
    mask[:, y0 : y0 + side, x0 : x0 + side] = 1.0
    return mask


def sparse_runnerup_attack(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    true_label_idx: int,
    *,
    quantum: float = QUANTUM,
    initial_k: int = 256,
    max_k: int = 16384,
    iters_per_k: int = 2,
    max_total_iters: int = 15,
    max_linf_quanta: int = 1,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    device = x_clean.device
    clean = _quantise(x_clean.detach())
    adv = clean.clone()
    mask = centre_crop_mask(clean.shape).to(device)

    with torch.no_grad():
        clean_logits = logits_for_images(model, clean.unsqueeze(0))[0]
    top2 = clean_logits.topk(2)
    top1_idx = int(top2.indices[0].item())
    runner_up_idx = int(top2.indices[1].item())
    initial_gap = float((top2.values[0] - top2.values[1]).item())

    if verbose:
        print(
            f"  sparse: clean top1={top1_idx} ({_label(top1_idx)}) "
            f"runner_up={runner_up_idx} ({_label(runner_up_idx)}) gap={initial_gap:+.4f}"
        )

    stats: dict[str, typing.Any] = {
        "clean_top1_idx": top1_idx,
        "runner_up_idx": runner_up_idx,
        "initial_gap": initial_gap,
        "iterations": 0,
        "k_history": [],
        "max_quanta_used": 0,
        "perturbed_pixel_channels": 0,
        "status": "init",
    }

    # If the model already disagrees with the validator's claimed true_label,
    # there is nothing useful to do: returning the clean image will trip the
    # label-match gate anyway. Caller should handle this case.
    if top1_idx != true_label_idx:
        stats["status"] = "clean_disagrees_with_true_label"
        stats["final_pred"] = top1_idx
        stats["final_linf"] = 0.0
        stats["final_rmse"] = 0.0
        return adv, stats

    k = initial_k
    iter_count = 0
    max_q = 0

    while iter_count < max_total_iters:
        with torch.no_grad():
            cur_logits = logits_for_images(model, adv.unsqueeze(0))[0]
        if int(cur_logits.argmax().item()) != true_label_idx:
            stats["status"] = "flipped"
            break

        adv_var = adv.detach().clone().requires_grad_(True)
        logits = logits_for_images(model, adv_var.unsqueeze(0))[0]
        gap = logits[true_label_idx] - logits[runner_up_idx]
        grad = torch.autograd.grad(gap, adv_var)[0]

        step_sign = -grad.sign()
        delta = adv - clean
        new_delta = (delta + step_sign * quantum).clamp(
            -max_linf_quanta * quantum, max_linf_quanta * quantum
        )
        actual_step = new_delta - delta
        movable = (actual_step.abs() > 0) & (mask > 0)

        score = grad.abs() * movable.to(grad.dtype)
        flat = score.flatten()
        num_movable = int(movable.sum().item())
        k_used = min(k, num_movable)
        if k_used == 0:
            stats["status"] = "no_movable_pixels"
            break

        topk = torch.topk(flat, k_used)
        sel = torch.zeros_like(flat, dtype=torch.bool)
        sel[topk.indices] = True
        sel = sel.view_as(grad)

        adv = adv + actual_step * sel.to(adv.dtype)
        adv = _quantise(adv)

        iter_count += 1
        stats["k_history"].append(k_used)
        max_q = max(max_q, int(((adv - clean).abs().max().item()) * 255 + 0.5))
        stats["max_quanta_used"] = max_q

        if verbose:
            with torch.no_grad():
                post_logits = logits_for_images(model, adv.unsqueeze(0))[0]
            post_pred = int(post_logits.argmax().item())
            post_gap = float((post_logits[true_label_idx] - post_logits[runner_up_idx]).item())
            flip = " FLIPPED" if post_pred != true_label_idx else ""
            print(
                f"  sparse #{iter_count:<2} K={k_used:<5} "
                f"pred={post_pred:>3} ({_label(post_pred)[:26]:<26}) "
                f"gap={post_gap:+7.4f}  L∞={max_q}/255{flip}"
            )

        if iter_count % iters_per_k == 0 and k < max_k:
            k = min(k * 2, max_k)

    with torch.no_grad():
        final_logits = logits_for_images(model, adv.unsqueeze(0))[0]
    final_pred = int(final_logits.argmax().item())

    if stats["status"] not in {"flipped", "no_movable_pixels", "clean_disagrees_with_true_label"}:
        stats["status"] = "flipped" if final_pred != true_label_idx else "budget_exhausted"

    stats["iterations"] = iter_count
    stats["final_pred"] = final_pred
    stats["perturbed_pixel_channels"] = int(((adv - clean).abs() > 0).sum().item())
    stats["final_linf"] = float((adv - clean).abs().max().item())
    stats["final_rmse"] = float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())
    return adv, stats


def baseline_pgd_attack(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    true_label_idx: int,
    *,
    epsilon: float = 0.03,
    min_delta: float = 0.003,
    steps: int = 10,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Original baseline: dense, untargeted, sign-of-gradient PGD.

    Returns (adv tensor, stats dict). Same surface as sparse_runnerup_attack.
    """
    device = x_clean.device
    clean = x_clean.detach()
    step_size = max(epsilon / 4.0, 1.0 / 255.0)
    adv = clean.clone()
    best = adv.clone()
    best_delta = 0.0
    final_pred = true_label_idx
    iters_done = 0

    if verbose:
        print(f"  baseline: step_size={step_size:.5f} epsilon={epsilon:.4f} min_delta={min_delta}")

    for _ in range(steps):
        iters_done += 1
        adv.requires_grad_(True)
        logits = logits_for_images(model=model, image_bchw=adv.unsqueeze(0))
        loss = F.cross_entropy(logits, torch.tensor([true_label_idx], device=device))
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + step_size * grad.sign()
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)
        pred = predict_index(model=model, image_chw=adv)
        final_pred = pred
        delta = float((adv - clean).abs().max().item())
        if verbose:
            flip = " FLIPPED" if pred != true_label_idx else ""
            print(
                f"  baseline #{iters_done:<2} pred={pred:>3} ({_label(pred)[:26]:<26}) "
                f"L∞={delta:.5f} (={int(round(delta * 255))}/255){flip}"
            )
        if delta > best_delta:
            best = adv.clone()
            best_delta = delta
        if pred != true_label_idx and delta >= min_delta:
            best = adv.clone()
            break

    adv = best
    stats = {
        "status": "flipped" if final_pred != true_label_idx else "budget_exhausted",
        "iterations": iters_done,
        "final_pred": int(final_pred),
        "perturbed_pixel_channels": int(((adv - clean).abs() > 0).sum().item()),
        "final_linf": float((adv - clean).abs().max().item()),
        "final_rmse": float(torch.sqrt(torch.mean((adv - clean) ** 2)).item()),
        "runner_up_idx": -1,
        "initial_gap": 0.0,
        "max_quanta_used": int(round(float((adv - clean).abs().max().item()) * 255)),
        "k_history": [],
    }
    return adv, stats


def cascade_attack(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    true_label_idx: int,
    *,
    sparse_iters: int = 15,
    sparse_max_quanta: int = 1,
    dense_iters: int = 12,
    dense_max_quanta: int = 7,
    quantum: float = QUANTUM,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """sparse_runnerup at ±1/255 first; on failure, dense PGD up to ±7/255.

    The dense phase keeps the sparse partial result and pushes ±1/255 per step
    along sign(∇ cross_entropy) across the entire centre crop, snapping to the
    8-bit grid each iteration and capping any single pixel at ±dense_max_quanta/255.
    """
    adv, sparse_stats = sparse_runnerup_attack(
        model,
        x_clean,
        true_label_idx,
        quantum=quantum,
        max_linf_quanta=sparse_max_quanta,
        max_total_iters=sparse_iters,
        verbose=verbose,
    )
    if sparse_stats.get("status") in {"flipped", "clean_disagrees_with_true_label"}:
        sparse_stats["phase"] = "sparse"
        sparse_stats["dense_iters"] = 0
        sparse_stats["sparse_iters"] = sparse_stats.get("iterations", 0)
        return adv, sparse_stats

    if verbose:
        print(f"  cascade: sparse exhausted, switching to dense PGD")

    clean = _quantise(x_clean.detach())
    device = x_clean.device
    mask = centre_crop_mask(clean.shape).to(device)
    dense_done = 0

    for _ in range(dense_iters):
        adv_var = adv.detach().clone().requires_grad_(True)
        logits = logits_for_images(model, adv_var.unsqueeze(0))[0]
        loss = F.cross_entropy(logits, torch.tensor([true_label_idx], device=device))
        grad = torch.autograd.grad(loss, adv_var)[0]

        delta = adv - clean
        new_delta = (delta + grad.sign() * quantum).clamp(
            -dense_max_quanta * quantum, dense_max_quanta * quantum
        )
        new_delta = new_delta * mask
        adv = (clean + new_delta).clamp(0.0, 1.0)
        adv = _quantise(adv)
        dense_done += 1

        with torch.no_grad():
            post_logits = logits_for_images(model, adv.unsqueeze(0))[0]
        post_pred = int(post_logits.argmax().item())
        max_q = int(((adv - clean).abs().max().item()) * 255 + 0.5)
        if verbose:
            flip = " FLIPPED" if post_pred != true_label_idx else ""
            print(
                f"  dense  #{dense_done:<2}         "
                f"pred={post_pred:>3} ({_label(post_pred)[:26]:<26}) "
                f"            L∞={max_q}/255{flip}"
            )
        if post_pred != true_label_idx:
            break

    with torch.no_grad():
        final_logits = logits_for_images(model, adv.unsqueeze(0))[0]
    final_pred = int(final_logits.argmax().item())

    sparse_iter_count = sparse_stats.get("iterations", 0)
    return adv, {
        "phase": "sparse_then_dense",
        "status": "flipped" if final_pred != true_label_idx else "budget_exhausted",
        "sparse_iters": sparse_iter_count,
        "dense_iters": dense_done,
        "iterations": sparse_iter_count + dense_done,
        "final_pred": final_pred,
        "perturbed_pixel_channels": int(((adv - clean).abs() > 0).sum().item()),
        "final_linf": float((adv - clean).abs().max().item()),
        "final_rmse": float(torch.sqrt(torch.mean((adv - clean) ** 2)).item()),
        "runner_up_idx": sparse_stats.get("runner_up_idx", -1),
        "initial_gap": sparse_stats.get("initial_gap", 0.0),
        "max_quanta_used": int(round(float((adv - clean).abs().max().item()) * 255)),
        "k_history": sparse_stats.get("k_history", []),
    }


ATTACKS: dict[str, typing.Callable[..., tuple[torch.Tensor, dict]]] = {
    "baseline": baseline_pgd_attack,
    "sparse_runnerup": sparse_runnerup_attack,
    "cascade": cascade_attack,
}
