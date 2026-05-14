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


def _prune_to_min_subset(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    true_label_idx: int,
    importance: torch.Tensor,
    flip_margin: float = 0.2,
    verbose: bool = False,
) -> tuple[torch.Tensor, int, int]:
    """Binary-search the smallest top-N (ranked by signed importance) such that
    the flip survives by at least `flip_margin` logits.

    A bare argmax check at the decision boundary is unreliable: cuDNN's
    non-deterministic conv kernels can return slightly different logits on
    different runs of the same input, occasionally flipping argmax. Requiring
    `runner_up_logit - true_label_logit >= flip_margin` keeps a few extra
    pixels but the result is robust against that noise (and against the
    validator running its own model evaluation later).
    """
    delta = adv - clean
    nz_mask = delta.abs() > 0
    n_total = int(nz_mask.sum().item())
    if n_total <= 1:
        return adv, n_total, n_total

    nz_indices = nz_mask.flatten().nonzero().squeeze(-1)
    ranking = importance.flatten()[nz_indices]
    order = torch.argsort(ranking, descending=True)
    sorted_nz = nz_indices[order]
    shape = adv.shape
    clean_flat = clean.flatten()
    adv_flat = adv.flatten()

    def adv_with_top(n: int) -> torch.Tensor:
        keep_flat = torch.zeros_like(clean_flat, dtype=torch.bool)
        keep_flat[sorted_nz[:n]] = True
        return torch.where(keep_flat, adv_flat, clean_flat).view(shape)

    def margin_of(tensor: torch.Tensor) -> float:
        with torch.no_grad():
            logits = logits_for_images(model, tensor.unsqueeze(0))[0]
        masked = logits.clone()
        masked[true_label_idx] = float("-inf")
        return float((masked.max() - logits[true_label_idx]).item())

    lo, hi = 1, n_total
    while lo < hi:
        mid = (lo + hi) // 2
        if margin_of(adv_with_top(mid)) >= flip_margin:
            hi = mid
        else:
            lo = mid + 1

    pruned = adv_with_top(lo)
    pruned_margin = margin_of(pruned)
    if pruned_margin < flip_margin:
        if verbose:
            print(
                f"  prune  : revert ({n_total} kept; pruned margin {pruned_margin:+.4f} "
                f"< required {flip_margin:+.4f})"
            )
        return adv, n_total, n_total

    if verbose:
        print(
            f"  prune  : {n_total} -> {lo} pixels ({100.0 * lo / n_total:.1f}% kept, "
            f"margin {pruned_margin:+.4f})"
        )
    return pruned, n_total, lo


def sparse_runnerup_attack(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    true_label_idx: int,
    *,
    quantum: float = QUANTUM,
    initial_k: int = 256,
    max_k: int = 16384,
    iters_per_k: int = 2,
    max_total_iters: int = 20,
    max_linf_quanta: int = 1,
    prune_after_flip: bool = True,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Iterative-refresh sparse adversarial attack at ±1/255.

    Each iteration:
      1. Compute a FRESH gradient ∇(logit[true] - logit[runner_up]) at the
         current adv (not clean!). The pixel importances update as we perturb.
      2. Pick top-K of the unsaturated, in-bounds, in-centre-crop pixels by
         |grad|.
      3. Apply step = sign(-grad) * 1/255 at those K, snap to the 8-bit grid.
      4. K grows geometrically every `iters_per_k` iterations.

    Stop on first flip. Then a binary-search pruner removes pixels that turned
    out not to be load-bearing.
    """
    import time as _time

    device = x_clean.device
    clean = _quantise(x_clean.detach())
    adv = clean.clone()
    mask = centre_crop_mask(clean.shape).to(device)

    # One forward to find the runner-up class.
    with torch.no_grad():
        init_logits = logits_for_images(model, clean.unsqueeze(0))[0]
    top2 = init_logits.topk(2)
    top1_idx = int(top2.indices[0].item())
    runner_up_idx = int(top2.indices[1].item())
    initial_gap = float((top2.values[0] - top2.values[1]).item())

    _t0 = _time.perf_counter()
    if verbose:
        modifiable = int(mask.sum().item())
        print(
            f"clean: top1={top1_idx} ({_label(top1_idx)}) "
            f"runner_up={runner_up_idx} ({_label(runner_up_idx)}) "
            f"gap={initial_gap:+.4f}  modifiable={modifiable}"
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

    if top1_idx != true_label_idx:
        stats["status"] = "clean_disagrees_with_true_label"
        stats["final_pred"] = top1_idx
        stats["final_linf"] = 0.0
        stats["final_rmse"] = 0.0
        return adv, stats

    saturated = torch.zeros_like(clean, dtype=torch.bool)
    k = initial_k
    iter_count = 0
    max_q = 0

    while iter_count < max_total_iters:
        # Check current state; break on flip.
        with torch.no_grad():
            cur_logits = logits_for_images(model, adv.unsqueeze(0))[0]
        if int(cur_logits.argmax().item()) != true_label_idx:
            stats["status"] = "flipped"
            break

        # Fresh gradient at the CURRENT adv (this is the key vs single-shot).
        var = adv.detach().clone().requires_grad_(True)
        logits = logits_for_images(model, var.unsqueeze(0))[0]
        gap_var = logits[true_label_idx] - logits[runner_up_idx]
        grad = torch.autograd.grad(gap_var, var)[0]

        step_sign = -grad.sign()
        delta = adv - clean
        new_delta = (delta + step_sign * quantum).clamp(
            -max_linf_quanta * quantum, max_linf_quanta * quantum
        )
        actual_step = new_delta - delta
        movable = (actual_step.abs() > 0) & (mask > 0) & ~saturated

        num_movable = int(movable.sum().item())
        k_used = min(k, num_movable)
        if k_used == 0:
            stats["status"] = "no_movable_pixels"
            break

        score = (grad.abs() * movable.to(grad.dtype)).flatten()
        topk = torch.topk(score, k_used)
        sel_flat = torch.zeros_like(score, dtype=torch.bool)
        sel_flat[topk.indices] = True
        sel = sel_flat.view_as(grad)

        adv = adv + actual_step * sel.to(adv.dtype)
        adv = _quantise(adv)
        saturated = saturated | sel

        iter_count += 1
        stats["k_history"].append(k_used)
        max_q = max(max_q, int(((adv - clean).abs().max().item()) * 255 + 0.5))
        stats["max_quanta_used"] = max_q

        if verbose:
            with torch.no_grad():
                post_logits = logits_for_images(model, adv.unsqueeze(0))[0]
            post_pred = int(post_logits.argmax().item())
            post_gap = float(
                (post_logits[true_label_idx] - post_logits[runner_up_idx]).item()
            )
            tag = "FLIPPED" if post_pred != true_label_idx else "no"
            elapsed_ms = int((_time.perf_counter() - _t0) * 1000)
            total_perturbed = int(((adv - clean).abs() > 0).sum().item())
            print(
                f"  iter #{iter_count:<2} K={k_used:>5}  gap={post_gap:+7.4f}  "
                f"pred={_label(post_pred)[:18]:<18}  pixels={total_perturbed:>6}  "
                f"{tag:<7} {elapsed_ms:>5}ms"
            )

        if iter_count % iters_per_k == 0 and k < max_k:
            k = min(k * 2, max_k)

    # Final state
    with torch.no_grad():
        final_logits = logits_for_images(model, adv.unsqueeze(0))[0]
    final_pred = int(final_logits.argmax().item())

    if stats["status"] not in {"flipped", "no_movable_pixels", "clean_disagrees_with_true_label"}:
        stats["status"] = "flipped" if final_pred != true_label_idx else "budget_exhausted"

    if prune_after_flip and stats["status"] == "flipped":
        # Fresh gradient at the post-flip state for accurate per-pixel importance.
        var = adv.detach().clone().requires_grad_(True)
        post_logits = logits_for_images(model, var.unsqueeze(0))[0]
        post_gap = post_logits[true_label_idx] - post_logits[runner_up_idx]
        grad_at_flip = torch.autograd.grad(post_gap, var)[0]
        # Signed contribution: helpful pixels have delta*grad < 0, so we negate
        # to rank helpful-first.
        importance = -((adv - clean) * grad_at_flip)
        adv, n_before, n_after = _prune_to_min_subset(
            model=model,
            clean=clean,
            adv=adv,
            true_label_idx=true_label_idx,
            importance=importance,
            verbose=verbose,
        )
        stats["pruned_from"] = n_before
        stats["pruned_to"] = n_after
        with torch.no_grad():
            final_pred = int(
                logits_for_images(model, adv.unsqueeze(0))[0].argmax().item()
            )

    stats["iterations"] = iter_count
    stats["final_pred"] = final_pred
    stats["perturbed_pixel_channels"] = int(((adv - clean).abs() > 0).sum().item())
    stats["final_linf"] = float((adv - clean).abs().max().item())
    stats["final_rmse"] = float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())

    if verbose:
        total_ms = int((_time.perf_counter() - _t0) * 1000)
        print(
            f"result {stats['status']}  pred={_label(final_pred)} "
            f"K={stats['perturbed_pixel_channels']}  L∞={stats['final_linf']:.5f}  "
            f"RMSE={stats['final_rmse']:.6f}  total={total_ms}ms ({iter_count} iters)"
        )
    return adv, stats


def sparse_fool_attack(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    true_label_idx: int,
    *,
    quantum: float = QUANTUM,
    initial_k: int = 128,
    max_k: int = 16384,
    iters_per_k: int = 1,
    max_total_iters: int = 12,
    max_linf_quanta: int = 1,
    num_candidates: int = 1,
    prune_after_flip: bool = True,
    flip_margin: float = 0.2,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """SparseFool-style adversarial attack at ±1/255.

    Differs from sparse_runnerup_attack in target selection:
      - sparse_runnerup_attack fixes the runner-up at the clean image and
        targets it every iteration.
      - sparse_fool_attack picks, every iteration, the class with the SHORTEST
        linearised L2 distance to its decision boundary:
            dist_k = (logit[true] - logit[k]) / ||grad(logit[true] - logit[k])||
        That dist captures both "how much logit gap to close" and "how
        sensitive the gap is to perturbation." Picking min(dist_k) over the
        top-`num_candidates` competing classes is exactly the DeepFool
        boundary-closest criterion from Modas et al. (SparseFool, CVPR 2020).

    Pixel selection within an iteration is the same as our other sparse
    attack: top-K by |grad of (logit[true] - logit[best_k])|, step each by
    sign(-grad) × 1/255, saturate-once.
    """
    import time as _time

    device = x_clean.device
    clean = _quantise(x_clean.detach())
    adv = clean.clone()
    mask = centre_crop_mask(clean.shape).to(device)

    with torch.no_grad():
        init_logits = logits_for_images(model, clean.unsqueeze(0))[0]
    top2_init = init_logits.topk(2)
    top1_idx = int(top2_init.indices[0].item())
    initial_runner_up = int(top2_init.indices[1].item())
    initial_gap = float((top2_init.values[0] - top2_init.values[1]).item())

    _t0 = _time.perf_counter()
    if verbose:
        modifiable = int(mask.sum().item())
        print(
            f"clean: top1={top1_idx} ({_label(top1_idx)}) "
            f"runner_up={initial_runner_up} ({_label(initial_runner_up)}) "
            f"gap={initial_gap:+.4f}  modifiable={modifiable}  num_candidates={num_candidates}"
        )

    stats: dict[str, typing.Any] = {
        "clean_top1_idx": top1_idx,
        "runner_up_idx": initial_runner_up,
        "initial_gap": initial_gap,
        "iterations": 0,
        "k_history": [],
        "target_class_history": [],
        "max_quanta_used": 0,
        "perturbed_pixel_channels": 0,
        "status": "init",
    }

    if top1_idx != true_label_idx:
        stats["status"] = "clean_disagrees_with_true_label"
        stats["final_pred"] = top1_idx
        stats["final_linf"] = 0.0
        stats["final_rmse"] = 0.0
        return adv, stats

    saturated = torch.zeros_like(clean, dtype=torch.bool)
    k = initial_k
    iter_count = 0
    max_q = 0
    target_class = initial_runner_up

    while iter_count < max_total_iters:
        # Check current state, break on flip.
        with torch.no_grad():
            cur_logits = logits_for_images(model, adv.unsqueeze(0))[0]
        if int(cur_logits.argmax().item()) != true_label_idx:
            stats["status"] = "flipped"
            break

        # One forward (with grad). We'll then call autograd.grad multiple times
        # against this same graph to get per-class boundary normals.
        var = adv.detach().clone().requires_grad_(True)
        logits = logits_for_images(model, var.unsqueeze(0))[0]
        true_logit = logits[true_label_idx]

        # Candidate classes: top by current logit value, excluding true_label.
        topn = logits.topk(num_candidates + 1).indices.tolist()
        candidates = [c for c in topn if c != true_label_idx][:num_candidates]
        if not candidates:
            stats["status"] = "no_movable_pixels"
            break

        # DeepFool closest-boundary criterion: dist_k = gap_k / ||grad_k||
        best_dist = float("inf")
        best_grad: torch.Tensor | None = None
        best_target = candidates[0]
        best_gap = 0.0
        for c in candidates:
            gap_c = true_logit - logits[c]
            grad_c = torch.autograd.grad(gap_c, var, retain_graph=True)[0]
            w_norm = float(grad_c.norm().item())
            gap_val = float(gap_c.item())
            dist = gap_val / max(w_norm, 1e-12)
            if dist < best_dist:
                best_dist = dist
                best_grad = grad_c
                best_target = c
                best_gap = gap_val

        assert best_grad is not None
        target_class = best_target
        grad = best_grad

        # Step direction: ±1/255 sign(-grad) per pixel; pick top-K by |grad|.
        step_sign = -grad.sign()
        delta = adv - clean
        new_delta = (delta + step_sign * quantum).clamp(
            -max_linf_quanta * quantum, max_linf_quanta * quantum
        )
        actual_step = new_delta - delta
        movable = (actual_step.abs() > 0) & (mask > 0) & ~saturated

        num_movable = int(movable.sum().item())
        k_used = min(k, num_movable)
        if k_used == 0:
            stats["status"] = "no_movable_pixels"
            break

        score = (grad.abs() * movable.to(grad.dtype)).flatten()
        topk = torch.topk(score, k_used)
        sel_flat = torch.zeros_like(score, dtype=torch.bool)
        sel_flat[topk.indices] = True
        sel = sel_flat.view_as(grad)

        adv = adv + actual_step * sel.to(adv.dtype)
        adv = _quantise(adv)
        saturated = saturated | sel

        iter_count += 1
        stats["k_history"].append(k_used)
        stats["target_class_history"].append(target_class)
        max_q = max(max_q, int(((adv - clean).abs().max().item()) * 255 + 0.5))
        stats["max_quanta_used"] = max_q

        if verbose:
            with torch.no_grad():
                post_logits = logits_for_images(model, adv.unsqueeze(0))[0]
            post_pred = int(post_logits.argmax().item())
            post_gap_target = float(
                (post_logits[true_label_idx] - post_logits[target_class]).item()
            )
            tag = "FLIPPED" if post_pred != true_label_idx else "no"
            elapsed_ms = int((_time.perf_counter() - _t0) * 1000)
            total_perturbed = int(((adv - clean).abs() > 0).sum().item())
            print(
                f"  iter #{iter_count:<2} K={k_used:>5}  target={target_class:>3}({_label(target_class)[:12]:<12}) "
                f"dist={best_dist:.4f}  gap_to_tgt={post_gap_target:+7.4f}  "
                f"pred={_label(post_pred)[:12]:<12}  pixels={total_perturbed:>6}  "
                f"{tag:<7} {elapsed_ms:>5}ms"
            )

        if iter_count % iters_per_k == 0 and k < max_k:
            k = min(k * 2, max_k)

    # Final state
    with torch.no_grad():
        final_logits = logits_for_images(model, adv.unsqueeze(0))[0]
    final_pred = int(final_logits.argmax().item())

    if stats["status"] not in {"flipped", "no_movable_pixels", "clean_disagrees_with_true_label"}:
        stats["status"] = "flipped" if final_pred != true_label_idx else "budget_exhausted"

    if prune_after_flip and stats["status"] == "flipped":
        # Use the actual winning class (not the last target) for pruning importance.
        var = adv.detach().clone().requires_grad_(True)
        post_logits = logits_for_images(model, var.unsqueeze(0))[0]
        post_gap = post_logits[true_label_idx] - post_logits[final_pred]
        grad_at_flip = torch.autograd.grad(post_gap, var)[0]
        importance = -((adv - clean) * grad_at_flip)
        adv, n_before, n_after = _prune_to_min_subset(
            model=model,
            clean=clean,
            adv=adv,
            true_label_idx=true_label_idx,
            importance=importance,
            flip_margin=flip_margin,
            verbose=verbose,
        )
        stats["pruned_from"] = n_before
        stats["pruned_to"] = n_after
        with torch.no_grad():
            final_pred = int(
                logits_for_images(model, adv.unsqueeze(0))[0].argmax().item()
            )

    stats["iterations"] = iter_count
    stats["final_pred"] = final_pred
    stats["perturbed_pixel_channels"] = int(((adv - clean).abs() > 0).sum().item())
    stats["final_linf"] = float((adv - clean).abs().max().item())
    stats["final_rmse"] = float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())

    if verbose:
        total_ms = int((_time.perf_counter() - _t0) * 1000)
        print(
            f"result {stats['status']}  pred={_label(final_pred)} "
            f"K={stats['perturbed_pixel_channels']}  L∞={stats['final_linf']:.5f}  "
            f"RMSE={stats['final_rmse']:.6f}  total={total_ms}ms ({iter_count} iters)"
        )
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


def sparse_single_shot_attack(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    true_label_idx: int,
    *,
    quantum: float = QUANTUM,
    initial_k: int = 32,
    max_k: int = 16384,
    max_linf_quanta: int = 1,  # API compat; this implementation always uses 1
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Single-shot sparse adversarial attack at ±1/255.

    One backward pass at the clean image, then a binary search over the
    minimum top-K of pixels (ranked by |grad|) that flips the label. Each
    selected pixel is touched exactly once with sign(grad) × 1/255.

    Algorithm:
      1. logits = model(clean); record runner_up = top2[1].
      2. grad = ∂(logit[true] - logit[runner_up]) / ∂clean    (one backward).
      3. Sort all pixel-channels by |grad| descending (apply centre-crop mask).
      4. Probe K = initial_k, 2*initial_k, ... until argmax(model(adv_K)) flips.
      5. Binary-search downward between (last-failing-K, found-K) to find the
         smallest K that still flips.

    Total cost: 1 backward + ~log2(max_k) + ~log2(K) forwards ≈ 12-15 forwards.
    No re-evaluation of the gradient mid-attack, so the linear approximation
    can mis-rank pixels on hard images; for those, sparse_runnerup_attack or
    cascade_attack are better.
    """
    import time as _time

    device = x_clean.device
    clean = _quantise(x_clean.detach())
    mask = centre_crop_mask(clean.shape).to(device)

    var = clean.clone().requires_grad_(True)
    logits = logits_for_images(model, var.unsqueeze(0))[0]
    top2 = logits.topk(2)
    top1_idx = int(top2.indices[0].item())
    runner_up_idx = int(top2.indices[1].item())
    initial_gap = float((top2.values[0] - top2.values[1]).item())

    _t0 = _time.perf_counter()
    if verbose:
        print(
            f"clean: top1={top1_idx} ({_label(top1_idx)}) "
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

    if top1_idx != true_label_idx:
        stats["status"] = "clean_disagrees_with_true_label"
        stats["final_pred"] = top1_idx
        stats["final_linf"] = 0.0
        stats["final_rmse"] = 0.0
        return clean, stats

    gap_var = logits[true_label_idx] - logits[runner_up_idx]
    grad = torch.autograd.grad(gap_var, var)[0] * mask

    step_sign = -grad.sign()
    movable = (
        ((step_sign > 0) & (clean < 1.0)) | ((step_sign < 0) & (clean > 0.0))
    ) & (mask > 0)
    score = (grad.abs() * movable.to(grad.dtype)).flatten()
    sorted_idx = torch.argsort(score, descending=True)
    step_flat = (step_sign * quantum).flatten()
    upper_k = int(min(max_k, movable.sum().item()))

    def adv_at_k(n: int) -> torch.Tensor:
        keep = torch.zeros_like(step_flat)
        keep[sorted_idx[:n]] = 1.0
        return _quantise((clean + (step_flat * keep).view_as(clean)).clamp(0.0, 1.0))

    def predict_pair(a: torch.Tensor) -> tuple[int, float]:
        with torch.no_grad():
            lg = logits_for_images(model, a.unsqueeze(0))[0]
        return int(lg.argmax().item()), float((lg[true_label_idx] - lg[runner_up_idx]).item())

    probes = 0
    last_fail_k = 0
    found_k: int | None = None
    adv = clean.clone()
    k = max(1, initial_k)
    while k <= upper_k:
        adv = adv_at_k(k)
        pred, gap_now = predict_pair(adv)
        probes += 1
        stats["k_history"].append(k)
        if verbose:
            tag = "FLIPPED" if pred != true_label_idx else "no"
            elapsed_ms = int((_time.perf_counter() - _t0) * 1000)
            print(
                f"  probe  K={k:<6}  gap={gap_now:+7.4f}  pred={_label(pred)[:18]:<18}  "
                f"{tag:<7} {elapsed_ms:>5}ms"
            )
        if pred != true_label_idx:
            found_k = k
            break
        last_fail_k = k
        if k == upper_k:
            break
        k = min(k * 2, upper_k)

    if found_k is None:
        pred, _ = predict_pair(adv)
        stats["status"] = "budget_exhausted"
        stats["iterations"] = probes
        stats["perturbed_pixel_channels"] = int(((adv - clean).abs() > 0).sum().item())
        stats["final_linf"] = float((adv - clean).abs().max().item())
        stats["final_rmse"] = float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())
        stats["max_quanta_used"] = int(round(stats["final_linf"] * 255))
        stats["final_pred"] = pred
        return adv, stats

    lo, hi = last_fail_k, found_k
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        adv = adv_at_k(mid)
        pred, gap_now = predict_pair(adv)
        probes += 1
        if verbose:
            tag = "FLIPPED" if pred != true_label_idx else "no"
            elapsed_ms = int((_time.perf_counter() - _t0) * 1000)
            print(
                f"  search K={mid:<6}  gap={gap_now:+7.4f}  pred={_label(pred)[:18]:<18}  "
                f"{tag:<7} {elapsed_ms:>5}ms"
            )
        if pred != true_label_idx:
            hi = mid
        else:
            lo = mid

    adv = adv_at_k(hi)
    pred, _ = predict_pair(adv)
    bumps = 0
    while pred == true_label_idx and hi < upper_k and bumps < 8:
        hi += 1
        adv = adv_at_k(hi)
        pred, _ = predict_pair(adv)
        bumps += 1

    stats["status"] = "flipped" if pred != true_label_idx else "budget_exhausted"
    stats["iterations"] = probes + bumps
    stats["perturbed_pixel_channels"] = hi
    stats["final_linf"] = float((adv - clean).abs().max().item())
    stats["final_rmse"] = float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())
    stats["max_quanta_used"] = int(round(stats["final_linf"] * 255))
    stats["final_pred"] = pred

    if verbose:
        total_ms = int((_time.perf_counter() - _t0) * 1000)
        print(
            f"result {stats['status']}  K={hi}  L∞={stats['final_linf']:.5f}  "
            f"RMSE={stats['final_rmse']:.6f}  total={total_ms}ms ({probes + bumps} forwards)"
        )
    return adv, stats


ATTACKS: dict[str, typing.Callable[..., tuple[torch.Tensor, dict]]] = {
    "baseline": baseline_pgd_attack,
    "sparse_runnerup": sparse_runnerup_attack,
    "sparse_fool": sparse_fool_attack,
    "sparse_single_shot": sparse_single_shot_attack,
    "cascade": cascade_attack,
}
