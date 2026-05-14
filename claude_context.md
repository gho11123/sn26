# Perturb Mining — Working Context

This file is a self-contained handoff. Reading it should get a new session
up to speed without re-deriving anything.

---

## 1. What this project is

Bittensor **subnet 26 (Perturb)** is an adversarial-image task:

- Validators pull an image from Pexels, classify it with **EfficientNetV2-M**
  (`torchvision`, `IMAGENET1K_V1` weights), and send the image + true label to
  miners via an `AttackChallenge` synapse.
- Miners must return a perturbed PNG (`perturbed_image_b64`) such that:
  1. The model's argmax on the perturbed image is **not** `true_label`.
  2. `min(epsilon, MAX_LINF_DELTA=0.03) ≥ L∞(adv − clean) ≥ MIN_LINF_DELTA=0.003`.
  3. `SSIM(clean, adv) ≥ 0.98` and `PSNR(clean, adv) ≥ 38 dB`.
- The validator scores the miner: see §3 for the formula.

We're operating a miner on wallet `onepiece-02` / hotkey `op-2`, registered on
netuid 26, served via PM2 on port 9000.

---

## 2. Repo layout (what we touched)

- `neurons/miner.py` — the live miner. Wired to `cascade_attack`. Saves each
  challenge into `tasks/<block>/{clean.png,perturbed.png,meta.json}` for
  later replay/analysis. Block id is parsed from `task_id.split("-")[0]`
  (used to call `subtensor.get_current_block()` which raced with the sync
  loop and produced `block_unknown/` directories — fixed).
- `perturbnet/attack.py` — all attack implementations + the registry.
- `perturbnet/model.py` — model loader; cudnn.benchmark + bf16 autocast +
  optional torch.compile.
- `perturbnet/image_io.py` — PNG ↔ base64 ↔ tensor helpers (unchanged).
- `scripts/score_task.py` — local replay of the validator's gates + score on a
  saved task. Supports `--attack {saved,baseline,sparse_runnerup,sparse_fool,cascade}`.
- `scripts/miner.env` — `WALLET_NAME=onepiece-02`, `WALLET_HOTKEY=op-2`,
  `NETUID=26`, `NETWORK=finney`, `MINER_PORT=9000`.
- `tasks/<block>/` — saved live challenges. `meta.json` includes
  `task_id`, `block`, `prompt`, `true_label`, `epsilon`, `norm_type`,
  `min_delta`, `timeout_seconds`, `caller_hotkey`, `status`,
  `target_index`, `final_pred`, `best_delta`, `response_time_ms`,
  `attack`, `attack_phase`, `attack_status`, `iterations`,
  `perturbed_pixel_channels`, `final_rmse`, `max_quanta_used`,
  `runner_up_idx`, `initial_gap`.

PM2: `pm2 list` shows `perturb-miner` (and `perturb-llm-endpoint`, irrelevant
to mining). Restart with `pm2 restart perturb-miner`.

---

## 3. Validator scoring (from `neurons/validator.py:674-796` and `perturbnet/constants.py`)

Constants (all env-overridable):
- `MIN_LINF_DELTA = 0.003`
- `MAX_LINF_DELTA = 0.03`
- `MIN_SSIM = 0.98`
- `MIN_PSNR_DB = 38.0`
- `LINF_COMPONENT_WEIGHT = 0.7`
- `RMSE_COMPONENT_WEIGHT = 0.3`
- `PERTURBATION_WEIGHT = 0.65`
- `SPEED_WEIGHT = 0.35`
- timeout per challenge = 30s

Gates (ANY failure → score 0):
1. `norm >= MIN_LINF_DELTA`
2. `norm <= min(epsilon, MAX_LINF_DELTA)` (effective cap = 0.03)
3. `argmax(model(adv)) != true_label`
4. `ssim >= MIN_SSIM`
5. `psnr_db >= MIN_PSNR_DB`

If all gates pass:
```
effective_max_delta = min(epsilon, MAX_LINF_DELTA)        # = 0.03 for our tasks
linf_ratio  = clamp((norm - MIN_LINF_DELTA) / (effective_max_delta - MIN_LINF_DELTA), 0, 1)
linf_score  = (1 - linf_ratio)²
rmse_ratio  = clamp(rmse / effective_max_delta, 0, 1)
rmse_score  = (1 - rmse_ratio)²
perturbation_score = (0.7*linf_score + 0.3*rmse_score) / 1.0
speed_score = 1 - min(response_time_ms / (timeout * 1000), 1)
final = 0.65 * perturbation_score + 0.35 * speed_score
```

At `L∞ = 1/255 = 0.00392`: `linf_score = 0.933` (essentially pinned).
The active levers are **RMSE** and **RT (speed)**.

Sensitivity: halving RT (e.g. 2000 → 1000 ms) adds ~+0.012 to final; halving
RMSE (e.g. 0.00043 → 0.00021) adds ~+0.003. **Speed is ~4× more leverage per
% change than RMSE**, in our regime.

---

## 4. Attacks (in `perturbnet/attack.py`)

The `ATTACKS` registry exposes four implementations. All share a common
`(model, x_clean, true_label_idx, *, verbose=False) -> (adv, stats)` signature.

### `baseline` — `baseline_pgd_attack`
Original repo baseline. Dense untargeted PGD, sign-of-gradient, step =
`max(epsilon/4, 1/255)`. Mostly fails the `norm ≤ 0.03` gate because step
is too big.

### `sparse_runnerup` — `sparse_runnerup_attack`
Iterative-refresh sparse PGD at **±1/255 max per pixel**.
- Fresh gradient every iter (key vs single-shot): `∇(logit[true] − logit[runner_up])`
  at the current `adv`.
- Pick top-K **unsaturated** pixels by `|grad|`; saturated pixels are masked
  out so each pixel is touched exactly once.
- Step = `sign(−grad) × 1/255`; quantise to 8-bit grid.
- K grows: `256 → 512 → 1024 → 2048 → ... → 16384`, doubling every `iters_per_k=2`.
- Max 15 iterations.
- Post-flip pruner trims redundant pixels (see below).
- `runner_up_idx` is fixed to the top-2 class at the **clean** image.

### `sparse_fool` — `sparse_fool_attack` (newest)
Same as `sparse_runnerup` but with **adaptive target selection** each iter
(SparseFool / DeepFool criterion from Modas et al. 2020):
- Each iter, consider the top-`num_candidates` non-true classes.
- For each candidate `k`, compute `grad_k = ∇(logit[true] − logit[k])` at
  current adv and the linearised distance `dist_k = (logit[true] − logit[k]) / ‖grad_k‖₂`.
- Pick `k*` with minimum `dist_k`. Use `grad_{k*}` for pixel selection.
- Defaults: `initial_k=128, iters_per_k=1, max_total_iters=12, num_candidates=1`.
- Verified to find better targets than the clean-image runner-up on hard
  tasks (e.g. on 8184124 it found `harvestman` even though clean's runner-up
  was `black and gold garden spider`).

### `cascade` — `cascade_attack` (what the live miner uses)
Two-phase fallback:
1. **Phase A**: call `sparse_runnerup_attack` (currently — could be swapped
   to `sparse_fool_attack`). Returns early if flipped.
2. **Phase B** (only if Phase A's status == `"budget_exhausted"`): dense
   quantised PGD across the entire centre crop, step `+sign(∇ce) × 1/255`
   per iter, clamping the running delta to `±7/255` total (since
   `8/255 ≈ 0.0314 > 0.03` MAX cap would fail the validator gate). Uses
   Phase A's partial result as initialisation. Up to 12 iters.

**To swap the live miner to `sparse_fool`:** edit `cascade_attack` to call
`sparse_fool_attack` instead of `sparse_runnerup_attack`. One-line change.

### Pruner — `_prune_to_min_subset`
After a flip, binary-search the smallest top-N (sorted by **signed
contribution** `-(δ × ∇_at_flip)`) such that the result still flips by
**`flip_margin=0.2`** logits.

- `flip_margin` is critical: cuDNN's non-deterministic conv kernels (and
  PNG-roundtrip rounding at the model's resize input) give slightly
  different argmax results across model invocations. Without margin, the
  pruner can return a tensor that flips in its own eval but not in the
  validator's. 0.2 logits of margin reliably absorbs this.
- Verify-or-revert at the end: if final pruned candidate doesn't have
  the required margin, return the unpruned `adv` instead.

---

## 5. Model wrapper (`perturbnet/model.py`)

Speed optimizations added:
- `torch.backends.cudnn.benchmark = True` at module import (auto-tunes
  fastest conv algorithm; safe since input shape is stable at 480×480).
- **bfloat16 autocast** in `logits_for_images` and `predict_index` on CUDA.
  Toggle: `PERTURB_BF16=0` env var to disable. **Default on.** ~2-3× speedup
  on the conv-heavy backbone, negligible adversarial precision loss because
  our final perturbation is uint8-quantised anyway.
- `torch.compile(model, mode="default")` — **opt-in via `PERTURB_TORCH_COMPILE=1`**.
  Default off because on the A4000 (small SM count) it hangs for minutes
  during graph capture with backward passes enabled. If retried on a bigger
  GPU, would add ~1.3-1.5× on top.

---

## 6. How to test

```bash
source .venv/bin/activate

# Pick a saved task block and an attack:
python scripts/score_task.py 8183339 --attack sparse_fool        # easy task (whale)
python scripts/score_task.py 8184124 --attack sparse_fool        # medium (spider)
python scripts/score_task.py 8183401 --attack cascade            # hard (banded gecko, gap≈4.9)
python scripts/score_task.py 8183339 --attack saved              # score the live miner's saved perturbed.png

# Disable optimizations temporarily:
PERTURB_BF16=0 python scripts/score_task.py 8184124 --attack sparse_fool
PERTURB_TORCH_COMPILE=1 python scripts/score_task.py 8184124 --attack sparse_fool  # don't do this; hangs

# Live miner control:
pm2 list
pm2 restart perturb-miner          # pick up code changes
pm2 logs perturb-miner --lines 30
```

`score_task.py` runs the chosen attack on the saved `clean.png`, re-encodes
to PNG, decodes again (mimicking the validator's path), then evaluates all 5
gates and the score formula. Final block prints the 4 dashboard metrics
(L∞, RMSE, SSIM, PSNR) plus FINAL_SCORE.

---

## 7. Performance snapshots

Reference: top leaderboard miners typically score 0.94–0.96 on subnet 26
with `L∞ = 1/255 = 0.00392`, RMSE 0.00018–0.00088, RT 600–1500 ms.

Our trajectory on task **8184124** (barn spider, initial gap = 3.49):

| Algorithm                                | K    | RMSE     | RT (ms) | Score   | Notes |
|---|---|---|---|---|---|
| baseline (PGD)                           | dense| 0.06+    | 200     | 0 (gate)| L∞ overflows 0.03 cap |
| single-shot sparse (probe+BS)            | 16384| —        | 1100    | 0 (no flip) | budget exhausted on gap>3 |
| iterative `sparse_runnerup`              | 7415 | 0.00043  | 2186    | 0 (no margin) | flip lost on PNG roundtrip |
| iterative + `flip_margin=0.2` in prune   | 5076 | 0.00036  | 2186    | ~0.93 | reliable |
| `sparse_fool` (num_candidates=3, K=64)   | 5076 | 0.00036  | 2896    | 0.931 | adaptive target → harvestman |
| `sparse_fool` (num_candidates=2, K=128)  | 4729 | 0.00035  | 1991    | 0.942 | iters_per_k=1 helps |
| **`sparse_fool` + bf16 + num_cands=1**   | TBD  | TBD      | ~700-900 (est) | ~0.95 (est) | current default |

Top miner reference (uid 159) on a marine_mammal task: **K≈2500, RMSE 0.00040,
RT 654 ms, score 0.957**.

The remaining gap to the top is mostly **wall-clock**, not pixel count. Our
RMSE is already in the same league.

---

## 8. Known issues / things to be aware of

1. **cuDNN non-determinism at decision boundaries.** Two evaluations of the
   same tensor can give different argmax when the top-2 logits are within
   ~0.1 of each other. This causes spurious `FAIL label_match_with_original`
   in score_task.py even when the pruner thought it had flipped.
   Mitigation: `flip_margin=0.2` in the pruner. Don't lower below ~0.15.

2. **PNG roundtrip is bit-exact** for tensors on the 8-bit grid (verified
   theoretically and empirically). The "PNG roundtrip lost the flip" failures
   are actually #1 above, not roundtrip — but the symptom presents at the
   `decode_image_b64(encode_image_b64(adv))` step.

3. **bf16 autocast slightly changes logits** vs fp32. The validator runs
   fp32. The flip_margin absorbs the difference. If we ever see flips
   disappearing after roundtrip, try `PERTURB_BF16=0` and see if the issue
   goes away.

4. **torch.compile hangs on A4000.** Don't enable. If we move to a bigger
   GPU, retry with `PERTURB_TORCH_COMPILE=1`.

5. **Validator nonce/signature errors** (`NotVerifiedException`) in the live
   miner log are normal — bittensor flakiness, not our bug. The validator
   retries and the next request usually lands. Roughly 30-50% of incoming
   signatures fail at boundaries, but we still get a steady stream of
   successful queries.

6. **EfficientNetV2-M preprocess** resizes shortest-side to 480 then
   centre-crops to 480×480, then normalises with ImageNet mean/std. So
   pixels outside the centre `min(H,W) × min(H,W)` square never reach the
   model. Our `centre_crop_mask` already excludes them — perturbing them
   would waste RMSE budget for zero adversarial effect.

7. **Within-class flips are "free" success.** The validator's
   `true_label` is the *exact* EfficientNet category. Flipping `marmot → fox
   squirrel` (both rodents, adjacent class ids) scores the same as a
   semantic cross-class flip. This is why our sparse_fool often finds an
   adjacent class as the closest boundary — that's the cheapest valid flip.

---

## 9. Vulnerabilities in the scoring system (for awareness)

Documented these earlier; flagged in case we want to tune our miner to
exploit them later or report them to the subnet operator.

1. **Image caching.** Validator pulls from Pexels with `query=<prompt>` and
   random page/photo from a pool of ~400 images per prompt × 20 prompts
   ≈ 8000 images total. A miner that hashes `sha256(clean_image_b64) →
   precomputed_perturbed` wins on RT for repeat images.
2. **Fallback image is a single static file** (`assets/dog_1.jpg`, label
   "dog"). Pre-compute once, win every Pexels outage.
3. **Within-class flips count as success** (above).
4. **Sparse / patch attacks exploit pixel-averaging in PSNR/RMSE/SSIM** —
   localised perturbations slip past the global-mean gates.
5. **Speed weight is 0.35** — caching wins outsized rewards.

The leader miners almost certainly exploit #1 and #3.

---

## 10. Next steps (where we left off)

Last action: set bf16 autocast + cudnn.benchmark + `num_candidates=1` as
defaults; disabled torch.compile because it hangs on A4000.

**Open items:**

1. **Validate the new defaults end-to-end.** Run
   `python scripts/score_task.py 8184124 --attack sparse_fool` and confirm
   RT is ~700-900 ms with score ~0.94+. Then retry several saved tasks
   (8183339, 8183401, 8183997, 8184124) to make sure nothing regressed.
2. **Swap `cascade_attack` to call `sparse_fool_attack`** in its Phase A
   (currently calls `sparse_runnerup_attack`). One-line edit in
   `perturbnet/attack.py`, then `pm2 restart perturb-miner`.
3. **If speed is still the bottleneck**, the next real lever is a
   surrogate-model approach: use a smaller fast model (EfficientNet-B0
   or MobileNet) for gradient computation, EffNetV2-M only for the
   final flip verification. Risk: perturbation may not transfer.
   Reward: potentially 3-5× more speedup on top of current.
4. **If RMSE is the bottleneck**, the next real lever is a research-grade
   L0 attack (SparseRS / CW-L0). Would land us in the 0.95-0.96 score
   range. ~2-3 hours of careful implementation.
5. **Bandwidth**: nothing else needs immediate attention. The miner is
   stable, saving tasks correctly, picking up validator requests, and
   scoring non-zero on most tasks.

**Convenient commands for the next session:**

```bash
# Sanity check
source .venv/bin/activate && python -c "from perturbnet.attack import ATTACKS; print(sorted(ATTACKS.keys()))"
# Expect: ['baseline', 'cascade', 'sparse_fool', 'sparse_runnerup']

# Check live miner
pm2 status
pm2 logs perturb-miner --lines 40

# Score a recent saved task with the current attack
ls tasks/ | tail -5
python scripts/score_task.py <block> --attack sparse_fool
```
