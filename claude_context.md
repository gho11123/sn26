# Claude Context — Perturb Dashboard

Working notes for the dashboard / log-streamer system around the Perturb
validator. Drop into this when resuming work or onboarding a new session.

## Project background

Perturb is a Bittensor subnet (netuid 26) where validators serve image
classification challenges (EfficientNetV2-M on Pexels-fetched photos) and
miners return adversarial perturbations bounded by L∞ and SSIM/PSNR gates.

- Validator: [neurons/validator.py](neurons/validator.py)
- Miner (baseline PGD): [neurons/miner.py](neurons/miner.py)
- Constants & defaults: [perturbnet/constants.py](perturbnet/constants.py)
- Scoring formula: see [validator.py:709-841](neurons/validator.py#L709-L841)
- Winner selection (per tempo, every ~72 min): [validator.py:849-962](neurons/validator.py#L849-L962)
- `HISTORY_SIZE = 50` rolling window. Log label `avg100` is misleading — actual window is 50 (stale variable name).
- **Recent rename:** in the new validator code, `avg100=` in rank lines is now `avg_score=`. Both forms parsed.

## Files I built / modified

| File | What it does | Notes |
|---|---|---|
| [download_run_logs.py](download_run_logs.py) | W&B log streamer for validator uid 0 | Forked from sn85 (hotkey-based) → rewritten for Perturb (UID-based, no proxy). Watch mode handles crash→restart, dedupes cursor drift. |
| [dashboard.py](dashboard.py) | Single-file stdlib dashboard server | Parses `wandb_logs/uid0.log`, serves HTML + JSON. Gzip on. Cache TTL 3 s. |
| [API.md](API.md) | Dashboard API doc | `/api/stats`, `/api/challenge?block=N&format=csv`, `/healthz` |
| [ecosystem.wandb-logs.config.js](ecosystem.wandb-logs.config.js) | pm2 config for both processes | `perturb-wandb-logs` (streamer) + `perturb-dashboard` (web). Watch disabled in pm2. |
| `wandb_logs/uid0.log` | Streamed validator log | Single file, grows continuously |
| `wandb_logs/.uid0.state.json` | Streamer cursor checkpoint | Resume after pm2 restart |

## Running services (pm2)

```
perturb-wandb-logs   → python download_run_logs.py --watch
perturb-dashboard    → python dashboard.py
```

Start both: `pm2 start ecosystem.wandb-logs.config.js`. Dashboard runs on `http://0.0.0.0:8800`.

## Dashboard architecture

Single-file Python, stdlib only (`http.server`, `urllib`, `gzip`). No Flask/FastAPI.

### Parser (`parse_log`)

Walks the log file once, extracting:

- **Per-uid stats** (`per_uid`): scores, norms, rmses, psnrs, rts (lists), plus `processed` (validator's running count) and `last_*` markers.
- **Per-challenge results** (`challenges_detailed`): one entry per challenge announcement, results list attached as scoring lines stream in. Each result carries `response_time_ms`.
- **Loop summaries** (`loops`): one per `loop_summary` line.
- **Validator-side ranking** (`validator_leaderboard`): parsed from `rank=N uid=M (avg100|avg_score)=X emission=Y` lines emitted during `_set_weights()`. Authoritative on-chain ranking.
- **Live leaderboard** (`live_leaderboard`): we simulate `_set_weights()` ourselves on captured data. Mirrors validator logic exactly (50-sample rolling history, ≥50 processed required, winner-take-all). **Includes non-eligible uids** sorted to the bottom with `eligible: false`.
- **Per-uid live rank change** (`rank_change`): also computes a "previous snapshot" with the most recent challenge's contributions removed → ↑/↓ arrows show movement vs the latest challenge.
- **Current epoch leaderboard** (`current_epoch`): aggregated only over responses since the last `set_weights` event.

### Log format handling — both formats supported

The validator log style changed substantially in May 2026. **Both formats parse:**

| Area | Old | New |
|---|---|---|
| Per-uid scoring | `[ts] uid=N status=N score=N processed=N reason=... norm=N rmse=N epsilon=N ssim=N psnr_db=N` | `uid=N status=N score=N response_time_ms=N processed=N reason=... ...` (no timestamp prefix; RT inlined) |
| Challenge announce | `Challenge task=X prompt=Y eps=Z` | `challenge_summary epsilon=N fallback_used=B llm_verified=B prompt=W task_id=X true_label=W` (alphabetical) |
| Loop summary | Keys in code-defined order | Keys alphabetical |
| Per-uid batch start | _(none)_ | `miner_response_evaluations block=N count=N` |
| Rank field | `avg100=` | `avg_score=` |
| Weights summary key | `top5=` | `top10=` |
| `verify_and_score` line | Emitted at INFO before each status==200 uid | No longer emitted (RT inlined) |

Implementation notes:
- `MINER_RE` has optional `(?:response_time_ms=(?P<rt_inline>\d+) )?` capture
- `_parse_loop_summary_fields` and `_parse_chal_summary_fields` use **independent per-key extraction** so order doesn't matter
- `EVAL_MARKER_RE` recognizes `miner_response_evaluations block=N count=N` (helps fall back to set `current_task_id` if `challenge_summary` was missed)

### Endpoints

| Path | Returns |
|---|---|
| `GET /` | HTML page (gzipped) |
| `GET /api/stats?force=1?` | Full JSON aggregate (gzip ~1 MB from ~17 MB raw) |
| `GET /api/challenge?block=N` or `?task_id=X` (`&format=json\|csv`) | Single challenge's full per-miner results |
| `GET /healthz` | `{"ok": true}` |

`StatsCache` (TTL 3 s) reuses parsed result across rapid requests.

### Frontend (inline JS+HTML inside `INDEX_HTML`)

Two main cards:

1. **set_weights ranking** — tabbed: `epoch (from log)` and `live (computed now)`. Same columns: rank, uid, avg×100, emission×100, avg RMSE (last 50), avg RT (ms, last 50). Live tab adds `↑/↓` rank-change arrows next to the rank pill (vs the previous challenge). Click any row to expand a **per-uid detail panel** below.

2. **Challenges (click for per-miner scores)** — sortable, filterable by block. Click any challenge row to expand a **per-challenge miner detail panel** showing every uid's response.

### Inner detail tables — per-tab unique ids

Both tabs' inner tables collided on `id="uid-detail-table"`, causing the second `ensurePager` call to *move* the first tab's pager. Fixed by namespacing:

- Epoch expansion → `id="uid-detail-table--val-table"` + pager `pager-uid-detail-table--val-table`
- Live expansion → `id="uid-detail-table--val-table-live"` + pager `pager-uid-detail-table--val-table-live`

**Important**: `renderUidDetailRow` takes a `parentTableId` param. Each tab's inner table is independent.

### State management

All persisted across the 30 s auto-refresh:

| Global | What | Keyed by |
|---|---|---|
| `sortState` | active sort col + direction | table id |
| `pageState` | current page, page size, totalPages | table id |
| `filterState` | filter query strings | table id |
| `selectedUid` | which uid (if any) is expanded in val-table(s) | scalar |
| `selectedTaskId` | which challenge is expanded in challenges-table | scalar |
| `activeRankingTab` | `"epoch"` or `"live"` | scalar |

`rewireAll()` runs after every fetch and restores all of this from the globals.

### Filtering

- **Challenges table**: substring match on the block column (`fs.block`)
- **Set_weights ranking (both tabs)**: **exact** uid match (`uidText === q`). User explicitly wanted exact, not substring — typing "10" should match only uid 10, not uid 100/101/110.

### Inner-table pagers

Inserted *above* the inner table (re-parented in `expandUidRow` / `expandChallengeRow` after `wireSortable`). With 400+ rows in the per-uid detail, a pager below would be off-screen. Subtle visual band via `.detail-scroll > .pager` CSS.

## Score display convention

All 0–1 scores display **×100** so they read intuitively (`96.13` not `0.9613`). Column headers say `×100`. Implemented via `fmtScore(s, d)` helper. Sort keys remain raw 0–1.

## Per-uid detail panel — column semantics

Header: `# | block | ts | prompt | score ×100 (Δ, winner) | rank | reason | L∞ | RMSE (Δ, winner) | SSIM | PSNR | RT ms (Δ, winner)`

Per-metric format: `<value> (<delta>) #<refUid>`:
- **Δ (delta)**: green `+X.XX` when ahead of reference; red `-X.XX` when behind. Score uses delta to runner-up if this uid won, else delta to winner.
- **#refUid (muted grey)**: shows the **runner-up** uid if this uid is the metric winner (i.e. who you're beating), otherwise the **winner** uid (who beat you). Always grey — never `#self`.
- **Tie-break**: if this uid is tied for the winning value, this uid is credited as the winner.
- **Best highlight**: cells where this uid owns the chal-wide min for RMSE/RT get the `.best` class (green-bold).

Rank cell: `[#N/100]` pill (green if #1; amber if 2-5; default if 6+).

## Live ranking — non-eligible inclusion

The live tab includes non-eligible uids (those with `<50` captured samples) at the bottom. They appear with:
- Dimmed text color (`.not-eligible` row class)
- Amber `NE (32/50)` tag next to the rank pill showing samples-captured / 50
- `emission = 0` always (never receives emission)

## Score formula (current — May 2026)

```python
# per challenge per uid:
score = PERTURBATION_WEIGHT * perturbation_score  +  SPEED_WEIGHT * speed_score
```

| Constant | Old | **Current** |
|---|---|---|
| `TIMEOUT_SECONDS` | 30 | **10** |
| `SPEED_WEIGHT` | 0.35 | **0.25** |
| `PERTURBATION_WEIGHT` | 0.65 | **0.75** |

So the speed window is 3× tighter than before, and the speed component carries slightly less weight in the total. Net: a 5 s response USED to cost 0.083 of the total score from speed alone; NOW it costs 0.125. Plus anything >10 s is full timeout penalty (`speed_score=0`).

`speed_score = 1 - response_time_ms / (timeout * 1000)`, clamped.

## How `response_time_ms` is actually measured (and a known issue)

The validator reads `response.dendrite.process_time` from bittensor's dendrite. Set at [bittensor/core/dendrite.py:590](.venv/lib/python3.12/site-packages/bittensor/core/dendrite.py#L590):

```python
synapse.dendrite.process_time = str(time.time() - start_time)
```

`start_time` is captured at line 561, *before* the HTTP POST. The stamp is recorded *after* `await response.json()` AND `process_server_response(...)` — both of which are partially synchronous.

For exceptions (timeouts, connection errors), `process_time` is **never set** → validator falls back to `timeout_seconds * 1000` (sentinel value `10000` ms).

**Known hypothesis** (the user shared this in discussion):
The `process_time` value is contaminated by validator-side asyncio queue position. Because `dendrite.forward` does `asyncio.gather` over all K miners, and the post-receive `json.loads` + `Pydantic` work after `await response.json()` is fully synchronous (no `await` between body-arrival and the `time.time()` stamp), the N-th coroutine to wake up has its `process_time` stamped AFTER N-1 prior coroutines' sync work has run. This makes large responses systematically worse-scored on speed, independent of true miner speed.

`response.axon.process_time` exists and is server-stamped at [bittensor/core/axon.py:1528](.venv/lib/python3.12/site-packages/bittensor/core/axon.py#L1528) — it doesn't include the validator's parse queue and would be a cleaner scoring input. The hypothesis suggests the validator should switch to it (with a `min(axon_pt, dendrite_pt)` clamp against miner-side lying).

The mechanism is provably real in code; magnitude is plausible but would need direct measurement to confirm dominance. Verified during session 2026-05-18.

## Miner selection (validator side)

Three-stage filter, per challenge ([validator.py:578-678](neurons/validator.py#L578-L678)):

1. **Build candidate pool** (`_available_miner_uids`):
   - Skip own hotkey
   - Skip uids with `axon.ip == "0.0.0.0"`
   - **Dedup by coldkey OR axon IP** via union-find. If A shares coldkey with B and B shares IP with C → A/B/C in one group. Keep only **lowest UID** per group.
2. **Mark valuable** (`_valuable_miner_uids`): subset with `processed_count >= 50`.
3. **Pick K=100** (`_select_random_miners`):
   - 80% from valuable, 20% from newcomers (`MINER_EXPLORATION_RATIO = 0.20`)
   - Deterministic per-block seed: `sha256("perturb:{netuid}:{block}")`
   - If valuable+newcomer < K: top up from remaining

**Implications:** Running multiple uids from the same coldkey or IP is wasted registration cost — only the lowest UID is queried. Each "operator group" yields ONE candidate.

After the 2026-05-18 validator restart, the dedup compressed the pool from 246 → 87 uids (~3 uids per operator on average).

## API endpoint for analysis

```bash
# Single challenge as JSON
curl 'http://127.0.0.1:8800/api/challenge?block=8201791'

# Same as CSV (for pandas)
curl 'http://127.0.0.1:8800/api/challenge?block=8201791&format=csv' -o c.csv
```

Pandas one-liner: `pd.read_csv("http://127.0.0.1:8800/api/challenge?block=N&format=csv")`.

## Performance

- **Parse cost**: ~600–700 ms for ~12k-line log with ~1200 challenges. Mostly JSON serialization.
- **Response size**: ~17 MB raw / **~1 MB gzipped**. Gzip is the critical fix — without it, the browser stalls on a 23 MB JSON download.
- **Parse optimization**: live-rank computation uses deque + running-sum incremental aggregation (O(U) per challenge instead of O(U × HISTORY_SIZE)).

## Validator log line patterns (current parser handles all)

```
# Old format (still in historical portion of the log)
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | Challenge task=BLOCK-SEED prompt=PROMPT eps=0.NNNN
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | verify_and_score task_id=BLOCK-SEED response_time_ms=NNNN
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | uid=N status=NNN score=0.NNNNNN processed=NN reason=... norm=0.NNN rmse=0.NNN epsilon=0.NNN ssim=0.NNN psnr_db=NN.NN

# Current format
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | [run_id=...] challenge_summary epsilon=N fallback_used=B llm_verified=B prompt=W task_id=X true_label=W
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | [run_id=...] miner_selection selected=N total_pool=N valuable_pool=N
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | miner_response_evaluations block=N count=N
uid=N status=N score=N response_time_ms=N processed=N reason=... ...   (no timestamp prefix)
... more uid lines ...
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | [run_id=...] loop_summary avg_norm=N avg_rmse=N avg_score=N block=N max_score=N min_score=N reasons=R selected=N success=N/N

# Both formats for set_weights output
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | rank=N uid=M (avg100|avg_score)=X emission_raw=Y emission=Z
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | [run_id=...] weights_summary eligible=N distributed=N (top5|top10)=...
2026-MM-DD HH:MM:SS,SSS | __main__ | INFO | set_weights success
```

## User preferences picked up

- **Terse responses**: short, get to the point. Don't over-explain.
- **Score readability**: 0–1 scale was misleading; user wants ×100 display everywhere.
- **No clutter**: removed several columns and features per request (winner-score column, global rank, rank trend arrows in detail). Keep tables tight.
- **Inline > separate columns**: when adding winner uid info, user wanted it inline in each metric column, not as a separate "winners" column.
- **Tie-handling**: when this uid is tied for first, display should consistently credit `self` as winner.
- **Make features visible**: pagers must be discoverable — moved above inner tables after user reported pagination "disappearing" (actually below the viewport).
- **Exact vs substring filter**: uid filter is *exact* match. Block filter is *substring* match. Don't change without asking.
- **Live ranking should include all uids**: non-eligible uids appear at the bottom with NE marker, not hidden.

## Removed features (don't re-add without asking)

- "Current epoch ranking" card (separate from set_weights)
- "Recent loops" card
- "Reason breakdown" + "HTTP status" cards
- "Norm distribution" card
- Per-challenge rank trend arrows (`↑1` / `↓1` next to rank pill in per-uid detail)
- Global rank with trend (`global #32 ↑5` line in rank cell)
- Running-average overall rank (`overall #9.69 ↑0.04`)
- Standalone `winners (S/R/T)` column
- `winner_score` column

Lean UI is preferred. Default to NOT adding stats — ask first.

## Things to verify when resuming

1. `pm2 list | grep perturb` — both `perturb-wandb-logs` and `perturb-dashboard` should be online.
2. `wandb_logs/uid0.log` should be growing (`wc -l` over time).
3. `curl http://127.0.0.1:8800/healthz` → `{"ok":true}`.
4. Most recent challenge timestamp in `/api/stats?force=1 | jq '.last_line_ts'` should be within ~minutes of now.

## Known caveats / TODOs

- Individual `rank=…` lines may now be emitted at **DEBUG** level by the new validator code (per [validator.py:929](neurons/validator.py#L929) — `logger.debug`). Most logger configs filter DEBUG, so the epoch leaderboard may only have `weights_summary` top10 data going forward. The `validator_leaderboard` field may degrade to 10 entries instead of 230+. Workaround: bump the validator's log level, or display top10 from `weights_summary` when full ranks aren't available.
- `set_weight_events` field exposes the last 10 set_weights events but the UI doesn't currently display them — only in the API for ad-hoc analysis.
- Page can grow heavy past ~2000 challenges. We'd need to cap `challenges_detailed` in the response or move per-challenge results to a separate endpoint at that point. Not yet at that scale.

## Recent diagnosis (2026-05-18 session)

- Validator restarted at `2026-05-18 15:01`. After restart, `processed_counts` reset to 0 → `valuable_pool=0` in the `miner_selection` log until uids cross 50 again.
- Pool dropped from 246 (yesterday) → 87 (today) due to the new aggressive coldkey/IP dedup in `_available_miner_uids`. uid 193 (formerly a top performer) is no longer in the pool — it's being merged with a lower-numbered uid that shares its coldkey or IP.
- User asked about fixing this: setting the lower uid's axon IP to `0.0.0.0` on-chain would filter it out at Stage 1 of `_available_miner_uids`, leaving uid 193 as the sole representative of its operator group.
- User shared a detailed hypothesis on validator-side asyncio queue position contaminating `process_time`. Reviewed and confirmed the mechanism is real and code references are accurate; magnitude needs empirical measurement. Suggested mitigation: switch scoring to `response.axon.process_time` (server-stamped, see [axon.py:1528](.venv/lib/python3.12/site-packages/bittensor/core/axon.py#L1528)) with a `min(axon_pt, dendrite_pt)` clamp against miner-side lying.
