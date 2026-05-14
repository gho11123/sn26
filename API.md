# Perturb Dashboard API

Local HTTP API served by [dashboard.py](dashboard.py). Parses the validator log
at `wandb_logs/uid<N>.log` (produced by `download_run_logs.py --watch`) and
exposes the parsed stats as JSON for the browser dashboard and for analysis
tools.

**Default base URL:** `http://127.0.0.1:8800`

The base URL, log path, and other options can be changed via flags:
```bash
python dashboard.py --host 0.0.0.0 --port 8800 \
  --log wandb_logs/uid0.log --top 20 --recent-loops 20 --cache-ttl 3.0
```

All endpoints are unauthenticated. Read-only. CORS is **not** enabled; this is
intended for local-host use (curl, jq, pandas, the bundled HTML page).

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | The HTML dashboard (validator-side ranking + challenges table) |
| `GET` | `/api/stats` | Full aggregate stats parsed from the log file |
| `GET` | `/api/challenge` | Per-miner results for a single challenge, by block or task_id |
| `GET` | `/healthz` | Liveness check |

---

### `GET /`

Returns the dashboard HTML page (`Content-Type: text/html; charset=utf-8`).
The page auto-refreshes every 30 s by polling `/api/stats`, and has a manual
"Refresh now" button that calls `/api/stats?force=1`.

---

### `GET /api/stats`

Returns the full aggregated state of the log file as a single JSON object.

**Query params**

| Param | Type | Default | Behavior |
|---|---|---|---|
| `force` | `1` | _(absent)_ | Bypass the in-process `StatsCache` (3 s TTL by default) and re-parse the log immediately |

**Response 200 — JSON**

Top-level fields:

| Field | Type | Description |
|---|---|---|
| `ok` | bool | Always `true` on success |
| `path` | string | Absolute path to the log file being parsed |
| `line_count` | int | Total lines parsed |
| `file_size_bytes` | int | Size of the log file |
| `file_mtime` | float | Unix mtime of the log file |
| `last_line_ts` | string | Most recent validator-side timestamp seen in the log (`YYYY-MM-DD HH:MM:SS`) |
| `run_ids` | string[] | Distinct W&B run IDs observed (set after restarts) |
| `restarts` | string[] | Timestamps of `RUN RESTARTED` banners written by the streamer |
| `loops_count` | int | Number of `loop_summary` lines parsed |
| `loops` | [Loop](#loop-object)[] | One entry per loop iteration, chronological |
| `challenges` | object | `{prompt: count}` histogram of challenge prompts |
| `epsilon` | object\|null | `{n, min, max, mean}` of all `Challenge … eps=…` values |
| `reasons` | object | `{reason: count}` across every per-uid scoring event |
| `statuses` | object | `{http_code: count}` across every per-uid scoring event |
| `norm_quantiles` | object\|null | L∞ distribution over successful responses: `{n, min, p25, median, p75, max}` |
| `miner_count` | int | Number of distinct uids seen scoring |
| `leaderboard` | [WindowEntry](#windowentry-object)[] | Per-uid stats over the **log-window** samples (not validator-authoritative) |
| `validator_leaderboard` | [ValidatorEntry](#validatorentry-object)[] | Most recent set_weights snapshot, enriched with window-side stats |
| `latest_weight_event` | object\|null | Banner stats for the latest set_weights: `{ts, eligible, distributed, top5, ranks_count}` |
| `weight_events_count` | int | Number of set_weights events captured in the log window |
| `set_weight_events` | object[] | Up to last 10 set_weights call outcomes: `{ts, result, msg}` |
| `current_epoch` | object | `{since_ts, challenges_count, leaderboard}` — aggregates responses since the last set_weights |
| `challenges_detailed` | [Challenge](#challenge-object)[] | Every challenge in the log window, newest first, with full per-miner results |
| `_cache_age_seconds` | float | Age of the cached parse used for this response |
| `_serve_ms` | float | Server-side time to serve this request |
| `_generated_at` | float | Unix epoch when this parse was produced |

#### Example

```bash
curl -s http://127.0.0.1:8800/api/stats | jq 'keys'
```

```bash
# How many challenges are in the current log window?
curl -s http://127.0.0.1:8800/api/stats | jq '.challenges_detailed | length'

# Show the top-3 validator-side ranks
curl -s http://127.0.0.1:8800/api/stats | jq '.validator_leaderboard[0:3] | map({rank, uid, avg100, emission})'
```

---

### `GET /api/challenge`

Returns a single challenge's full per-miner results. Designed for analysis use
cases (pandas, jq, custom scripts).

**Query params** — at least one of `block` or `task_id` is required.

| Param | Type | Behavior |
|---|---|---|
| `block` | int | Exact match on the challenge's block number |
| `task_id` | string | Exact match on the full task_id (`"{block}-{seed}"`) |
| `format` | `json` \| `csv` | Output format. Default `json` |

**Response 200 — JSON** (when `format=json`)

```json
{
  "ok": true,
  "query": { "block": "8181890", "task_id": "" },
  "matched_count": 1,
  "challenge": {
    "task_id": "8181890-2616943861694322662",
    "block": 8181890,
    "prompt": "rabbit",
    "epsilon": 0.1662,
    "ts": "2026-05-14 12:15:05",
    "total_responses": 100,
    "success_count": 28,
    "avg_score": 0.2558,
    "max_score": 0.9495,
    "avg_norm_success": 0.00420,
    "avg_response_time_ms": 2882,
    "results": [ /* one entry per uid, see Result schema */ ]
  }
}
```

**Response 200 — CSV** (when `format=csv`)

```
uid,status,score,reason,norm,rmse,ssim,psnr_db,epsilon,response_time_ms,processed,ts
228,200,0.949518,"success",0.00392,0.00088,0.9997,61.15,0.0,748,175,"2026-05-14 12:15:08"
...
```

CSV columns are fixed: `uid, status, score, reason, norm, rmse, ssim, psnr_db,
epsilon, response_time_ms, processed, ts`. String fields are quoted with `"`
and embedded quotes are doubled.

**Response 400**

```json
{ "ok": false, "error": "specify ?block=<n> or ?task_id=<id>" }
```

```json
{ "ok": false, "error": "block must be an integer, got 'notanumber'" }
```

**Response 404**

```json
{
  "ok": false,
  "error": "no challenge matched",
  "query": { "block": "999999999", "task_id": "" },
  "hint": "the log window may not contain this challenge; try /api/stats to see what's available"
}
```

#### Examples

**curl + jq** — top 5 successful miners on a specific block:
```bash
curl -s 'http://127.0.0.1:8800/api/challenge?block=8181890' \
  | jq '.challenge.results
        | map(select(.reason=="success"))
        | sort_by(-.score)
        | .[0:5]
        | map({uid, score, norm, rmse, response_time_ms})'
```

**curl + jq** — pull every uid that hit the minimum representable L∞ (0.003922):
```bash
curl -s 'http://127.0.0.1:8800/api/challenge?block=8181890' \
  | jq -r '.challenge.results | map(select(.norm==0.003922)) | .[].uid'
```

**Python + pandas** — load and aggregate one challenge:
```python
import pandas as pd
import requests

r = requests.get(
    "http://127.0.0.1:8800/api/challenge",
    params={"block": 8181890},
).json()

c = r["challenge"]
df = pd.DataFrame(c["results"])
print(c["task_id"], c["prompt"], c["success_count"], "/", c["total_responses"])
print(df[df.reason == "success"].nsmallest(10, "rmse")[
    ["uid", "score", "norm", "rmse", "response_time_ms"]
])
```

**Pandas directly from CSV** (one-liner):
```python
df = pd.read_csv("http://127.0.0.1:8800/api/challenge?block=8181890&format=csv")
```

**Save CSV to disk**:
```bash
curl -s 'http://127.0.0.1:8800/api/challenge?block=8181890&format=csv' \
  -o challenge_8181890.csv
```

**Look up by task_id**:
```bash
curl -s 'http://127.0.0.1:8800/api/challenge?task_id=8181890-2616943861694322662' \
  | jq '.challenge | {task_id, success_count, total_responses}'
```

---

### `GET /healthz`

Liveness probe.

**Response 200** — `application/json`:
```json
{ "ok": true }
```

---

## Schemas

### Challenge object

Used in `challenges_detailed` (`/api/stats`) and as the `challenge` field of
`/api/challenge`.

| Field | Type | Description |
|---|---|---|
| `task_id` | string | Full `"{block}-{seed}"` identifier |
| `block` | int\|null | Block number extracted from `task_id` |
| `prompt` | string\|null | Animal category sampled from `PROMPTS` (e.g. `rabbit`, `marine_mammal`) |
| `epsilon` | float\|null | Per-challenge L∞ budget announced to miners (range `[0.06, 0.20)`) |
| `ts` | string\|null | Timestamp of the `Challenge task=…` log line |
| `total_responses` | int | Count of miner responses recorded for this challenge |
| `success_count` | int | Count of `reason="success"` rows |
| `avg_score` | float | Mean score across all responses (including zeros) |
| `max_score` | float | Best single score observed for this challenge |
| `avg_norm_success` | float\|null | Mean L∞ over successful responses only |
| `avg_response_time_ms` | float\|null | Mean response time over rows with a recorded RT |
| `results` | [Result](#result-object)[] | One entry per miner queried for this challenge |

### Result object

Each row corresponds to one miner's response to one challenge.

| Field | Type | Description |
|---|---|---|
| `uid` | int | Miner UID in the metagraph |
| `status` | int | HTTP status of the miner's response (200, 408, 503, …) |
| `score` | float | Final score in `[0, 1]`, computed by the validator |
| `reason` | string | One of: `success`, `response_missing_or_status_error`, `above_max_delta`, `below_min_delta`, `label_match_with_original`, `below_min_ssim`, `below_min_psnr_db`, `decode_failed`, `shape_mismatch`, `value_out_of_range`, `model_inference_failed:…` |
| `norm` | float | L∞ of `adv - clean` |
| `rmse` | float | RMSE of `adv - clean` |
| `ssim` | float | SSIM(clean, adv); only meaningful when `status==200` and gates passed |
| `psnr_db` | float | PSNR in dB |
| `epsilon` | float | Challenge ε for this miner's evaluation (0 when not status==200) |
| `response_time_ms` | int\|null | Wall-clock response time, only present for `status==200` rows |
| `processed` | int | Cumulative count of times this validator has scored this uid |
| `ts` | string\|null | Timestamp of the validator-side `uid=…` log line |

### ValidatorEntry object

Each row of `validator_leaderboard`. Captured at the most recent `set_weights`.

| Field | Type | Description |
|---|---|---|
| `rank` | int | 1-based rank from the validator's perspective |
| `uid` | int | Miner UID |
| `avg100` | float | Mean of the validator's last 50 scores for this uid (variable named `avg100` in the log for historical reasons; window is actually `HISTORY_SIZE=50`) |
| `emission_raw` | float | Raw emission share before scaling |
| `emission` | float | Final on-chain weight; winner-take-all (rank 1 = 1.0, rest = 0.0) |
| `samples_window` | int | How many times this uid appeared in the log window |
| `last_reason_window` | string\|null | The most recent reason seen for this uid |
| `avg_norm_success` | float\|null | Mean L∞ across successful responses (window-side) |
| `avg_rmse_success` | float\|null | Mean RMSE across successful responses (window-side) |
| `avg_psnr_success` | float\|null | Mean PSNR across successful responses (window-side) |

### WindowEntry object

Each row of `leaderboard` (the window-side per-uid aggregation, not validator-authoritative).

| Field | Type | Description |
|---|---|---|
| `uid` | int | Miner UID |
| `samples` | int | Number of rows in this window |
| `success` | int | Number of `success` rows in this window |
| `success_rate` | float | `success / samples` |
| `avg_score` | float | Mean score across this window |
| `avg_norm_success` | float\|null | Mean L∞ across successful rows |
| `avg_rmse_success` | float\|null | Mean RMSE across successful rows |
| `avg_psnr_success` | float\|null | Mean PSNR across successful rows |
| `processed_total` | int | Highest `processed` value seen for this uid (the validator's running counter) |
| `last_status` | int\|null | Most recent HTTP status |
| `last_reason` | string\|null | Most recent reason |
| `last_ts` | string\|null | Timestamp of the most recent row |

### Loop object

Each row of `loops`. One entry per `loop_summary` line.

| Field | Type | Description |
|---|---|---|
| `ts` | string\|null | Timestamp of the `loop_summary` line |
| `run_id` | string | The validator's W&B run identifier |
| `block` | int | Subtensor block when the loop was emitted |
| `selected` | int | Number of miners selected this round (= `K_MINERS`) |
| `success` | int | Successful responses |
| `total` | int | Total responses (selected) |
| `avg` | float | Mean score across the round |
| `min` | float | Min score |
| `max` | float | Max score |
| `avg_norm` | float | Mean L∞ across all responses (incl. zeros) |
| `avg_rmse` | float | Mean RMSE across all responses |
| `reasons` | object | `{reason: count}` for this loop |

---

## Caching & freshness

- `dashboard.py` keeps an in-process `StatsCache` with a 3-second TTL. Repeated
  calls to `/api/stats` and `/api/challenge` within that window reuse the same
  parsed result. Pass `?force=1` to `/api/stats` to bypass.
- The underlying log file (`wandb_logs/uid<N>.log`) is written incrementally by
  `download_run_logs.py --watch`, polling W&B every 30 s. So the freshest data
  in the API is bounded by that streamer's polling cadence, not the cache TTL.
- The window includes only challenges captured since the streamer started. For
  older challenges, you would need to re-fetch from W&B (e.g. by deleting
  `wandb_logs/uid0.log` and `.uid0.state.json` and restarting the watcher;
  that re-downloads the entire run history).

---

## Error handling conventions

- 2xx — JSON body with `"ok": true` (or CSV body for `format=csv`).
- 4xx — JSON body with `"ok": false` and an `error` field; sometimes a `hint`
  field for the common case of "challenge not in the log window."
- 5xx — should not happen during normal operation; the parser is defensive and
  returns `"ok": false, "error": ...` even if the log file is missing.

`Cache-Control: no-store` is set on all responses so browsers and proxies don't
serve stale data.
