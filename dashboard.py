#!/usr/bin/env python3
"""
Perturb validator log dashboard.

Parses `wandb_logs/uid<N>.log` (produced by download_run_logs.py --watch) and
serves a small web dashboard summarising:
    - Run identity (validator run_id / netuid / pid)
    - Reason breakdown across all captured miner responses
    - HTTP status code breakdown
    - Top-N miner leaderboard (avg score, success rate, avg L_inf / RMSE / PSNR)
    - Last K loop_summary rows
    - Challenge prompt distribution

The page auto-refreshes every 30s and has a manual "Refresh now" button.

Usage:
    python dashboard.py
    python dashboard.py --port 8800 --log wandb_logs/uid0.log
    python dashboard.py --recent-loops 30 --top 25

Stdlib only — no Flask, no FastAPI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import threading
import time
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


MINER_RE = re.compile(
    r"uid=(?P<uid>\d+) status=(?P<status>\d+) score=(?P<score>[\d.]+) "
    r"processed=(?P<processed>\d+) reason=(?P<reason>\w+) "
    r"norm=(?P<norm>[\d.]+) rmse=(?P<rmse>[\d.]+) epsilon=(?P<eps>[\d.]+) "
    r"ssim=(?P<ssim>[\d.]+) psnr_db=(?P<psnr>[\d.\-]+)"
)
LOOP_RE = re.compile(
    r"\[run_id=(?P<run_id>[^\]]+)\] loop_summary "
    r"block=(?P<block>\d+) selected=(?P<selected>\d+) "
    r"success=(?P<succ>\d+)/(?P<total>\d+) "
    r"avg_score=(?P<avg>[\d.]+) min_score=(?P<min>[\d.]+) max_score=(?P<max>[\d.]+) "
    r"avg_norm=(?P<an>[\d.]+) avg_rmse=(?P<ar>[\d.]+) reasons=(?P<reasons>\S+)"
)
CHAL_RE = re.compile(
    r"Challenge task=(?P<task>\S+) prompt=(?P<prompt>\w+) eps=(?P<eps>[\d.]+)"
)
RESTART_RE = re.compile(r"\*\*\* RUN RESTARTED at (?P<ts>\S+)")

# Validator-side authoritative ranking lines, emitted during _set_weights():
#   rank=1 uid=10 avg100=0.945720 emission_raw=1.000000 emission=1.000000
RANK_RE = re.compile(
    r"rank=(?P<rank>\d+) uid=(?P<uid>\d+) avg100=(?P<avg100>[\d.]+) "
    r"emission_raw=(?P<er>[\d.]+) emission=(?P<emission>[\d.]+)"
)
# Summary line that closes each rank batch:
#   [run_id=...] weights_summary eligible=230 distributed=5
#                  top5=r1:uid10:avg=0.9457:w=1.0000|r2:uid170:avg=0.9452:w=0.0000|...
WEIGHTS_RE = re.compile(
    r"weights_summary eligible=(?P<eligible>\d+) distributed=(?P<distributed>\d+) "
    r"top5=(?P<top5>\S+)"
)
SET_WEIGHTS_RE = re.compile(r"set_weights (?P<result>success|failed)(?::\s*(?P<msg>.+))?")
# verify_and_score lines: emitted at the start of each per-uid scoring call.
# They carry the task_id and response_time_ms that the very next uid line belongs to
# — but only for uids whose HTTP status was 200 (status != 200 skips verify_and_score).
VERIFY_RE = re.compile(
    r"verify_and_score task_id=(?P<task>\S+) response_time_ms=(?P<rt>\d+)"
)

# Lines we tag onto the most recently seen loop so they can be displayed
# inline with their corresponding loop_summary.
TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse_log(path: str) -> dict:
    """Read the log file end-to-end and produce an aggregate stats dict."""
    if not os.path.exists(path):
        return {
            "ok": False,
            "error": f"Log file not found: {path}",
            "path": path,
        }

    # Per-uid running stats across the whole file.
    per_uid: dict[int, dict] = defaultdict(
        lambda: {
            "n": 0,
            "succ": 0,
            "score_sum": 0.0,
            "norms": [],
            "rmses": [],
            "psnrs": [],
            "processed": 0,
            "last_status": None,
            "last_reason": None,
            "last_ts": None,
        }
    )
    reasons: Counter[str] = Counter()
    statuses: Counter[int] = Counter()
    challenges: Counter[str] = Counter()
    epsilons: list[float] = []
    loops: list[dict] = []
    success_norms: list[float] = []
    success_rmses: list[float] = []
    restarts: list[str] = []
    run_ids: set[str] = set()
    last_line_ts: str | None = None

    # Validator-side rankings come in batches (one per _set_weights call).
    # We buffer rank lines and flush the batch when we see a weights_summary,
    # so the latest weight_events[-1] is always the most recent authoritative ranking.
    pending_rank_batch: list[dict] = []
    weight_events: list[dict] = []
    set_weight_events: list[dict] = []

    # Per-challenge detail: each "Challenge task=… prompt=… eps=…" line opens a new
    # challenge bucket; subsequent per-uid score lines (until the next Challenge) attach
    # to it. verify_and_score lines carry response_time_ms for the next status==200 row.
    challenges_detailed: dict[str, dict] = {}
    current_task_id: str | None = None
    pending_verify_rt: int | None = None

    line_count = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            line = line.rstrip("\n")
            if not line:
                continue

            m_ts = TIMESTAMP_RE.match(line)
            if m_ts:
                last_line_ts = m_ts.group(1)

            m = RESTART_RE.search(line)
            if m:
                restarts.append(m.group("ts"))
                continue

            m = CHAL_RE.search(line)
            if m:
                challenges[m.group("prompt")] += 1
                epsilons.append(float(m.group("eps")))
                task_id = m.group("task")
                current_task_id = task_id
                pending_verify_rt = None
                # task_id format: "{block}-{seed}"
                block = task_id.split("-", 1)[0] if "-" in task_id else ""
                challenges_detailed[task_id] = {
                    "task_id": task_id,
                    "block": int(block) if block.isdigit() else None,
                    "prompt": m.group("prompt"),
                    "epsilon": float(m.group("eps")),
                    "ts": last_line_ts,
                    "results": [],
                }
                continue

            m = VERIFY_RE.search(line)
            if m:
                # Will be consumed by the next status==200 MINER_RE line.
                pending_verify_rt = int(m.group("rt"))
                # If the task_id in this verify line differs from current, switch context
                # (defensive against any reordering of Challenge vs verify lines).
                vtask = m.group("task")
                if vtask and vtask not in challenges_detailed:
                    block = vtask.split("-", 1)[0] if "-" in vtask else ""
                    challenges_detailed[vtask] = {
                        "task_id": vtask,
                        "block": int(block) if block.isdigit() else None,
                        "prompt": None,
                        "epsilon": None,
                        "ts": last_line_ts,
                        "results": [],
                    }
                current_task_id = vtask
                continue

            m = RANK_RE.search(line)
            if m:
                pending_rank_batch.append(
                    {
                        "rank": int(m.group("rank")),
                        "uid": int(m.group("uid")),
                        "avg100": float(m.group("avg100")),
                        "emission_raw": float(m.group("er")),
                        "emission": float(m.group("emission")),
                    }
                )
                continue

            m = WEIGHTS_RE.search(line)
            if m:
                # Parse top5 string: r1:uid10:avg=0.9457:w=1.0000|r2:uid170:avg=0.9452:w=0.0000|...
                top_items = []
                for chunk in m.group("top5").split("|"):
                    parts = chunk.split(":")
                    item = {}
                    for p in parts:
                        if p.startswith("r") and p[1:].isdigit():
                            item["rank"] = int(p[1:])
                        elif p.startswith("uid"):
                            try:
                                item["uid"] = int(p[3:])
                            except ValueError:
                                pass
                        elif p.startswith("avg="):
                            try:
                                item["avg"] = float(p[4:])
                            except ValueError:
                                pass
                        elif p.startswith("w="):
                            try:
                                item["w"] = float(p[2:])
                            except ValueError:
                                pass
                    if item:
                        top_items.append(item)

                weight_events.append(
                    {
                        "ts": last_line_ts,
                        "eligible": int(m.group("eligible")),
                        "distributed": int(m.group("distributed")),
                        "top5": top_items,
                        "ranks": pending_rank_batch,  # full ranking from this batch
                    }
                )
                pending_rank_batch = []
                continue

            m = SET_WEIGHTS_RE.search(line)
            if m:
                set_weight_events.append(
                    {
                        "ts": last_line_ts,
                        "result": m.group("result"),
                        "msg": m.group("msg"),
                    }
                )
                continue

            m = LOOP_RE.search(line)
            if m:
                d = m.groupdict()
                run_ids.add(d["run_id"])
                # reasons field is like: success:33,above_max_delta:7,...
                breakdown: dict[str, int] = {}
                for chunk in d["reasons"].split(","):
                    if ":" in chunk:
                        k, v = chunk.split(":", 1)
                        try:
                            breakdown[k] = int(v)
                        except ValueError:
                            pass
                loops.append(
                    {
                        "ts": last_line_ts,
                        "run_id": d["run_id"],
                        "block": int(d["block"]),
                        "selected": int(d["selected"]),
                        "success": int(d["succ"]),
                        "total": int(d["total"]),
                        "avg": float(d["avg"]),
                        "min": float(d["min"]),
                        "max": float(d["max"]),
                        "avg_norm": float(d["an"]),
                        "avg_rmse": float(d["ar"]),
                        "reasons": breakdown,
                    }
                )
                continue

            m = MINER_RE.search(line)
            if m:
                uid = int(m.group("uid"))
                status = int(m.group("status"))
                score = float(m.group("score"))
                processed = int(m.group("processed"))
                reason = m.group("reason")
                norm = float(m.group("norm"))
                rmse = float(m.group("rmse"))
                psnr = float(m.group("psnr"))

                d = per_uid[uid]
                d["n"] += 1
                d["score_sum"] += score
                d["processed"] = max(d["processed"], processed)
                d["last_status"] = status
                d["last_reason"] = reason
                d["last_ts"] = last_line_ts
                if reason == "success":
                    d["succ"] += 1
                    d["norms"].append(norm)
                    d["rmses"].append(rmse)
                    d["psnrs"].append(psnr)
                    success_norms.append(norm)
                    success_rmses.append(rmse)
                reasons[reason] += 1
                statuses[status] += 1

                # Attach this result to the current challenge bucket.
                if current_task_id is not None:
                    chal = challenges_detailed.get(current_task_id)
                    if chal is not None:
                        # Pull response_time_ms from the preceding verify line ONLY for
                        # status==200 rows. Other statuses skip verify_and_score in the
                        # validator, so no rt is available.
                        rt = pending_verify_rt if status == 200 else None
                        if status == 200:
                            pending_verify_rt = None
                        chal["results"].append(
                            {
                                "uid": uid,
                                "status": status,
                                "score": score,
                                "reason": reason,
                                "norm": norm,
                                "rmse": rmse,
                                "psnr_db": psnr,
                                "ssim": float(m.group("ssim")),
                                "epsilon": float(m.group("eps")),
                                "processed": processed,
                                "response_time_ms": rt,
                                "ts": last_line_ts,
                            }
                        )
                continue

    # Build leaderboard
    leaderboard: list[dict] = []
    for uid, d in per_uid.items():
        if d["n"] < 1:
            continue
        avg = d["score_sum"] / d["n"]
        succ_rate = d["succ"] / d["n"] if d["n"] else 0.0
        leaderboard.append(
            {
                "uid": uid,
                "samples": d["n"],
                "success": d["succ"],
                "success_rate": succ_rate,
                "avg_score": avg,
                "avg_norm_success": (
                    statistics.mean(d["norms"]) if d["norms"] else None
                ),
                "avg_rmse_success": (
                    statistics.mean(d["rmses"]) if d["rmses"] else None
                ),
                "avg_psnr_success": (
                    statistics.mean(d["psnrs"]) if d["psnrs"] else None
                ),
                "processed_total": d["processed"],
                "last_status": d["last_status"],
                "last_reason": d["last_reason"],
                "last_ts": d["last_ts"],
            }
        )
    leaderboard.sort(key=lambda r: (-r["avg_score"], -r["samples"]))

    # Norm distribution quantiles (across all successful responses).
    norm_quantiles = None
    if success_norms:
        qs = statistics.quantiles(success_norms, n=4) if len(success_norms) >= 4 else None
        norm_quantiles = {
            "n": len(success_norms),
            "min": min(success_norms),
            "p25": qs[0] if qs else None,
            "median": statistics.median(success_norms),
            "p75": qs[2] if qs else None,
            "max": max(success_norms),
        }

    eps_stats = None
    if epsilons:
        eps_stats = {
            "n": len(epsilons),
            "min": min(epsilons),
            "max": max(epsilons),
            "mean": statistics.mean(epsilons),
        }

    file_size = os.path.getsize(path)
    file_mtime = os.path.getmtime(path)

    # ── Current epoch boundary ──────────────────────────────────────────────
    # Most recent successful set_weights marks the end of the previous epoch.
    # Everything after it belongs to the current (incomplete) epoch.
    last_setweights_ts: str | None = None
    for evt in set_weight_events:
        if evt["result"] == "success" and evt["ts"]:
            last_setweights_ts = evt["ts"]

    # ── Aggregate the current epoch's challenges into a leaderboard ─────────
    current_epoch_results: dict[int, dict] = defaultdict(
        lambda: {
            "n": 0,
            "succ": 0,
            "score_sum": 0.0,
            "norms": [],
            "rmses": [],
            "psnrs": [],
            "rts": [],
            "last_reason": None,
            "last_score": 0.0,
            "last_ts": None,
        }
    )
    current_epoch_challenges = 0
    for chal in challenges_detailed.values():
        # Treat the whole file as the current epoch if no set_weights has fired yet.
        if last_setweights_ts is not None and (chal["ts"] is None or chal["ts"] <= last_setweights_ts):
            continue
        current_epoch_challenges += 1
        for r in chal["results"]:
            d = current_epoch_results[r["uid"]]
            d["n"] += 1
            d["score_sum"] += r["score"]
            d["last_reason"] = r["reason"]
            d["last_score"] = r["score"]
            d["last_ts"] = r["ts"]
            if r["reason"] == "success":
                d["succ"] += 1
                d["norms"].append(r["norm"])
                d["rmses"].append(r["rmse"])
                d["psnrs"].append(r["psnr_db"])
                if r.get("response_time_ms") is not None:
                    d["rts"].append(r["response_time_ms"])

    current_epoch_leaderboard: list[dict] = []
    for uid, d in current_epoch_results.items():
        if d["n"] == 0:
            continue
        current_epoch_leaderboard.append(
            {
                "uid": uid,
                "samples": d["n"],
                "success": d["succ"],
                "success_rate": d["succ"] / d["n"],
                "avg_score": d["score_sum"] / d["n"],
                "avg_norm_success": statistics.mean(d["norms"]) if d["norms"] else None,
                "avg_rmse_success": statistics.mean(d["rmses"]) if d["rmses"] else None,
                "avg_psnr_success": statistics.mean(d["psnrs"]) if d["psnrs"] else None,
                "avg_rt_ms_success": statistics.mean(d["rts"]) if d["rts"] else None,
                "last_reason": d["last_reason"],
                "last_score": d["last_score"],
                "last_ts": d["last_ts"],
            }
        )
    current_epoch_leaderboard.sort(key=lambda r: (-r["avg_score"], -r["samples"]))

    # ── Serialize challenges as an ordered list, newest first ───────────────
    challenges_list: list[dict] = []
    for chal in challenges_detailed.values():
        results = chal["results"]
        success = sum(1 for r in results if r["reason"] == "success")
        avg = sum(r["score"] for r in results) / len(results) if results else 0.0
        max_score = max((r["score"] for r in results), default=0.0)
        avg_norm = (
            statistics.mean([r["norm"] for r in results if r["reason"] == "success"])
            if any(r["reason"] == "success" for r in results)
            else None
        )
        avg_rt = (
            statistics.mean(
                [r["response_time_ms"] for r in results if r["response_time_ms"] is not None]
            )
            if any(r["response_time_ms"] is not None for r in results)
            else None
        )
        challenges_list.append(
            {
                "task_id": chal["task_id"],
                "block": chal["block"],
                "prompt": chal["prompt"],
                "epsilon": chal["epsilon"],
                "ts": chal["ts"],
                "total_responses": len(results),
                "success_count": success,
                "avg_score": avg,
                "max_score": max_score,
                "avg_norm_success": avg_norm,
                "avg_response_time_ms": avg_rt,
                "results": results,
            }
        )
    # Sort newest first by block (block is monotonically increasing).
    challenges_list.sort(key=lambda c: (c["block"] is None, -(c["block"] or 0)))

    # Surface the latest authoritative validator ranking as a top-level field.
    latest_weight_event = weight_events[-1] if weight_events else None
    validator_leaderboard: list[dict] = []
    if latest_weight_event:
        for entry in latest_weight_event["ranks"]:
            # Enrich with the window-side stats if we have any samples for this uid.
            d = per_uid.get(entry["uid"])
            enriched = dict(entry)
            enriched["samples_window"] = d["n"] if d else 0
            enriched["last_reason_window"] = d["last_reason"] if d else None
            enriched["avg_norm_success"] = (
                statistics.mean(d["norms"]) if d and d["norms"] else None
            )
            enriched["avg_rmse_success"] = (
                statistics.mean(d["rmses"]) if d and d["rmses"] else None
            )
            enriched["avg_psnr_success"] = (
                statistics.mean(d["psnrs"]) if d and d["psnrs"] else None
            )
            validator_leaderboard.append(enriched)
        # Already sorted by rank ascending in the log, but be explicit.
        validator_leaderboard.sort(key=lambda r: r["rank"])

    return {
        "ok": True,
        "path": path,
        "line_count": line_count,
        "file_size_bytes": file_size,
        "file_mtime": file_mtime,
        "last_line_ts": last_line_ts,
        "run_ids": sorted(run_ids),
        "restarts": restarts,
        "loops_count": len(loops),
        "loops": loops,
        "challenges": dict(challenges),
        "epsilon": eps_stats,
        "reasons": dict(reasons),
        "statuses": {str(k): v for k, v in statuses.items()},
        "norm_quantiles": norm_quantiles,
        "miner_count": len(per_uid),
        "leaderboard": leaderboard,
        "validator_leaderboard": validator_leaderboard,
        "latest_weight_event": (
            {
                "ts": latest_weight_event["ts"],
                "eligible": latest_weight_event["eligible"],
                "distributed": latest_weight_event["distributed"],
                "top5": latest_weight_event["top5"],
                "ranks_count": len(latest_weight_event["ranks"]),
            }
            if latest_weight_event
            else None
        ),
        "weight_events_count": len(weight_events),
        "set_weight_events": set_weight_events[-10:],  # most recent 10
        "current_epoch": {
            "since_ts": last_setweights_ts,
            "challenges_count": current_epoch_challenges,
            "leaderboard": current_epoch_leaderboard,
        },
        "challenges_detailed": challenges_list,
    }


# ---------------------------------------------------------------------------
# Caching layer — stats are computed at most once every CACHE_TTL seconds.
# ---------------------------------------------------------------------------

CACHE_TTL = 3.0  # seconds


class StatsCache:
    def __init__(self, path: str, ttl: float = CACHE_TTL) -> None:
        self.path = path
        self.ttl = ttl
        self._lock = threading.Lock()
        self._stats: dict | None = None
        self._fetched_at: float = 0.0

    def get(self, force: bool = False) -> dict:
        now = time.monotonic()
        with self._lock:
            if (
                not force
                and self._stats is not None
                and (now - self._fetched_at) < self.ttl
            ):
                return self._stats
            self._stats = parse_log(self.path)
            self._fetched_at = now
            self._stats["_cache_age_seconds"] = 0.0
            self._stats["_generated_at"] = time.time()
            return self._stats


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Perturb Validator Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {
      --bg: #0f1419;
      --panel: #1a2028;
      --panel-2: #232c37;
      --border: #2c3a47;
      --text: #d8e0ea;
      --muted: #7a8b9d;
      --accent: #4fc3f7;
      --good: #66bb6a;
      --warn: #ffa726;
      --bad: #ef5350;
      --dim: #455a64;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text); font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 13px; }
    header { padding: 14px 22px; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
    header h1 { margin: 0; font-size: 16px; font-weight: 600; }
    header .meta { color: var(--muted); font-size: 12px; }
    header .meta b { color: var(--text); font-weight: 500; }
    button { background: var(--panel-2); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 6px 12px; font: inherit; cursor: pointer; }
    button:hover { background: var(--border); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; padding: 14px 22px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px; overflow: hidden; }
    .card h2 { margin: 0 0 10px 0; font-size: 13px; font-weight: 600; color: var(--accent); letter-spacing: 0.4px; text-transform: uppercase; }
    .span-3 { grid-column: span 3; } .span-4 { grid-column: span 4; } .span-6 { grid-column: span 6; } .span-8 { grid-column: span 8; } .span-12 { grid-column: span 12; }
    @media (max-width: 1100px) {
      .span-3, .span-4, .span-6, .span-8 { grid-column: span 12; }
    }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--panel-2); white-space: nowrap; }
    th { text-align: right; color: var(--muted); font-weight: 500; }
    th:first-child, td:first-child { text-align: left; }
    tr:hover td { background: var(--panel-2); }
    .num { font-variant-numeric: tabular-nums; }
    table.sortable thead th { cursor: pointer; user-select: none; position: relative; padding-right: 16px; }
    table.sortable thead th:hover { color: var(--text); }
    table.sortable thead th.sort-asc::after  { content: " ▲"; color: var(--accent); position: absolute; right: 4px; }
    table.sortable thead th.sort-desc::after { content: " ▼"; color: var(--accent); position: absolute; right: 4px; }
    table.clickable tbody tr:not(.detail-row) { cursor: pointer; }
    table.clickable tbody tr.selected td { background: rgba(79,195,247,0.12); }
    table.clickable tbody tr.selected td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
    table.clickable tbody tr.detail-row > td { background: var(--bg); padding: 14px 16px; border-bottom: 2px solid var(--border); cursor: default; }
    table.clickable tbody tr.detail-row:hover > td { background: var(--bg); }
    .detail-meta { color: var(--muted); margin-bottom: 8px; font-size: 12px; }
    .detail-meta b { color: var(--text); font-weight: 500; }
    .bar { height: 14px; background: var(--panel-2); border-radius: 3px; position: relative; overflow: hidden; }
    .bar > div { height: 100%; background: var(--accent); }
    .bar.good > div { background: var(--good); }
    .bar.warn > div { background: var(--warn); }
    .bar.bad  > div { background: var(--bad); }
    .kv { display: grid; grid-template-columns: 130px 1fr; row-gap: 4px; column-gap: 12px; font-size: 12px; }
    .kv .k { color: var(--muted); }
    .pill { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; background: var(--panel-2); border: 1px solid var(--border); color: var(--text); }
    .pill.good { background: rgba(102,187,106,0.15); border-color: var(--good); color: var(--good); }
    .pill.warn { background: rgba(255,167,38,0.15); border-color: var(--warn); color: var(--warn); }
    .pill.bad  { background: rgba(239,83,80,0.15);  border-color: var(--bad);  color: var(--bad); }
    .muted { color: var(--muted); }
    .err { color: var(--bad); padding: 14px 22px; }
    .reason-row { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 12px; }
    .reason-row .label { width: 200px; }
    .reason-row .count { width: 60px; text-align: right; color: var(--muted); }
    .reason-row .bar { flex: 1; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Perturb Validator Dashboard</h1>
      <div class="meta" id="meta">loading…</div>
    </div>
    <div>
      <span class="muted" id="next-refresh">refresh in 30s</span>
      &nbsp;
      <button id="refresh">Refresh now</button>
    </div>
  </header>
  <div id="root">
    <div class="muted" style="padding: 14px 22px;">Loading…</div>
  </div>

<script>
const REFRESH_MS = 30000;
let timer = null;
let countdown = null;
let nextRefreshAt = 0;

function fmtNum(n, d=4) {
  if (n === null || n === undefined) return "—";
  if (typeof n !== "number") return String(n);
  return n.toFixed(d);
}
function pct(n) {
  if (n === null || n === undefined) return "—";
  return (100*n).toFixed(1) + "%";
}
function bytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1024*1024) return (n/1024).toFixed(1) + " KB";
  return (n/1024/1024).toFixed(2) + " MB";
}
function ago(epochS) {
  if (!epochS) return "—";
  const diff = Date.now()/1000 - epochS;
  if (diff < 60) return Math.floor(diff) + "s ago";
  if (diff < 3600) return Math.floor(diff/60) + "m ago";
  return Math.floor(diff/3600) + "h ago";
}
function reasonClass(reason) {
  if (reason === "success") return "good";
  if (reason === "response_missing_or_status_error") return "bad";
  return "warn";
}
function statusClass(code) {
  if (code === 200) return "good";
  if (code === 503 || code === 408) return "bad";
  return "warn";
}

function renderMeta(s) {
  const runs = s.run_ids && s.run_ids.length ? s.run_ids.join(", ") : "unknown";
  const lastTs = s.last_line_ts || "—";
  document.getElementById("meta").innerHTML =
    `<b>run_id</b>: ${runs}` +
    `  &middot;  <b>last log ts</b>: ${lastTs}` +
    `  &middot;  <b>file</b>: ${bytes(s.file_size_bytes)} (${s.line_count} lines)`;
}

let challengeData = []; // populated each render so the click handler can look up by task_id
let selectedTaskId = null;
const CHALLENGE_COLSPAN = 12; // matches the number of <th> in the challenges table

function renderChallenges(challenges) {
  challengeData = challenges || [];
  if (challengeData.length === 0) {
    return `<div class='muted'>No challenges captured yet.</div>`;
  }
  let html = `<div class="muted" style="margin-bottom:8px">
    Click a row to expand per-miner results. ${challengeData.length} challenge(s) in the log window.
  </div>`;
  html += `<table class="sortable clickable" id="challenges-table"><thead><tr>
    <th>block</th><th>task id</th><th>ts</th><th>prompt</th><th>ε</th>
    <th>responses</th><th>succ</th><th>succ%</th>
    <th>avg score</th><th>max score</th><th>avg L∞ (succ)</th><th>avg RT (ms)</th>
  </tr></thead><tbody>`;
  for (const c of challengeData) {
    const sr = c.total_responses > 0 ? c.success_count / c.total_responses : 0;
    const selectedCls = (c.task_id === selectedTaskId) ? " selected" : "";
    html += `<tr class="${selectedCls.trim()}" data-task="${c.task_id}">
      <td data-sort="${c.block ?? 0}"><b>${c.block ?? "—"}</b></td>
      <td data-sort="${c.task_id || ''}" title="${c.task_id || ''}" style="font-size:11px; max-width: 260px; overflow:hidden; text-overflow: ellipsis;">${c.task_id || "—"}</td>
      <td data-sort="${c.ts || ''}">${c.ts || "—"}</td>
      <td data-sort="${c.prompt || ''}">${c.prompt || "—"}</td>
      <td class="num" data-sort="${c.epsilon ?? ''}">${fmtNum(c.epsilon, 4)}</td>
      <td class="num" data-sort="${c.total_responses}">${c.total_responses}</td>
      <td class="num" data-sort="${c.success_count}">${c.success_count}</td>
      <td class="num" data-sort="${sr}">${pct(sr)}</td>
      <td class="num" data-sort="${c.avg_score}">${c.avg_score.toFixed(4)}</td>
      <td class="num" data-sort="${c.max_score}">${c.max_score.toFixed(4)}</td>
      <td class="num" data-sort="${c.avg_norm_success ?? ''}">${fmtNum(c.avg_norm_success, 5)}</td>
      <td class="num" data-sort="${c.avg_response_time_ms ?? ''}">${c.avg_response_time_ms != null ? c.avg_response_time_ms.toFixed(0) : "—"}</td>
    </tr>`;
  }
  html += "</tbody></table>";
  return html;
}

// Returns an HTML <tr class="detail-row"> that should be inserted into the
// challenges table tbody immediately after the corresponding data row.
function renderChallengeDetailRow(taskId) {
  const c = challengeData.find(x => x.task_id === taskId);
  if (!c) return "";
  let inner = `<div class="detail-meta"><b>Challenge</b> ${c.task_id}
    &middot; block <b>${c.block ?? "—"}</b>
    &middot; prompt <b>${c.prompt || "—"}</b>
    &middot; ε <b>${fmtNum(c.epsilon, 4)}</b>
    &middot; ts <b>${c.ts || "—"}</b>
    &middot; <b>${c.success_count}/${c.total_responses}</b> succeeded
  </div>`;
  inner += `<table class="sortable" id="detail-table"><thead><tr>
    <th>uid</th><th>status</th><th>score</th><th>reason</th>
    <th>L∞</th><th>RMSE</th><th>SSIM</th><th>PSNR</th><th>RT (ms)</th><th>proc</th>
  </tr></thead><tbody>`;
  // Default ordering: success first (sorted by score desc), then non-success.
  const rows = c.results.slice().sort((a, b) => {
    const ak = a.reason === "success" ? 1 : 0;
    const bk = b.reason === "success" ? 1 : 0;
    if (ak !== bk) return bk - ak;
    return b.score - a.score;
  });
  for (const r of rows) {
    inner += `<tr>
      <td data-sort="${r.uid}"><b>${r.uid}</b></td>
      <td data-sort="${r.status}"><span class="pill ${statusClass(r.status)}">${r.status}</span></td>
      <td class="num" data-sort="${r.score}">${r.score.toFixed(6)}</td>
      <td data-sort="${r.reason}"><span class="pill ${reasonClass(r.reason)}">${r.reason}</span></td>
      <td class="num" data-sort="${r.norm}">${fmtNum(r.norm, 5)}</td>
      <td class="num" data-sort="${r.rmse}">${fmtNum(r.rmse, 5)}</td>
      <td class="num" data-sort="${r.ssim}">${fmtNum(r.ssim, 4)}</td>
      <td class="num" data-sort="${r.psnr_db}">${fmtNum(r.psnr_db, 2)}</td>
      <td class="num" data-sort="${r.response_time_ms ?? ''}">${r.response_time_ms ?? "—"}</td>
      <td class="num" data-sort="${r.processed}">${r.processed}</td>
    </tr>`;
  }
  inner += "</tbody></table>";
  return `<tr class="detail-row" data-task="${taskId}"><td colspan="${CHALLENGE_COLSPAN}">${inner}</td></tr>`;
}

function renderValidatorLeaderboard(rows, latestEvent, topN) {
  if (!rows || rows.length === 0) {
    return `<div class='muted'>No weights_summary captured yet. The validator only emits the
      authoritative ranking when it calls <code>set_weights()</code> — typically once per
      tempo (~72 minutes). Once a set_weights event has flowed through the log, this card
      will populate.</div>`;
  }
  // Only show miners with avg100 > 0 (the rest are unranked / no rolling history).
  const ranked = rows.filter(r => r.avg100 > 0).slice(0, topN);
  const banner = latestEvent
    ? `<div class="muted" style="margin-bottom:8px">
         Last set_weights @ ${latestEvent.ts || "—"} &middot;
         eligible=${latestEvent.eligible} &middot;
         distributed=${latestEvent.distributed} &middot;
         ranked uids=${latestEvent.ranks_count}
       </div>`
    : "";
  let html = banner + `<table class="sortable" id="val-table"><thead><tr>
    <th>rank</th><th>uid</th><th>avg100</th><th>emission</th>
  </tr></thead><tbody>`;
  for (const r of ranked) {
    const pillCls = r.emission > 0 ? "good" : (r.rank <= 5 ? "warn" : "");
    html += `<tr>
      <td data-sort="${r.rank}"><span class="pill ${pillCls}">#${r.rank}</span></td>
      <td data-sort="${r.uid}"><b>${r.uid}</b></td>
      <td class="num" data-sort="${r.avg100}">${r.avg100.toFixed(6)}</td>
      <td class="num" data-sort="${r.emission}">${r.emission.toFixed(4)}</td>
    </tr>`;
  }
  html += "</tbody></table>";
  if (ranked.length === 0) {
    html += `<div class='muted' style="margin-top:8px">No miners with avg100 > 0 yet.</div>`;
  }
  return html;
}

function render(s) {
  if (!s.ok) {
    document.getElementById("root").innerHTML = `<div class="err">${s.error || "unknown error"}</div>`;
    document.getElementById("meta").textContent = "no data";
    return;
  }
  renderMeta(s);
  const topN = TOP_N_DEFAULT;
  const recentN = RECENT_LOOPS_DEFAULT;
  document.getElementById("root").innerHTML = `
    <div class="grid">
      <div class="card span-12">
        <h2>set_weights ranking</h2>
        ${renderValidatorLeaderboard(s.validator_leaderboard, s.latest_weight_event, topN)}
      </div>

      <div class="card span-12">
        <h2>Challenges (click a row for per-miner scores)</h2>
        ${renderChallenges(s.challenges_detailed || [])}
      </div>
    </div>
  `;
}

// Sortable tables — sort state is global and survives re-renders so 30s
// auto-refresh doesn't reset the user's chosen column/direction.
const sortState = {}; // { tableId: { col: number, dir: 'asc'|'desc' } }

function applySort(table) {
  const tid = table.id;
  const st = sortState[tid];
  if (!st) return;
  const idx = st.col;
  const tbody = table.querySelector("tbody");
  if (!tbody) return;
  // Only sort data rows; detail rows piggy-back on their parent data row.
  const dataRows = Array.from(tbody.querySelectorAll(":scope > tr:not(.detail-row)"));
  const detailByTask = {};
  Array.from(tbody.querySelectorAll(":scope > tr.detail-row")).forEach(d => {
    if (d.dataset.task) detailByTask[d.dataset.task] = d;
  });
  dataRows.sort((a, b) => {
    const av = a.cells[idx]?.dataset.sort ?? a.cells[idx]?.textContent ?? "";
    const bv = b.cells[idx]?.dataset.sort ?? b.cells[idx]?.textContent ?? "";
    const an = parseFloat(av);
    const bn = parseFloat(bv);
    let cmp;
    if (av === "" && bv !== "") cmp = -1;
    else if (bv === "" && av !== "") cmp = 1;
    else if (!isNaN(an) && !isNaN(bn) && /^-?[\d.eE+-]+$/.test(String(av)) && /^-?[\d.eE+-]+$/.test(String(bv))) {
      cmp = an - bn;
    } else {
      cmp = String(av).localeCompare(String(bv));
    }
    return st.dir === "asc" ? cmp : -cmp;
  });
  // Re-append in sorted order; a detail row follows its data row.
  for (const r of dataRows) {
    tbody.appendChild(r);
    const t = r.dataset.task;
    if (t && detailByTask[t]) tbody.appendChild(detailByTask[t]);
  }
  Array.from(table.querySelectorAll("thead th")).forEach((th, i) => {
    th.classList.remove("sort-asc", "sort-desc");
    if (i === idx) th.classList.add(st.dir === "asc" ? "sort-asc" : "sort-desc");
  });
}

function wireSortable(table) {
  const tid = table.id;
  if (!tid) return;
  Array.from(table.querySelectorAll("thead th")).forEach((th, idx) => {
    th.addEventListener("click", () => {
      const cur = sortState[tid];
      let dir;
      if (cur && cur.col === idx) {
        dir = cur.dir === "asc" ? "desc" : "asc";
      } else {
        // First click on a column: numeric columns sort descending first,
        // text columns ascending first.
        const sample = table.querySelector(`tbody tr td:nth-child(${idx+1})`);
        const v = sample ? (sample.dataset.sort ?? sample.textContent) : "";
        dir = (!isNaN(parseFloat(v)) && /^-?[\d.eE+-]+$/.test(String(v))) ? "desc" : "asc";
      }
      sortState[tid] = { col: idx, dir };
      applySort(table);
    });
  });
  if (sortState[tid]) applySort(table);
}

function expandChallengeRow(dataRow) {
  const taskId = dataRow.dataset.task;
  if (!taskId) return;
  const tbody = dataRow.parentNode;
  // Remove any other expanded detail row (single-select).
  Array.from(tbody.querySelectorAll(":scope > tr.detail-row")).forEach(d => d.remove());
  Array.from(tbody.querySelectorAll(":scope > tr.selected")).forEach(x => x.classList.remove("selected"));
  dataRow.classList.add("selected");
  dataRow.insertAdjacentHTML("afterend", renderChallengeDetailRow(taskId));
  // Wire sort on the freshly inserted inner table.
  const det = document.getElementById("detail-table");
  if (det) wireSortable(det);
}

function collapseChallengeRow(dataRow) {
  const tbody = dataRow.parentNode;
  Array.from(tbody.querySelectorAll(":scope > tr.detail-row")).forEach(d => d.remove());
  dataRow.classList.remove("selected");
}

function wireChallengeClicks() {
  const t = document.getElementById("challenges-table");
  if (!t) return;
  Array.from(t.querySelectorAll("tbody tr:not(.detail-row)")).forEach(tr => {
    tr.addEventListener("click", () => {
      const taskId = tr.dataset.task;
      if (!taskId) return;
      const next = tr.nextElementSibling;
      const isExpanded = next && next.classList.contains("detail-row") && next.dataset.task === taskId;
      if (isExpanded) {
        collapseChallengeRow(tr);
        selectedTaskId = null;
      } else {
        expandChallengeRow(tr);
        selectedTaskId = taskId;
      }
    });
  });
}

function rewireAll() {
  document.querySelectorAll("table.sortable").forEach(wireSortable);
  wireChallengeClicks();
  // If a challenge was expanded before the refresh, re-insert its detail row.
  if (selectedTaskId) {
    const t = document.getElementById("challenges-table");
    if (t) {
      const row = t.querySelector(`tbody tr[data-task="${selectedTaskId}"]:not(.detail-row)`);
      if (row) {
        expandChallengeRow(row);
      } else {
        selectedTaskId = null;
      }
    }
  }
}

async function fetchStats(force) {
  const btn = document.getElementById("refresh");
  btn.disabled = true;
  btn.textContent = "Refreshing…";
  try {
    const r = await fetch("/api/stats" + (force ? "?force=1" : ""));
    const j = await r.json();
    render(j);
    rewireAll();
  } catch (e) {
    document.getElementById("root").innerHTML = `<div class="err">Fetch failed: ${e}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Refresh now";
    nextRefreshAt = Date.now() + REFRESH_MS;
  }
}

function tickCountdown() {
  const sec = Math.max(0, Math.ceil((nextRefreshAt - Date.now()) / 1000));
  document.getElementById("next-refresh").textContent = `refresh in ${sec}s`;
}

document.getElementById("refresh").addEventListener("click", () => fetchStats(true));
fetchStats(false);
timer = setInterval(() => fetchStats(false), REFRESH_MS);
countdown = setInterval(tickCountdown, 250);

const TOP_N_DEFAULT = __TOP_N__;
const RECENT_LOOPS_DEFAULT = __RECENT_LOOPS__;
</script>
</body>
</html>
"""


def make_handler(cache: StatsCache, top_n: int, recent_loops: int):
    rendered_index = (
        INDEX_HTML.replace("__TOP_N__", str(top_n)).replace(
            "__RECENT_LOOPS__", str(recent_loops)
        )
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            # Quieter default logging.
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self._send(200, rendered_index, "text/html; charset=utf-8")
                return

            if path == "/api/stats":
                force = "force=1" in (parsed.query or "")
                t0 = time.monotonic()
                stats = cache.get(force=force)
                stats["_cache_age_seconds"] = round(
                    time.monotonic() - cache._fetched_at, 3
                )
                stats["_serve_ms"] = round((time.monotonic() - t0) * 1000, 1)
                body = json.dumps(stats, default=str).encode("utf-8")
                self._send(200, body, "application/json")
                return

            if path == "/healthz":
                self._send(200, b'{"ok":true}', "application/json")
                return

            self._send(404, b"not found", "text/plain")

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description="Perturb validator dashboard")
    p.add_argument("--log", default="wandb_logs/uid0.log", help="Path to the validator log file")
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p.add_argument("--port", default=8800, type=int, help="Bind port (default 8800)")
    p.add_argument("--top", default=20, type=int, help="Top-N miners shown in leaderboard")
    p.add_argument("--recent-loops", default=20, type=int, help="Recent loop_summary rows shown")
    p.add_argument("--cache-ttl", default=3.0, type=float, help="Stats cache TTL (seconds)")
    args = p.parse_args()

    log_path = os.path.abspath(args.log)
    cache = StatsCache(log_path, ttl=args.cache_ttl)
    handler_cls = make_handler(cache, args.top, args.recent_loops)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(
        f"Perturb dashboard listening on http://{args.host}:{args.port}  "
        f"(log={log_path}, cache_ttl={args.cache_ttl}s)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down…", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
