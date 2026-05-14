#!/usr/bin/env python3
"""
Download W&B console logs for Perturb validators, keyed by UID.

The Perturb validator (perturbnet/neurons/validator.py) builds its W&B run name
as `{YYYYmmdd-HHMMSS}-uid{N}`, where N is the validator's UID in the metagraph.
This script:

Step 1 — Fetch UID -> run mapping
    Queries the project run table (GraphQL) and parses the `-uid{N}` suffix
    from each run's displayName. For each UID we keep the LATEST run by
    createdAt, so a crashed-and-restarted validator always maps to its
    currently-active run.

Step 2 — Download logs per UID (in parallel)
    All UIDs run concurrently (--workers, default 8). Within each run, pages
    are fetched sequentially (cursor-based). Saved to <output_dir>/uid{N}.log.
    Incremental: state is keyed by UID; if the run_id changes (new run
    after crash), the cursor state is reset and a restart banner is appended.

Single-run mode (--run RUN_ID)
    Skips the mapping step and downloads one run, saved as <run_id>.log.

Watch mode (--watch)
    Streams logs continuously, re-fetching the mapping each cycle so that
    validators which crash + restart automatically get followed onto the new
    run, with a restart banner inserted into the same per-UID log file.

Usage:
    python download_run_logs.py                 # follow UIDS list, 8 workers, one-shot
    python download_run_logs.py --watch         # follow UIDS list, stream forever
    python download_run_logs.py --watch --workers 16
    python download_run_logs.py --fresh         # ignore saved cursor state
    python download_run_logs.py --run abcd1234  # single run by id
    python download_run_logs.py --list-only     # print uid->run map and exit
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Defaults — point at the Perturb W&B project
# ---------------------------------------------------------------------------

DEFAULT_ENTITY     = "perturb-ai"
DEFAULT_PROJECT    = "perturb-validator"
DEFAULT_OUTPUT_DIR = "wandb_logs"
DEFAULT_PAGE       = 1000
DEFAULT_PAGE_SIZE  = 1000
DEFAULT_WORKERS    = 8
DEFAULT_MAPPING_PAGES = 1   # pages to fetch when building uid->run map (100 runs/page)
PAGE_DELAY    = 0.1         # seconds between log pages within one run
POLL_INTERVAL = 30          # seconds between polls in watch mode when caught up

GRAPHQL_URL = "https://api.wandb.ai/graphql"
SEPARATOR   = "=" * 80

# ---------------------------------------------------------------------------
# Validator filter — edit these to control which validators are fetched
# ---------------------------------------------------------------------------

# If non-empty, only download logs for these UIDs (exact match).
# Leave empty [] to download every active validator run found in the project.
UIDS: list[int] = [
    0,
]

# If True, skip runs whose W&B state is not "running".
# Set to False to also include crashed / finished runs in the mapping step.
ONLY_RUNNING: bool = True

# ---------------------------------------------------------------------------
# Run name parsing
# ---------------------------------------------------------------------------
#
# Perturb validator constructs its run name in validator.py as:
#     f"{time.strftime('%Y%m%d-%H%M%S')}-uid{uid_suffix}"
# We extract the trailing "-uid<digits>" group as the validator UID.

RUN_NAME_UID_RE = re.compile(r"-uid(\d+)\b")

# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

# List all runs in a project (paginated), returning id + displayName + state.
RUNS_LIST_QUERY = """
query ProjectRuns($entity: String!, $project: String!, $first: Int!, $after: String) {
  project(name: $project, entityName: $entity) {
    runs(first: $first, after: $after, order: "-created_at") {
      edges {
        node {
          name
          displayName
          createdAt
          state
        }
        cursor
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""

# Fetch log lines for a single run (paginated).
LOGS_QUERY = """
query RunLogs($entity: String!, $project: String!, $run: String!, $after: String) {
  project(name: $project, entityName: $entity) {
    run(name: $run) {
      name
      displayName
      logLines(first: %(page_size)s, after: $after) {
        edges {
          node {
            line
            level
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
}
"""

# ---------------------------------------------------------------------------
# ANSI
# ---------------------------------------------------------------------------

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mGKHFJA-Z]")

def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)

# ---------------------------------------------------------------------------
# GraphQL helper
# ---------------------------------------------------------------------------

def gql(query: str, variables: dict, retries: int = 5) -> dict:
    delay = 2.0
    for attempt in range(retries):
        try:
            r = requests.post(
                GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if r.status_code == 429:
                wait = delay * (2 ** attempt)
                print(f"  Rate-limited — waiting {wait:.0f}s…", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL errors: {data['errors']}")
            return data
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = delay * (2 ** attempt)
            print(f"  Request error ({exc}) — retrying in {wait:.0f}s…", flush=True)
            time.sleep(wait)
    raise RuntimeError("GraphQL request failed after all retries")

# ---------------------------------------------------------------------------
# UID -> Run mapping
# ---------------------------------------------------------------------------

def _extract_uid(run_node: dict) -> int | None:
    """Parse the trailing -uid<N> token out of a run's displayName."""
    display = (run_node.get("displayName") or "").strip()
    if not display:
        return None
    m = RUN_NAME_UID_RE.search(display)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def fetch_uid_run_mapping(
    entity: str,
    project: str,
    runs_per_page: int = 100,
    max_pages: int = DEFAULT_MAPPING_PAGES,
) -> dict[int, dict]:
    """
    Query the project run table and return:
        { uid: {"run_id": ..., "display_name": ..., "state": ..., "created_at": ...} }

    Runs are fetched newest-first; the first occurrence of each UID wins
    (= the currently active run after any crash/restart).

    Filters applied:
      - ONLY_RUNNING: skip runs whose state != "running"
      - UIDS:         skip runs whose UID is not in the whitelist (if non-empty)

    Stops early once all target UIDs are found, or after max_pages pages.
    """
    target_set = set(UIDS)  # empty = accept all
    filters_desc = []
    if ONLY_RUNNING:
        filters_desc.append("state=running")
    if target_set:
        filters_desc.append(f"{len(target_set)} uid(s) whitelisted")
    filter_str = ", ".join(filters_desc) if filters_desc else "no filters"

    print(
        f"Fetching run table for {entity}/{project} "
        f"(max {max_pages} page(s), {runs_per_page} runs/page, {filter_str})…",
        flush=True,
    )

    uid_map: dict[int, dict] = {}
    cursor = None
    page = 0

    while True:
        page += 1
        variables: dict = {
            "entity": entity,
            "project": project,
            "first": runs_per_page,
        }
        if cursor:
            variables["after"] = cursor

        data = gql(RUNS_LIST_QUERY, variables)
        runs_data = data["data"]["project"]["runs"]
        edges = runs_data["edges"]
        page_info = runs_data["pageInfo"]

        for edge in edges:
            node = edge["node"]
            run_id = node.get("name")  # W&B uses 'name' for the short run ID
            if not run_id:
                continue

            state = node.get("state", "unknown")
            if ONLY_RUNNING and state != "running":
                continue

            uid = _extract_uid(node)
            if uid is None:
                continue

            if target_set and uid not in target_set:
                continue

            # Keep only the first (latest) run per UID
            if uid not in uid_map:
                uid_map[uid] = {
                    "run_id":       run_id,
                    "display_name": node.get("displayName", run_id),
                    "state":        state,
                    "created_at":   node.get("createdAt", ""),
                }

        found = len(uid_map)
        print(
            f"  Page {page}/{max_pages}: {len(edges)} runs scanned, {found} uid(s) matched"
            f"  hasNextPage={page_info['hasNextPage']}",
            flush=True,
        )

        # Early stop if we've already found every UID we care about
        if target_set and uid_map.keys() >= target_set:
            print("  All target UIDs found — stopping early.")
            break

        if not page_info["hasNextPage"] or page >= max_pages:
            break

        cursor = page_info["endCursor"]
        time.sleep(0.3)

    return uid_map

# ---------------------------------------------------------------------------
# State helpers (keyed by UID so crashes don't break incremental fetch)
# ---------------------------------------------------------------------------

def state_path(output_dir: str, uid: int | None, run_id: str) -> str:
    if uid is not None:
        return os.path.join(output_dir, f".uid{uid}.state.json")
    return os.path.join(output_dir, f".{run_id}.state.json")


def load_state(output_dir: str, uid: int | None, run_id: str) -> dict:
    path = state_path(output_dir, uid, run_id)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_state(
    output_dir: str,
    uid: int | None,
    run_id: str,
    cursor: str | None,
    page: int,
    total_lines: int,
) -> None:
    path = state_path(output_dir, uid, run_id)
    with open(path, "w") as f:
        json.dump({
            "run_id":      run_id,
            "cursor":      cursor,
            "page":        page,
            "total_lines": total_lines,
            "updated_at":  datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

# ---------------------------------------------------------------------------
# Core log download
# ---------------------------------------------------------------------------

def iter_log_pages(
    entity: str,
    project: str,
    run_id: str,
    page_size: int = DEFAULT_PAGE,
    start_cursor: str | None = None,
    start_page: int = 0,
    log=print,
):
    """
    Generator — yields one page at a time as:
        (page_lines, display_name, cursor, page_num, has_next)
    Caller can flush to disk and save state after each yield.
    """
    query = LOGS_QUERY % {"page_size": page_size}

    cursor = start_cursor
    display_name = run_id
    page = start_page

    while True:
        page += 1
        variables: dict = {"entity": entity, "project": project, "run": run_id}
        if cursor:
            variables["after"] = cursor

        data = gql(query, variables)
        run_data = data["data"]["project"]["run"]
        display_name = run_data.get("displayName", run_id)
        log_data = run_data["logLines"]
        edges = log_data["edges"]
        page_info = log_data["pageInfo"]

        # Dedup only within this page's edges (cursor handles cross-page dedup)
        seen_this_page: set[str] = set()
        page_lines = []
        for e in edges:
            line = strip_ansi(e["node"]["line"])
            if line not in seen_this_page:
                seen_this_page.add(line)
                page_lines.append(line)

        raw      = len(edges)
        new      = len(page_lines)
        has_next = page_info["hasNextPage"]

        # W&B returns endCursor=None when hasNextPage=False.
        # Keep the last non-None cursor so the next poll resumes correctly
        # instead of rewinding to the beginning.
        api_cursor = page_info["endCursor"]
        if api_cursor is not None:
            cursor = api_cursor

        dedup_note = f"  ({raw - new} intra-page dupes dropped)" if raw != new else ""
        log(f"  page {page}: {raw} edges → {new} lines written  hasNextPage={has_next}{dedup_note}")

        yield page_lines, display_name, cursor, page, has_next

        if not has_next:
            break

        time.sleep(PAGE_DELAY)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_seen_lines(output_path: str) -> set[str]:
    """Seed the dedup set with every line already present in an existing log file."""
    seen: set[str] = set()
    if not os.path.exists(output_path):
        return seen
    try:
        with open(output_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line:
                    seen.add(line)
    except OSError:
        pass
    return seen


# ---------------------------------------------------------------------------
# Download one validator's log
# ---------------------------------------------------------------------------

def download_validator_log(
    entity: str,
    project: str,
    uid: int,
    run_id: str,
    display_name: str,
    output_dir: str,
    page_size: int,
    fresh: bool,
) -> str:
    """One-shot: download logs for one validator UID, flushing to disk after every page."""
    tag = f"uid{uid}"
    def log(msg: str) -> None:
        print(f"[{tag}] {msg}", flush=True)

    output  = os.path.join(output_dir, f"uid{uid}.log")
    run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}/logs"

    state = {} if fresh else load_state(output_dir, uid, run_id)

    # If the run changed (crash → new run), start fresh for this uid
    saved_run_id = state.get("run_id")
    if saved_run_id and saved_run_id != run_id:
        log(f"run changed ({saved_run_id} → {run_id}) — re-downloading from scratch")
        state = {}
        if os.path.exists(output):
            os.remove(output)

    start_cursor = state.get("cursor")
    start_page   = state.get("page", 0)
    total        = state.get("total_lines", 0)
    is_resume    = bool(start_cursor)
    first_page   = True

    # Seed dedup from the existing file so a restart never re-appends
    # lines already written (handles W&B cursor drift across sessions).
    seen_lines = _load_seen_lines(output)
    if seen_lines:
        log(f"  seeded {len(seen_lines)} already-written lines from existing file")

    log(f"{'resuming from page ' + str(start_page + 1) if is_resume else 'starting download'}  run={run_id}")

    last_good_cursor = start_cursor
    last_good_page   = start_page

    for page_lines, dn, cursor, page, _has_next in iter_log_pages(
        entity, project, run_id, page_size, start_cursor, start_page, log=log,
    ):
        fresh_lines = [l for l in page_lines if l not in seen_lines]
        n_drift = len(page_lines) - len(fresh_lines)

        if n_drift and not fresh_lines:
            log(
                f"  drift detected on page {page} ({n_drift} lines all already seen) "
                f"— stopping, keeping cursor at page {last_good_page}"
            )
            break

        if n_drift:
            log(f"  {n_drift} cross-page duplicate line(s) dropped on page {page}")

        if not fresh_lines:
            save_state(output_dir, uid, run_id, cursor, page, total)
            continue

        if not is_resume and first_page:
            with open(output, "w", encoding="utf-8") as f:
                f.write("Perturb W&B Console Log\n")
                f.write(f"UID      : {uid}\n")
                f.write(f"Run      : {dn}  [{run_id}]\n")
                f.write(f"URL      : {run_url}\n")
                f.write(f"Exported : {datetime.now(timezone.utc).isoformat()}\n")
                f.write(SEPARATOR + "\n\n")

        first_page = False

        with open(output, "a", encoding="utf-8") as f:
            f.write("\n".join(fresh_lines) + "\n")

        seen_lines.update(fresh_lines)
        total += len(fresh_lines)
        last_good_cursor = cursor
        last_good_page   = page

        save_state(output_dir, uid, run_id, last_good_cursor, last_good_page, total)

    summary = f"done — {total} total lines → {output}"
    log(summary)
    return summary


# ---------------------------------------------------------------------------
# Watch mode — stream logs forever, handling crash/restart
# ---------------------------------------------------------------------------

def watch_validator_log(
    entity: str,
    project: str,
    uid: int,
    output_dir: str,
    page_size: int,
) -> None:
    """
    Stream logs for one validator UID indefinitely.

    Each cycle:
      1. Re-fetch the uid->run mapping to discover the current run for this UID.
         If the previous run crashed, the validator's restart created a new run
         with the same `-uid{N}` suffix, and this mapping will pick it up.
      2. If the run_id changed since last cycle, append a "RUN RESTARTED" banner
         to the same per-UID log file and reset the cursor.
      3. Stream any new pages until caught up (hasNextPage=False).
      4. Sleep POLL_INTERVAL seconds, then repeat.

    All log output goes into a single uid{N}.log per UID, continuously appended.
    """
    import random

    tag = f"uid{uid}"
    def log(msg: str) -> None:
        print(f"[{tag}] {msg}", flush=True)

    output = os.path.join(output_dir, f"uid{uid}.log")

    log(f"watch mode started — polling every {POLL_INTERVAL}s")

    # Cross-cycle dedup: never bounded — holds every line written so far.
    # W&B live-run cursors are unstable and can rewind by many pages;
    # an unbounded set is the only reliable guard against re-writing old lines.
    seen_across_cycles: set[str] = _load_seen_lines(output)
    if seen_across_cycles:
        log(f"seeded {len(seen_across_cycles)} already-written lines from existing file")

    # Small random jitter so all threads don't hit the mapping API simultaneously.
    time.sleep(random.uniform(0, 3))

    while True:
        # ── Step 1: resolve current run for this UID ─────────────────────────
        try:
            mapping = fetch_uid_run_mapping(entity, project)
        except Exception as exc:
            log(f"mapping refresh failed ({exc}), retrying in {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        info = mapping.get(uid)
        if not info:
            log(f"no active run found for uid={uid}, retrying in {POLL_INTERVAL}s…")
            time.sleep(POLL_INTERVAL)
            continue

        run_id  = info["run_id"]
        run_url = f"https://wandb.ai/{entity}/{project}/runs/{run_id}/logs"

        # ── Step 2: load state; detect run change ───────────────────────────
        state        = load_state(output_dir, uid, run_id)
        saved_run_id = state.get("run_id")

        if saved_run_id and saved_run_id != run_id:
            ts = datetime.now(timezone.utc).isoformat()
            log(f"run restarted: {saved_run_id} → {run_id}")
            with open(output, "a", encoding="utf-8") as f:
                f.write(f"\n{SEPARATOR}\n")
                f.write(f"*** RUN RESTARTED at {ts} ***\n")
                f.write(f"    old run : {saved_run_id}\n")
                f.write(f"    new run : {run_id}  ({run_url})\n")
                f.write(f"{SEPARATOR}\n\n")
            state = {}  # fresh cursor for new run
            # Don't clear seen_across_cycles — keep old run's lines so the
            # restart banner and any overlapping lines are never re-written.

        start_cursor = state.get("cursor")
        start_page   = state.get("page", 0)
        total        = state.get("total_lines", 0)
        is_new_file  = not os.path.exists(output)
        first_page   = True

        log(
            f"resuming from page {start_page + 1}  "
            f"cursor={'<none — full rescan!>' if not start_cursor else start_cursor[:20] + '…'}"
        )

        # ── Step 3: stream pages until caught up ────────────────────────────
        new_this_cycle  = 0
        drift_discarded = 0
        last_good_cursor = start_cursor
        last_good_page   = start_page
        try:
            for page_lines, dn, cursor, page, _has_next in iter_log_pages(
                entity, project, run_id, page_size, start_cursor, start_page, log=log,
            ):
                fresh_lines = [l for l in page_lines if l not in seen_across_cycles]
                n_drift = len(page_lines) - len(fresh_lines)
                drift_discarded += n_drift

                if fresh_lines:
                    if is_new_file and first_page:
                        with open(output, "w", encoding="utf-8") as f:
                            f.write("Perturb W&B Console Log\n")
                            f.write(f"UID      : {uid}\n")
                            f.write(f"Run      : {dn}  [{run_id}]\n")
                            f.write(f"URL      : {run_url}\n")
                            f.write(f"Started  : {datetime.now(timezone.utc).isoformat()}\n")
                            f.write(SEPARATOR + "\n\n")
                        is_new_file = False

                    first_page = False
                    with open(output, "a", encoding="utf-8") as f:
                        f.write("\n".join(fresh_lines) + "\n")
                    total          += len(fresh_lines)
                    new_this_cycle += len(fresh_lines)
                    seen_across_cycles.update(fresh_lines)

                    last_good_cursor = cursor
                    last_good_page   = page
                    save_state(output_dir, uid, run_id, last_good_cursor, last_good_page, total)

                elif page_lines and n_drift == len(page_lines):
                    log(
                        f"  drift detected on page {page} ({n_drift} lines all already seen) "
                        f"— stopping cycle early, keeping cursor at page {last_good_page}"
                    )
                    break

        except Exception as exc:
            log(f"error during fetch ({exc}), will retry next cycle")

        drift_note = f"  ({drift_discarded} cursor-drift dupes discarded)" if drift_discarded else ""
        log(
            f"caught up — {new_this_cycle} new lines this cycle, "
            f"{total} total. Sleeping {POLL_INTERVAL}s…{drift_note}"
        )
        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--entity",    default=DEFAULT_ENTITY,  help="W&B entity (user/org)")
    p.add_argument("--project",   default=DEFAULT_PROJECT, help="W&B project name")
    p.add_argument("--run",       default=None,
                   help="Single run ID (skip mapping, download this run only)")
    p.add_argument("--page-size", default=DEFAULT_PAGE_SIZE, type=int,
                   help="Log lines per GraphQL request (max 1000)")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore saved state and re-download from scratch")
    p.add_argument("--list-only", action="store_true",
                   help="Print uid->run mapping and exit without downloading")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory to save log files (default: wandb_logs)")
    p.add_argument("--workers", default=DEFAULT_WORKERS, type=int,
                   help=f"Parallel download workers (default: {DEFAULT_WORKERS})")
    p.add_argument("--mapping-pages", default=DEFAULT_MAPPING_PAGES, type=int,
                   help=f"Pages to fetch when building uid->run map, 100 runs/page (default: {DEFAULT_MAPPING_PAGES})")
    p.add_argument("--watch", action="store_true",
                   help=f"Stream logs continuously, polling every {POLL_INTERVAL}s. "
                        "Handles run restarts — appends to the same file forever.")
    p.add_argument("--poll-interval", default=POLL_INTERVAL, type=int,
                   help=f"Seconds between polls in watch mode (default: {POLL_INTERVAL})")
    p.add_argument("--uid", action="append", type=int, default=None,
                   help="Override the UIDS list at the top of this file. Repeat for multiple UIDs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Allow CLI to override the hardcoded UIDS list.
    if args.uid:
        global UIDS
        UIDS = list(args.uid)

    # ── Single-run mode ──────────────────────────────────────────────────────
    if args.run:
        print(f"Single-run mode: {args.run}")
        output = os.path.join(args.output_dir, f"{args.run}.log")
        run_url = f"https://wandb.ai/{args.entity}/{args.project}/runs/{args.run}/logs"

        state = {} if args.fresh else load_state(args.output_dir, None, args.run)
        start_cursor = state.get("cursor")
        start_page   = state.get("page", 0)
        total        = state.get("total_lines", 0)
        is_resume    = bool(start_cursor)
        first_page   = True

        print(f"{'Resuming from page ' + str(start_page + 1) if is_resume else 'Starting fresh'}")
        print(f"Run     : {run_url}")
        print(f"Output  : {output}\n", flush=True)

        seen_lines = _load_seen_lines(output)
        if seen_lines:
            print(f"Seeded {len(seen_lines)} already-written lines from existing file")

        last_good_cursor = start_cursor
        last_good_page   = start_page

        for page_lines, display_name, cursor, page, _has_next in iter_log_pages(
            args.entity, args.project, args.run, args.page_size, start_cursor, start_page,
        ):
            fresh_lines = [l for l in page_lines if l not in seen_lines]
            n_drift = len(page_lines) - len(fresh_lines)

            if n_drift and not fresh_lines:
                print(
                    f"  drift detected on page {page} ({n_drift} lines all already seen) "
                    f"— stopping, keeping cursor at page {last_good_page}"
                )
                break

            if n_drift:
                print(f"  {n_drift} cross-page duplicate line(s) dropped on page {page}")

            if not fresh_lines:
                save_state(args.output_dir, None, args.run, cursor, page, total)
                continue

            if not is_resume and first_page:
                with open(output, "w", encoding="utf-8") as f:
                    f.write("Perturb W&B Console Log\n")
                    f.write(f"Run      : {display_name}  [{args.run}]\n")
                    f.write(f"URL      : {run_url}\n")
                    f.write(f"Exported : {datetime.now(timezone.utc).isoformat()}\n")
                    f.write(SEPARATOR + "\n\n")

            first_page = False
            with open(output, "a", encoding="utf-8") as f:
                f.write("\n".join(fresh_lines) + "\n")

            seen_lines.update(fresh_lines)
            total += len(fresh_lines)
            last_good_cursor = cursor
            last_good_page   = page
            save_state(args.output_dir, None, args.run, last_good_cursor, last_good_page, total)

        print(f"\nDone — {total} total lines → {output}")
        return

    # ── By-uid mode: fetch mapping then download each validator ──────────────
    uid_map = fetch_uid_run_mapping(args.entity, args.project, max_pages=args.mapping_pages)

    if not uid_map:
        print("No matching validator runs found. Check UIDS list, ONLY_RUNNING, and project name.")
        return

    print(f"\nFound {len(uid_map)} validator run(s):\n")
    for uid, info in sorted(uid_map.items()):
        print(f"  uid={uid:<5}  run={info['run_id']}  state={info['state']}  created={info['created_at'][:10]}  name={info['display_name']}")

    if args.list_only:
        return

    n = len(uid_map)
    workers = min(args.workers, n)

    # ── Watch mode: stream forever, one thread per validator ─────────────────
    if args.watch:
        global POLL_INTERVAL
        POLL_INTERVAL = args.poll_interval
        print(
            f"\nWatch mode — streaming {n} validator(s) with {workers} worker(s), "
            f"polling every {POLL_INTERVAL}s.  Press Ctrl+C to stop.\n",
            flush=True,
        )
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for uid in uid_map:
                fut = pool.submit(
                    watch_validator_log,
                    entity=args.entity,
                    project=args.project,
                    uid=uid,
                    output_dir=args.output_dir,
                    page_size=args.page_size,
                )
                futures[fut] = uid
            try:
                for fut in as_completed(futures):
                    uid = futures[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        print(f"[uid{uid}] FATAL: {exc}", flush=True)
            except KeyboardInterrupt:
                print("\nStopping watch mode…", flush=True)
        return

    # ── One-shot mode: download current logs and exit ────────────────────────
    print(f"\nDownloading {n} validator(s) with {workers} parallel worker(s)…\n", flush=True)

    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for uid, info in uid_map.items():
            fut = pool.submit(
                download_validator_log,
                entity=args.entity,
                project=args.project,
                uid=uid,
                run_id=info["run_id"],
                display_name=info["display_name"],
                output_dir=args.output_dir,
                page_size=args.page_size,
                fresh=args.fresh,
            )
            futures[fut] = uid

    print("\n── Results ─────────────────────────────────────────────────────")
    ok = err = 0
    for fut in as_completed(futures):
        uid = futures[fut]
        try:
            fut.result()
            ok += 1
        except Exception as exc:
            print(f"[uid{uid}] ERROR: {exc}", flush=True)
            err += 1
    print(f"Done — {ok} succeeded, {err} failed.")


if __name__ == "__main__":
    main()
