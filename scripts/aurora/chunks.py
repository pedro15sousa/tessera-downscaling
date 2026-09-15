#!/usr/bin/env python3
"""The chunk schedule of the Aurora latent extraction on CSD3 (``chunks.json``).

Standard library only and Python 3.6 compatible (the login nodes' system
``python3``), so it runs there as well as inside the ROCm env; the shell
scripts call it to turn a chunk id into shell variables.

A chunk is a contiguous window of *valid* times (the timestamps of
``dataset_timestamp_global``; 6-hourly, no gaps between 2010-01-01-00 and
2023-01-10-18). Chunks partition the valid times. Everything else derives
from the window and the leads (6/24/72 h):

* inits of the chunk: ``valid - lead`` for every valid time and lead, i.e.
  ``[valid_start - 72 h, valid_end - 6 h]``;
* ERA5 input frames: each init and its ``init - 6 h`` frame, so the staging
  window is ``[valid_start - 78 h, valid_end]`` (the two frames after
  ``valid_end - 12 h`` are not needed but keep the window simple);
* the 12 inits at the start of a chunk also feed the previous chunk (their
  short leads), so consecutive chunks share 12 encoder files and 13 staging
  frames -- ``clean`` keeps whatever the next chunk still needs.

Commands:
    chunks.py list
    chunks.py show <chunk_id> [--shard I --n-shards N [--window valid|staging]]
                                                         -> KEY=VALUE lines for `eval`
    chunks.py stems <chunk_id> --kind backbone|encoder|staging [--until-next]
    chunks.py check                                      -> validates chunks.json
"""

# ruff: noqa: UP006, UP007, UP035, UP045  -- typing.* forms: must run on the login nodes' Python 3.6
import argparse
import datetime as dt
import json
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
DEFAULT_JSON = HERE / "chunks.json"
STEP = dt.timedelta(hours=6)
FMT = "%Y-%m-%d-%H"


def parse(stem: str) -> dt.datetime:
    return dt.datetime.strptime(stem, FMT)


def fmt(t: dt.datetime) -> str:
    return t.strftime(FMT)


def load(path: Path = DEFAULT_JSON) -> dict:
    cfg = json.loads(Path(path).read_text())
    ids = [c["id"] for c in cfg["chunks"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate chunk ids")
    return cfg


def chunk(cfg: dict, chunk_id: str) -> dict:
    for c in cfg["chunks"]:
        if c["id"] == chunk_id:
            return c
    raise SystemExit(
        f"unknown chunk {chunk_id!r}; known: {[c['id'] for c in cfg['chunks']]}"
    )


def next_chunk(cfg: dict, chunk_id: str) -> Optional[dict]:
    ids = [c["id"] for c in cfg["chunks"]]
    i = ids.index(chunk_id)
    return cfg["chunks"][i + 1] if i + 1 < len(ids) else None


def slots(start: str, end: str) -> List[dt.datetime]:
    t, e = parse(start), parse(end)
    out = []
    while t <= e:
        out.append(t)
        t += STEP
    return out


def shard_window(
    c: dict, shard: int, n_shards: int, window: str = "valid"
) -> Tuple[str, str]:
    """Contiguous sub-window ``shard`` (0-based) of ``n_shards`` equal parts of
    the chunk's valid (or staging) window. Contiguous, not interleaved: an
    init's rollout serves three valid times, so only the inits at a
    sub-window boundary are rolled out twice (by both neighbouring shards,
    for different leads)."""
    if not 0 <= shard < n_shards:
        raise SystemExit(f"shard {shard} outside 0..{n_shards - 1}")
    s = slots(c[f"{window}_start"], c[f"{window}_end"])
    n = len(s)
    lo = shard * n // n_shards
    hi = (shard + 1) * n // n_shards
    if hi <= lo:
        raise SystemExit(
            f"chunk {c['id']} has {n} slots, too few for {n_shards} shards"
        )
    return fmt(s[lo]), fmt(s[hi - 1])


def windows(cfg: dict, c: dict) -> Dict[str, Tuple[str, str]]:
    leads = cfg["leads_hours"]
    vs, ve = parse(c["valid_start"]), parse(c["valid_end"])
    return {
        "backbone": (fmt(vs), fmt(ve)),
        "encoder": (
            fmt(vs - dt.timedelta(hours=max(leads))),
            fmt(ve - dt.timedelta(hours=min(leads))),
        ),
        "staging": (c["staging_start"], c["staging_end"]),
    }


def stems(cfg: dict, c: dict, kind: str, until_next: bool) -> List[str]:
    """Timestamp stems of the chunk's files of one kind. With ``until_next``
    the stems the next chunk also needs are left out (what ``clean`` deletes)."""
    start, end = windows(cfg, c)[kind]
    limit = None
    nxt = next_chunk(cfg, c["id"])
    if until_next and nxt is not None:
        limit = parse(windows(cfg, nxt)[kind][0])
    return [fmt(t) for t in slots(start, end) if limit is None or t < limit]


def check(cfg: dict) -> None:
    lead_max = max(cfg["leads_hours"])
    stage_h = cfg["staging_lead_hours"]
    if stage_h != lead_max + 6:
        raise ValueError(f"staging_lead_hours must be {lead_max + 6}, got {stage_h}")
    prev_end = None
    for c in cfg["chunks"]:
        vs, ve = parse(c["valid_start"]), parse(c["valid_end"])
        if ve < vs:
            raise ValueError(f"{c['id']}: valid_end before valid_start")
        if any(t.hour % 6 for t in (vs, ve)):
            raise ValueError(f"{c['id']}: window must start/end on a 6-hourly slot")
        if parse(c["staging_start"]) != vs - dt.timedelta(hours=stage_h):
            raise ValueError(f"{c['id']}: staging_start != valid_start - {stage_h} h")
        if parse(c["staging_end"]) != ve:
            raise ValueError(f"{c['id']}: staging_end != valid_end")
        if c["marker"] != f"{cfg['marker_dir']}/{c['id']}.done":
            raise ValueError(
                f"{c['id']}: marker path does not follow marker_dir/<id>.done"
            )
        if prev_end is not None and vs != prev_end + STEP:
            raise ValueError(
                f"{c['id']}: does not start right after the previous chunk"
            )
        prev_end = ve
    first, last = cfg["chunks"][0], cfg["chunks"][-1]
    if (first["valid_start"], last["valid_end"]) != tuple(cfg["valid_range"]):
        raise ValueError("chunks do not cover valid_range exactly")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", type=Path, default=DEFAULT_JSON)
    sub = p.add_subparsers(dest="cmd")
    sub.required = True  # the keyword form needs Python >= 3.7
    sub.add_parser("list")
    sub.add_parser("check")
    s = sub.add_parser("show")
    s.add_argument("chunk_id")
    s.add_argument("--shard", type=int, default=None)
    s.add_argument("--n-shards", type=int, default=1)
    s.add_argument("--window", choices=["valid", "staging"], default="valid")
    s = sub.add_parser("stems")
    s.add_argument("chunk_id")
    s.add_argument("--kind", choices=["backbone", "encoder", "staging"], required=True)
    s.add_argument("--until-next", action="store_true")
    a = p.parse_args(argv)

    cfg = load(a.json)
    check(cfg)
    if a.cmd == "check":
        print(f"{a.json}: {len(cfg['chunks'])} chunks, ok")
        return 0
    if a.cmd == "list":
        print(
            f"{'id':10s} {'valid_start':14s} {'valid_end':14s} {'staging_start':14s} slots"
        )
        for c in cfg["chunks"]:
            n = len(slots(c["valid_start"], c["valid_end"]))
            print(
                f"{c['id']:10s} {c['valid_start']:14s} {c['valid_end']:14s} {c['staging_start']:14s} {n}"
            )
        return 0
    c = chunk(cfg, a.chunk_id)
    if a.cmd == "stems":
        for stem in stems(cfg, c, a.kind, a.until_next):
            print(stem)
        return 0
    w = windows(cfg, c)
    nxt = next_chunk(cfg, c["id"])
    out = {
        "CHUNK_ID": c["id"],
        "VALID_START": c["valid_start"],
        "VALID_END": c["valid_end"],
        "STAGING_START": c["staging_start"],
        "STAGING_END": c["staging_end"],
        "INIT_START": w["encoder"][0],
        "INIT_END": w["encoder"][1],
        "MARKER": c["marker"],
        "MARKER_DIR": cfg["marker_dir"],
        "N_SLOTS": len(slots(c["valid_start"], c["valid_end"])),
        "LEADS": " ".join(str(h) for h in cfg["leads_hours"]),
        "REGIONS": " ".join(cfg["regions"]),
        "NEXT_CHUNK_ID": nxt["id"] if nxt else "",
        "NEXT_STAGING_START": nxt["staging_start"] if nxt else "",
    }
    if a.shard is not None:
        ss, se = shard_window(c, a.shard, a.n_shards, a.window)
        out["SHARD_START"], out["SHARD_END"] = ss, se
        # ISO spellings for scripts/data/download_era5_wb2.py --start/--end.
        out["SHARD_START_ISO"] = parse(ss).strftime("%Y-%m-%dT%H:00")
        out["SHARD_END_ISO"] = parse(se).strftime("%Y-%m-%dT%H:00")
    for k, v in out.items():
        print(f"{k}={shlex.quote(str(v))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
