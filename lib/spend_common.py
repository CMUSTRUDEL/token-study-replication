"""Shared loading, windows and coding for the telemetry extract.

The data lives outside the repository; point TOKEN_SPEND_DATA at it.

Two windows matter and they are not the same, which is the single most
important fact about this dataset:

  FULL     2026-05-10 .. 2026-08-08   the 90 days the contribution files cover
  TELEMETRY 2026-06-26 .. 2026-08-08  every AI usage event there is, 44 days
  LINK     2026-06-29 .. 2026-08-08   41 days, the part fit to join on

Telemetry begins at 12:56 on 26 June and the first three days run 76, 72 and
1,489 events against a weekday norm near 90,000. That is collection being
switched on, not three quiet days, so LINK starts at the first full weekday.

Use FULL to describe either dataset on its own. Use LINK whenever a statement
puts usage and output in the same sentence.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("TOKEN_SPEND_DATA", Path(__file__).resolve().parents[1] / "data"))
FINAL = Path(__file__).resolve().parents[1]          # repository root
CACHE = FINAL / "out" / "cache"                       # raw-derived pickles, rebuilt from scratch
CACHE.mkdir(parents=True, exist_ok=True)
ROOT = FINAL
OUT = FINAL / "out"

FULL = (pd.Timestamp("2026-05-10"), pd.Timestamp("2026-08-08"))
TELEMETRY = (pd.Timestamp("2026-06-26"), pd.Timestamp("2026-08-08"))
LINK = (pd.Timestamp("2026-06-29"), pd.Timestamp(os.environ.get("LINK_END", "2026-08-02")))  # window end: LINK_END env, shared with panels.py and the R stages
LINK_DAYS = (LINK[1] - LINK[0]).days + 1
LINK_WEEKDAYS = int(np.busday_count(LINK[0].date(), LINK[1].date()) + 1)

# count/line columns are binned: "7"->10, "140"->200, ">=1000"->"1000+"
BINNED = "1000+"


def debin(s: pd.Series, top: float = 1000.0) -> pd.Series:
    """Binned size columns to numbers, with the open top bin as its floor.

    Every value is an upper bound on the true count except '1000+', which is a
    lower bound. Treat the result as ordinal, not as a line count.
    """
    return pd.to_numeric(s.astype(str).str.replace(r"\+$", "", regex=True),
                         errors="coerce").fillna(top)


def _cached(name, build):
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    df = build()
    df.to_pickle(f)
    return df


def people() -> pd.DataFrame:
    def build():
        p = pd.read_csv(DATA / "people.csv")
        p["is_manager"] = p.is_manager.eq("Yes")
        parts = p.team.fillna("").str.split(r"\s*\\\s*")
        p["org_l2"] = parts.str[1]          # division, e.g. Division 04
        p["org_l3"] = parts.str[2]          # ecosystem within the division
        p["commits_binned"] = debin(p.commits)
        return p
    return _cached("people", build)


def usages(window=None) -> pd.DataFrame:
    """AI usage events. `window` is a (start, end) pair of Timestamps."""
    def build():
        cols = ["user_id", "event_date", "event_timestamp", "feature_id", "spent",
                "charged", "additional_data_tokens", "additional_data_model_name",
                "additional_data_tokens_context", "additional_data_tokens_generation",
                "tool", "tool_run_type"]
        u = pd.read_csv(DATA / "usages.csv", usecols=cols,
                        dtype={"user_id": "category", "tool": "category",
                               "tool_run_type": "category", "feature_id": "category",
                               "additional_data_model_name": "category"},
                        parse_dates=["event_date"])
        return u.rename(columns={"additional_data_tokens": "tokens",
                                 "additional_data_model_name": "model",
                                 "additional_data_tokens_context": "tokens_in",
                                 "additional_data_tokens_generation": "tokens_out"})
    u = _cached("usages", build)
    if window:
        u = u[u.event_date.between(*window)]
    return u


def commits(window=None) -> pd.DataFrame:
    def build():
        c = pd.read_csv(DATA / "commits.csv", parse_dates=["date"])
        c["adds"] = debin(c.additions)
        c["dels"] = debin(c.deletions)
        return c
    c = _cached("commits", build)
    return c[c.date.between(*window)] if window else c


def reviews(window=None) -> pd.DataFrame:
    def build():
        r = pd.read_csv(DATA / "reviews.csv", parse_dates=["created"])
        for col in ["additions", "deletions", "n_files", "window_additions",
                    "window_deletions", "pr_commit_count"]:
            r[col + "_n"] = debin(r[col])
        return r
    r = _cached("reviews", build)
    return r[r.created.between(*window)] if window else r


def issues() -> pd.DataFrame:
    return _cached("issues", lambda: pd.read_csv(DATA / "issues.csv"))
