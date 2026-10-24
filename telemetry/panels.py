"""Build the analysis panels for the telemetry experiments.

Writes four CSVs to out/telemetry, which models.R and the stage scripts read:

    persons.csv       one row per person: covariates, pre-period and link-window
                      output, usage totals, first-use date
    person_week.csv   person x week, 13 Monday-start weeks 2026-05-11..2026-08-09
    person_day.csv    person x weekday in the link window
    commits_link.csv  one row per commit authored in the link window, with the
                      author's AI use that week

Definitions every model shares, and why:

- Output is commits by author date, because that is the only date the file has.
  Raw counts, no daily cap (CAP_Q = None).
- PRE is the seven full weeks before telemetry (2026-05-11..2026-06-28); the
  telemetry ramp-up days (26-28 June) fall in its last week and carry almost no
  events. LINK is the five weeks 2026-06-29..2026-08-02 (LINK_END).
- An inactive spell ("away" in the code) is a run of at least AWAY_RUN
  consecutive weekdays with no commit and no AI event. Those weekdays are
  treated as inactive, not as zero-output working days, in the within-person
  models. The extract cannot say why a spell is inactive (leave, work in
  repositories outside the extract, work that produces no commits). Short
  gaps stay in as genuine zero days, so the rule does not condition on either
  variable day by day.
- A late adopter is a user whose first AI event falls on or after
  LATE_FROM: their earlier telemetry weeks are observed non-use, which gives
  them a before and an after.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import spend_common as S  # noqa: E402

OUT = S.FINAL / "out" / "telemetry"
OUT.mkdir(parents=True, exist_ok=True)

WEEK0 = pd.Timestamp("2026-05-11")              # Monday
PRE = (WEEK0, pd.Timestamp("2026-06-28"))
LINK = (pd.Timestamp("2026-06-29"), pd.Timestamp(os.environ.get("LINK_END", "2026-08-02")))
# PANEL_DEDUP=1 drops GitHub commits that duplicate a commit on the company's internal
# host (a mirror, from the week of 1 June): same person, author date, size bins and issue
# ids, matched one-to-one. Same signature as companion/build_dataset.py.
DEDUP = os.environ.get("PANEL_DEDUP", "1") == "1"
LINK_WEEK0 = 7                                   # index of the week starting 06-29
LATE_FROM = pd.Timestamp("2026-07-06")
AWAY_RUN = 5
# PANEL_RULE=any: a unit (day or week) is in the panel if it has anything recorded
# (an AI event or a commit); no consecutive-day inactivity rule. Implemented as
# AWAY_RUN = 1 plus a weekly any_rec flag computed over all seven days. Authored
# GitHub PRs no longer count (2026-10-23): output is commits only.
RULE = os.environ.get("PANEL_RULE", "any")
if RULE == "any":
    AWAY_RUN = 1
NEW_MODEL = "model_02"           # the frontier model first seen 2026-07-24, inside the window
CAP_Q = None   # no daily cap: the paper and the companion pipeline use raw commit counts (a P99 cap, 42/day, was applied until 2026-10-12)
ISSUE_CLASS = {"Bug": "bug", "Exception": "bug",
               "Task": "task", "Internal Task": "task", "Technical task": "task", "Automation Task": "task",
               "Feature": "feature"}
TYPE_COLS = ["bug", "task", "feature", "otype"]

def week_index(d):
    return ((d - WEEK0).dt.days // 7).astype(int)


def main():
    p = S.people()[["user_id", "family", "is_manager", "org_l2", "team"]].copy()   # job family: coded from the title before release
    p["div_top"] = p.org_l2.eq(p.org_l2.value_counts().idxmax())   # the largest division

    c = S.commits()
    n_mirror = 0
    if DEDUP:
        sig = (c.user_id.astype(str) + "|" + c.date.astype(str) + "|" + c.additions.astype(str)
               + "|" + c.deletions.astype(str) + "|" + c.issue_ids.fillna("").astype(str))
        k = sig.groupby([sig, c.source]).cumcount()
        n_internal = sig[c.source.eq("internal")].value_counts()
        mirror = c.source.eq("github") & (k < sig.map(n_internal).fillna(0))
        n_mirror = int(mirror.sum())
        print(f"mirror dedup: dropping {n_mirror} of {int(c.source.eq('github').sum())} GitHub commits")
        c = c[~mirror]
    c = c[c.date.between(WEEK0, LINK[1])].copy()
    c["large"] = (c.adds > 100).astype(int)          # bins 200+ : more than 100 lines
    c["linked"] = c.issue_ids.notna() & c.issue_ids.astype(str).str.len().gt(0)
    c["linked"] = c.linked.astype(int)
    c["direct"] = c.product_tag.eq("direct").astype(int)
    c["github"] = c.source.eq("github").astype(int)
    # Work type of a commit, from the type of the issues its message mentions
    # (the paper splits output by type; our extract has no PR labels, so the
    # linked issue's type stands in). Taxonomy fixed before looking at spend.
    # A commit naming issues of several types is split equally across them;
    # unlinked commits get no type, so bug + task + feature + otype = linked.
    ityp = S.issues().set_index("id").type.map(lambda t: ISSUE_CLASS.get(t, "otype"))
    ids = c.issue_ids.fillna("").astype(str).str.split()
    ex = ids.explode().dropna()
    ex = ex[ex != ""]
    cls = ex.map(ityp).fillna("otype")              # 1% of ids are not in issues.csv: kept as otype
    frac = pd.get_dummies(cls).groupby(level=0).mean()
    for k in TYPE_COLS:
        c[k] = frac.reindex(c.index)[k].fillna(0.0) if k in frac else 0.0
        c[k] = c[k] * c.linked

    # --- daily commit counts, capped -------------------------------------------
    cd = c.groupby(["user_id", "date"]).agg(
        commits_raw=("date", "size"), direct=("direct", "sum"), large=("large", "sum"),
        linked=("linked", "sum"), github=("github", "sum"),
        adds=("adds", "sum"), dels=("dels", "sum"),
        **{k: (k, "sum") for k in TYPE_COLS}).reset_index()   # binned sizes: upper bounds
    cap = float(np.ceil(cd.commits_raw.quantile(CAP_Q))) if CAP_Q is not None else np.inf
    scale = np.where(cd.commits_raw > cap, cap / cd.commits_raw, 1.0)
    cd["commits"] = np.minimum(cd.commits_raw, cap)
    for k in ["direct", "large", "linked", "github", "adds", "dels"] + TYPE_COLS:
        cd[k] = cd[k] * scale                       # keep composition when capping
    n_capped = int((cd.commits_raw > cap).sum())

    u = S.usages()
    u = u.assign(user_id=u.user_id.astype(str))
    u["new_model"] = u.model.astype(str).eq(NEW_MODEL)
    u["spent_new_model"] = u.spent.where(u.new_model, 0.0)
    # tool mix, for the tool-adjusted bound: tokens on Agent B and on the new model
    u["tokens_agent_b"] = u.tokens.where(u.tool.astype(str).eq("agent_b"), 0.0)
    u["tokens_new_model"] = u.tokens.where(u.new_model, 0.0)
    # mode of use, for RQ2: remote runs, off-hours events (weekend or 20:00-06:00 UTC),
    # and sessions (runs of one person's events with no gap over 30 minutes)
    ts = pd.to_datetime(u.event_timestamp)
    u["events_remote"] = u.tool_run_type.astype(str).eq("remote").astype(float)
    u["events_off"] = ((ts.dt.dayofweek >= 5) | (ts.dt.hour < 6) | (ts.dt.hour >= 20)).astype(float)
    u = u.sort_values(["user_id", "event_timestamp"])
    gap = ts.loc[u.index].groupby(u.user_id, observed=True).diff().dt.total_seconds() / 60
    u["sess"] = (gap.isna() | (gap > 30)).cumsum()
    sl = u.groupby("sess").agg(start=("event_timestamp", "min"), end=("event_timestamp", "max"))
    sdur = (pd.to_datetime(sl.end) - pd.to_datetime(sl.start)).dt.total_seconds() / 3600
    u["events_long"] = u.sess.map(sdur > 2).astype(float)
    u["sessions"] = (~u.sess.duplicated()).astype(float)     # a session counts on the day it starts
    ud = u.groupby(["user_id", "event_date"], observed=True).agg(
        spent_new_model=("spent_new_model", "sum"), tokens_agent_b=("tokens_agent_b", "sum"), tokens_new_model=("tokens_new_model", "sum"),
        tokens_out=("tokens_out", "sum"), events_remote=("events_remote", "sum"), events_off=("events_off", "sum"),
        events_long=("events_long", "sum"), sessions=("sessions", "sum"),
        events=("spent", "size"), spent=("spent", "sum"), tokens=("tokens", "sum")
    ).reset_index().rename(columns={"event_date": "date"})

    # --- PR-authored days (GitHub), descriptive only (not part of the activity rule) ---
    rv = S.reviews()
    rv = rv[rv.roles.astype(str).str.contains("author") & rv.created.between(WEEK0, LINK[1])]
    pd_prs = rv.groupby(["user_id", "created"]).size().rename("prs").reset_index().rename(columns={"created": "date"})
    pd_prs["user_id"] = pd_prs.user_id.astype(str)

    # --- person-day grid over every calendar day PRE..LINK -----------------------
    # Totals (commits, spend) include weekends. Presence and absence are judged
    # on weekdays only: a weekend without commits is not inactivity.
    days = pd.date_range(WEEK0, LINK[1])
    grid = pd.MultiIndex.from_product([p.user_id, days], names=["user_id", "date"]).to_frame(index=False)
    d = grid.merge(cd, on=["user_id", "date"], how="left").merge(ud, on=["user_id", "date"], how="left")
    d = d.merge(pd_prs, on=["user_id", "date"], how="left")
    d["prs_auth"] = d.prs.fillna(0.0); d = d.drop(columns="prs")
    for k in ["commits_raw", "commits", "direct", "large", "linked", "github", "adds", "dels", "events", "spent",
              "tokens", "spent_new_model", "tokens_agent_b", "tokens_new_model", "tokens_out", "events_remote", "events_off",
              "events_long", "sessions"] + TYPE_COLS:
        d[k] = d[k].fillna(0.0)

    users = set(u.user_id)
    d["user"] = d.user_id.isin(users)
    d["weekday"] = d.date.dt.dayofweek < 5
    # activity for absence detection: commits anywhere, AI only where telemetry exists
    d["active"] = (d.commits_raw > 0) | (d.events > 0)
    d = d.sort_values(["user_id", "date"]).reset_index(drop=True)

    def away_flags(a):
        a = a.to_numpy()
        out = np.zeros(len(a), bool)
        i = 0
        while i < len(a):
            if not a[i]:
                j = i
                while j < len(a) and not a[j]:
                    j += 1
                if j - i >= AWAY_RUN:
                    out[i:j] = True
                i = j
            else:
                i += 1
        return out

    # runs are counted over the weekday sequence only (Fri -> Mon is adjacent)
    wdx = d.index[d.weekday]
    wd = d.loc[wdx]
    d["away"] = False
    d.loc[wdx, "away"] = np.concatenate([away_flags(g.active) for _, g in wd.groupby("user_id", sort=False)])
    # commits-only version, identical for users and non-users; used where the two
    # groups are compared (the event study), since AI events exist only for users
    d["away_c"] = False
    d.loc[wdx, "away_c"] = np.concatenate([away_flags(g.commits_raw > 0) for _, g in wd.groupby("user_id", sort=False)])
    d["week"] = week_index(d.date)
    d["present"] = (d.weekday & ~d.away).astype(int)

    # --- first use, late adopters ----------------------------------------------
    first = u.groupby("user_id").event_date.min()
    p["user"] = p.user_id.isin(users)
    p["first_use"] = p.user_id.map(first)
    p["late"] = p.first_use >= LATE_FROM
    p["cohort_week"] = np.where(p.first_use.notna(), ((p.first_use - WEEK0).dt.days // 7), np.nan)

    # --- person table ----------------------------------------------------------
    pre = d[d.date.between(*PRE)]
    link = d[d.date.between(*LINK)]
    agg = lambda x, pfx: x.groupby("user_id").agg(**{
        f"{pfx}_commits": ("commits", "sum"), f"{pfx}_commits_raw": ("commits_raw", "sum"),
        f"{pfx}_direct": ("direct", "sum"), f"{pfx}_large": ("large", "sum"),
        f"{pfx}_linked": ("linked", "sum"), f"{pfx}_github": ("github", "sum"),
        f"{pfx}_commit_days": ("commits_raw", lambda s: int((s > 0).sum())),
        f"{pfx}_present": ("present", "sum")})
    p = p.merge(agg(pre, "pre"), on="user_id").merge(agg(link, "link"), on="user_id")
    ul = u[u.event_date.between(*LINK)]
    up = ul.groupby("user_id").agg(events=("spent", "size"), spent=("spent", "sum"),
                                   tokens=("tokens", "sum"), ai_days=("event_date", "nunique"))
    p = p.merge(up, on="user_id", how="left")
    for k in ["events", "spent", "tokens", "ai_days"]:
        p[k] = p[k].fillna(0)
    # vendor mix among users (by spend)
    vs = ul.groupby(["user_id", "tool"], observed=True).spent.sum().unstack(fill_value=0)
    share = vs.div(vs.sum(axis=1), axis=0)
    p["agent_a_share"] = p.user_id.map(share.get("agent_a", pd.Series(dtype=float)))
    p["agent_b_share"] = p.user_id.map(share.get("agent_b", pd.Series(dtype=float)))
    for k in ["direct", "github", "linked", "large"]:
        p[f"pre_{k}_share"] = (p[f"pre_{k}"] / p.pre_commits).where(p.pre_commits > 0)
    # usage tiers among users, by link-window spend
    tiers = pd.qcut(p.loc[p.user, "spent"], 3, labels=["light", "medium", "heavy"])
    p["tier"] = "non-user"
    p.loc[p.user, "tier"] = tiers.astype(str)
    # baseline productivity tercile (pre-period commits), for heterogeneity
    p["base_tercile"] = pd.qcut(p.pre_commits.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str)
    p["n_capped_days"] = n_capped
    p["n_mirror_dropped"] = n_mirror
    p["rule"] = RULE
    p["dedup"] = int(DEDUP)
    p["cap"] = cap

    # --- person-week -----------------------------------------------------------
    w = d.groupby(["user_id", "week"]).agg(
        commits=("commits", "sum"), commits_raw=("commits_raw", "sum"), direct=("direct", "sum"),
        large=("large", "sum"), linked=("linked", "sum"), adds=("adds", "sum"), dels=("dels", "sum"),
        **{k: (k, "sum") for k in TYPE_COLS},
        commit_days=("commits_raw", lambda s: int((s > 0).sum())),
        events=("events", "sum"), spent=("spent", "sum"), tokens=("tokens", "sum"),
        spent_new_model=("spent_new_model", "sum"), tokens_agent_b=("tokens_agent_b", "sum"), tokens_new_model=("tokens_new_model", "sum"),
        tokens_out=("tokens_out", "sum"), events_remote=("events_remote", "sum"), events_off=("events_off", "sum"),
        events_long=("events_long", "sum"), sessions=("sessions", "sum"),
        ai_days=("events", lambda s: int((s > 0).sum())),
        present=("present", "sum"), weekdays=("weekday", "sum"),
        prs_auth=("prs_auth", "sum"), away_c=("away_c", "sum")).reset_index()
    # anything recorded in the week, over all seven days (panel inclusion under PANEL_RULE=any)
    w["any_rec"] = ((w.tokens > 0) | (w.commits_raw > 0)).astype(int)
    # an active-output week for the adoption designs (users and non-users alike)
    w["active_out"] = (w.commits_raw > 0).astype(int) if RULE == "any" else (w.away_c < 3).astype(int)
    w["week_start"] = WEEK0 + pd.to_timedelta(w.week * 7, unit="D")
    w["link"] = (w.week >= LINK_WEEK0).astype(int)
    w = w.merge(p[["user_id", "user", "late", "cohort_week", "is_manager", "family", "tier",
                   "base_tercile", "agent_a_share"]], on="user_id")

    # --- person-day, link window, weekdays only ---------------------------------
    dl = d[d.date.between(*LINK) & d.weekday].copy()
    dl["dow"] = dl.date.dt.dayofweek

    # --- commit level, link window, with that week's usage ---------------------
    cl = c[c.date.between(*LINK)].copy()
    cl["week"] = week_index(cl.date)
    cl = cl.merge(w[["user_id", "week", "spent", "events", "present"]], on=["user_id", "week"], how="left")
    cl = cl.merge(p[["user_id", "user"]], on="user_id")
    cl["size_bin"] = cl.adds

    # Panel rule (agreed 2026-10-07): drop only what is zero on everything, at every level.
    # A person with commits in the window stays even if their metered use fell outside
    # it (ramp-up days, or the dropped last week); their weeks are zero-token weeks with output.
    if RULE == "any":
        print(f"panel rule: {int((p.user & (p.tokens <= 0)).sum())} users have metered events only outside the window; kept")
    p.to_csv(OUT / "persons.csv", index=False)
    w.to_csv(OUT / "person_week.csv", index=False)
    dl.to_csv(OUT / "person_day.csv", index=False)
    cl[["user_id", "date", "week", "large", "linked", "direct", "github", "size_bin", "dels",
        "spent", "events", "present", "user"]].to_csv(OUT / "commits_link.csv", index=False)
    meta = {"week0": str(WEEK0.date()), "pre": [str(PRE[0].date()), str(PRE[1].date())],
            "link": [str(LINK[0].date()), str(LINK[1].date())], "late_from": str(LATE_FROM.date()),
            "away_run": AWAY_RUN, "cap_q": CAP_Q, "cap": (cap if np.isfinite(cap) else None), "n_capped_days": n_capped,
            "n_person_days_with_commits": int(len(cd)), "new_model": NEW_MODEL,
            "away_weekdays": int(d.away.sum()), "weekday_rows": int(d.weekday.sum()),
            "away_weekdays_link": int(d[d.date.between(*LINK)].away.sum()),
            "weekday_rows_link": int(d[d.date.between(*LINK)].weekday.sum()),
            "commits_link_rows": int(len(cl)),
            "persons": int(len(p)), "person_weeks": int(len(w)), "person_days_link": int(len(dl))}
    (OUT / "panels_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"persons {len(p)}  person-weeks {len(w)}  person-days {len(dl)}  commits {len(cl)}")
    print("no daily commit cap" if CAP_Q is None else f"daily cap {cap:.0f} commits (P{CAP_Q*100:.0f}), {n_capped} person-days capped")
    print(f"users {int(p.user.sum())}, late adopters {int(p.late.sum())}, inactive weekdays {int(d.away.sum())} of {len(d)}")


if __name__ == "__main__":
    main()
