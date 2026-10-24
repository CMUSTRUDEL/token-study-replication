"""Build the analysis tables from the five CSVs of the extract.

The 2026-09-15 extract dates commits (git author date) and PRs (creation
date); issues remain undated. This script produces:

  analysis/out/people.csv        one row per person, all sources joined
  analysis/out/person_week.csv   one row per person x Monday-week, 2026-05-11
                                 .. 2026-08-03: artifact output and telemetry
  analysis/out/person_day.csv    one row per person x day with any commit or
                                 metered use (for lag structure)
  analysis/out/tokens_weekly.csv one row per person x telemetry week
  analysis/out/tokens_daily.csv  one row per person x day

Two corrections apply to every commit count:
  * Mirror deduplication. Most GitHub `direct` commits duplicate a commit on the
    company's internal host (same person, author date, size bins, and linked
    issues). A GitHub commit is dropped when an internal commit carries the same
    signature, matched one-to-one.
  * Dating. Rows were selected on committer date, so commits authored before
    the window are a partial set; they are excluded. Commits authored late in
    the window that landed after it are missing (right truncation); weekly
    analyses absorb this with week fixed effects.

Size bins ("10", "200", "1000+") are mapped to interval midpoints; "1000+" is
mapped to 1500 (see BIN_TOP). Run from the work directory that run_all.sh sets
up. Set USAGES_PKL to a pickled copy of usages.csv to skip the CSV parse.
"""
import os
import numpy as np
import pandas as pd

D = "data/"
OUT = "analysis/out/"
os.makedirs(OUT, exist_ok=True)

WINDOW_START = pd.Timestamp("2026-05-10")   # artifact window (README)
WINDOW_END = pd.Timestamp("2026-08-08")
TELEM_START = pd.Timestamp("2026-06-29")    # Monday of the metering rollout
PANEL_START = pd.Timestamp("2026-05-11")    # first Monday inside the window
# panel weeks are indexed relative to TELEM_START: -7..-1 pre-telemetry,
# 0..5 telemetry (week 5 = 2026-08-03..08, six days and heavily truncated)
BIN_TOP = 1500.0                            # midpoint stand-in for "1000+"

RESOLVED_STATES = {
    "Fixed", "Done", "Verified", "Fixed in Branch", "Fixed In Branch",
    "Fixed in branch", "Released", "Completed", "Merged", "Solved",
    "Deployed", "Ready for Merge", "Ready to be merged", "Ready To Release",
    "Ready for Dev (Fixed)", "Duplicate of Fixed", "Closed",
}
CLOSED_NOFIX_STATES = {
    "Obsolete", "Duplicate", "Declined", "Won't fix", "Can't Reproduce",
    "Can't reproduce", "As Designed", "As designed", "Works As Intended",
    "Works as Intended", "Third Party Problem", "Third-Party Problem",
    "Cancelled", "Canceled", "Withdrawn", "Invalid", "Not Applicable to my product",
    "Incomplete", "Shelved", "Deprecated", "Misconfiguration", "Answered",
}


def unbin(s):
    """Bin label -> interval midpoint. '0'->0, '10'->5.5 (1..10), '20'->15.5,
    '200'->150.5 (101..200), '1000+'->BIN_TOP."""
    s = s.astype(str)
    out = pd.Series(np.nan, index=s.index)
    top = s == "1000+"
    out[top] = BIN_TOP
    v = pd.to_numeric(s[~top], errors="coerce")
    lo = np.where(v <= 0, 0, np.where(v <= 100, v - 10, v - 100)) + 1
    mid = (lo + v) / 2.0
    mid = np.where(v <= 0, 0.0, mid)
    out[~top] = mid
    return out


def mirror_flags(cm):
    """True for GitHub commits that duplicate an internal-host commit: same person,
    author date, size bins, and linked issues, matched one-to-one."""
    sig = (cm["user_id"] + "|" + cm["date"] + "|" + cm["additions"].astype(str)
           + "|" + cm["deletions"].astype(str) + "|"
           + cm["issue_ids"].fillna(""))
    k = sig.groupby([sig, cm["source"]]).cumcount()
    n_internal = sig[cm["source"] == "internal"].value_counts()
    return (cm["source"] == "github") & (k < sig.map(n_internal).fillna(0))


def week_index(d):
    """Monday-week index relative to TELEM_START (0 = 2026-06-29)."""
    return (d - TELEM_START).dt.days // 7


def person_week(ids, cm, rv, yt, us):
    """Person x week panel, weeks -7..5. Zero-filled for every person."""
    bug_ids = set(yt.loc[yt["type"].fillna("").str.contains(
        "Bug|Problem|Exception", regex=True), "id"])
    c = cm[cm["d"] >= PANEL_START].copy()
    c["week"] = week_index(c["d"])
    c["bug_linked"] = c["issue_ids"].fillna("").str.split().map(
        lambda ids_: any(i in bug_ids for i in ids_))
    gc = c.groupby(["user_id", "week"])
    Cw = pd.DataFrame({
        "commits": gc.size(),
        "commits_internal": gc.apply(lambda x: (x.source == "internal").sum()),
        "commits_linked": gc["linked"].sum(),
        "commits_bug": gc["bug_linked"].sum(),
        "commits_big": gc["big"].sum(),
        "lines": gc["lines"].sum(),
        "commit_days": gc["d"].nunique(),
    })
    r = rv[rv["d"] >= PANEL_START].copy()
    r["week"] = week_index(r["d"])
    a = r[r["is_author"]].groupby(["user_id", "week"])
    Aw = pd.DataFrame({
        "prs_authored": a.size(),
        "pr_lines": a["pr_lines"].sum(),
        "pr_files": a["pr_files"].sum(),
        "pr_toiw_mean": a["toiw"].mean(),
        "pr_reviewers_mean": a["n_reviewers"].mean(),
        "pr_comments_self": a["n_comments_by_person_in_window"].sum(),
    })
    n = r[~r["is_author"]].groupby(["user_id", "week"])
    Rw = pd.DataFrame({
        "prs_reviewed": n.size(),
        "review_comments": n["n_comments_by_person_in_window"].sum(),
    })
    u = us.copy()
    u["week"] = week_index(u["event_date"])
    u = u[u["week"] >= 0]                       # 4 people, 3 days before rollout
    gu = u.groupby(["user_id", "week"])
    Uw = pd.DataFrame({
        "tokens": gu["tok"].sum(), "gen_tokens": gu["gen"].sum(),
        "credits": gu["spent"].sum(), "events": gu.size(),
        "ai_days": gu["event_date"].nunique(),
        "tok_agent_a": gu.apply(lambda x: x.loc[x.agent_a, "tok"].sum()),
        "tok_agent_b": gu.apply(lambda x: x.loc[x.agent_b, "tok"].sum()),
        "tok_frontier": gu.apply(lambda x: x.loc[x.tier == "frontier", "tok"].sum()),
        "tok_remote": gu.apply(lambda x: x.loc[x.remote, "tok"].sum()),
        "n_models": gu["additional_data_model_name"].nunique(),
    })
    grid = pd.MultiIndex.from_product([ids, range(-7, 6)],
                                      names=["user_id", "week"])
    PW = pd.DataFrame(index=grid).join(Cw).join(Aw).join(Rw).join(Uw)
    keep_na = ["pr_toiw_mean", "pr_reviewers_mean"]
    fill = [col for col in PW.columns if col not in keep_na]
    PW[fill] = PW[fill].fillna(0.0)
    PW = PW.reset_index()
    PW["week_start"] = TELEM_START + pd.to_timedelta(7 * PW["week"], unit="D")
    PW["telemetry"] = (PW["week"] >= 0).astype(int)
    return PW


def person_day(cm, us):
    """Person x day counts of commits (author date) and metered tokens."""
    c = cm[cm["d"] >= PANEL_START].groupby(["user_id", "d"]).size() \
        .rename("commits")
    t = us.groupby(["user_id", "event_date"])["tok"].sum().rename("tokens")
    t.index = t.index.set_names(["user_id", "d"])
    return pd.concat([c, t], axis=1).fillna(0.0).reset_index()


def main():
    people = pd.read_csv(D + "people.csv")
    yt = pd.read_csv(D + "issues.csv")
    cm = pd.read_csv(D + "commits.csv")
    rv = pd.read_csv(D + "reviews.csv")
    pkl = os.environ.get("USAGES_PKL", "")
    us = pd.read_pickle(pkl) if os.path.exists(pkl) else pd.read_csv(
        D + "usages.csv", dtype={"feature_id": str})
    us["event_date"] = pd.to_datetime(us["event_date"])

    # role and seniority are coded from the job title before release, by keyword:
    # role intern / qa (qa, test, sdet) / dev (developer, engineer, architect,
    # contractor) / lead_mgmt (lead, head, manager, adviser) / other, in that order
    P = people.rename(columns={"is_manager": "manager"}).copy()
    P["manager"] = (P["manager"] == "Yes").astype(int)
    P["team2"] = P["team"].str.split(r" \\ ").str[1].fillna("Company")
    P["team3"] = P["team"].str.split(r" \\ ").str[:3].str.join(" / ")
    P["commits_bin_people"] = P["commits"].astype(str)
    P = P.drop(columns=["family", "commits"])
    P = P.set_index("user_id")

    # ---------------------------------------------------------- commits
    cm["d"] = pd.to_datetime(cm["date"])
    cm["mirror"] = mirror_flags(cm)
    cm_raw = cm
    cm = cm[(~cm["mirror"]) & (cm["d"] >= WINDOW_START)].copy()
    cm["adds"] = unbin(cm["additions"])
    cm["dels"] = unbin(cm["deletions"])
    cm["lines"] = cm["adds"] + cm["dels"]
    cm["linked"] = cm["issue_ids"].notna()
    cm["big"] = (cm["additions"].astype(str) == "1000+") | \
                (cm["deletions"].astype(str) == "1000+")
    g = cm.groupby("user_id")
    C = pd.DataFrame({
        "commits": g.size(),
        "commits_direct": g.apply(lambda x: (x.product_tag == "direct").sum()),
        "commits_github": g.apply(lambda x: (x.source == "github").sum()),
        "commits_linked": g["linked"].sum(),
        "commits_big": g["big"].sum(),
        "lines": g["lines"].sum(),
        "adds": g["adds"].sum(),
        "dels": g["dels"].sum(),
        "lines_median_commit": g["lines"].median(),
    })
    C["commits_raw"] = cm_raw.groupby("user_id").size()
    C["commits_mirror"] = cm_raw.groupby("user_id")["mirror"].sum()
    C["commits_predated"] = cm_raw[~cm_raw.mirror & (cm_raw.d < WINDOW_START)] \
        .groupby("user_id").size()
    C["commits_linked_share"] = C["commits_linked"] / C["commits"]
    C["lines_per_commit"] = C["lines"] / C["commits"]

    # ---------------------------------------------------------- PRs
    rv["d"] = pd.to_datetime(rv["created"])
    rv["is_author"] = rv["roles"].str.contains("author")
    rv["is_reviewer"] = rv["roles"].str.contains("reviewer") & ~rv["is_author"]
    rv["is_commenter_only"] = (rv["roles"] == "commenter")
    rv["pr_lines"] = unbin(rv["additions"]) + unbin(rv["deletions"])
    rv["pr_files"] = unbin(rv["n_files"])
    rv["pr_commits"] = unbin(rv["pr_commit_count"])
    rv["pr_linked"] = rv["issue_ids"].notna()
    au = rv[rv["is_author"]]
    ga = au.groupby("user_id")
    A = pd.DataFrame({
        "prs_authored": ga.size(),
        "pr_lines": ga["pr_lines"].sum(),
        "pr_files": ga["pr_files"].sum(),
        "pr_commits": ga["pr_commits"].sum(),
        "pr_toiw_mean": ga["toiw"].mean(),
        "pr_toiw_median": ga["toiw"].median(),
        "pr_reviewers_mean": ga["n_reviewers"].mean(),
        "pr_linked_share": ga["pr_linked"].mean(),
        "pr_comments_self": ga["n_comments_by_person_in_window"].sum(),
    })
    nr = rv[~rv["is_author"]]
    gn = nr.groupby("user_id")
    R = pd.DataFrame({
        "prs_reviewed": gn.size(),
        "review_comments": gn["n_comments_by_person_in_window"].sum(),
    })

    # ---------------------------------------------------------- issues
    yt["resolved"] = yt["state"].isin(RESOLVED_STATES)
    yt["closed_nofix"] = yt["state"].isin(CLOSED_NOFIX_STATES)
    yt["bug"] = yt["type"].fillna("").str.contains("Bug|Problem|Exception",
                                                  regex=True)
    yt["whole_window"] = yt["toiw"] >= 90
    gy = yt.groupby("user_id")
    Y = pd.DataFrame({
        "issues": gy.size(),
        "issues_resolved": gy["resolved"].sum(),
        "issues_closed_nofix": gy["closed_nofix"].sum(),
        "issues_bug": gy["bug"].sum(),
        "issues_open_whole_window": gy["whole_window"].sum(),
        "issue_toiw_mean": gy["toiw"].mean(),
        "issue_toiw_median_resolved": gy.apply(
            lambda x: x.loc[x.resolved, "toiw"].median()),
    })

    # ---------------------------------------------------------- telemetry
    us["week"] = ((us["event_date"] - TELEM_START).dt.days // 7).clip(lower=0)
    us["agent_a"] = us["tool"] == "agent_a"
    us["agent_b"] = us["tool"] == "agent_b"
    us["remote"] = us["tool_run_type"] == "remote"
    us["tok"] = us["additional_data_tokens"].astype(float)
    us["ctx"] = us["additional_data_tokens_context"].astype(float)
    us["gen"] = us["additional_data_tokens_generation"].astype(float)
    us["fam"] = us["model_vendor"]                 # vendor of the model, coded before release
    us["tier"] = us["model_tier"]                  # frontier / mid / small, coded from the model name
    gu = us.groupby("user_id")
    U = pd.DataFrame({
        "tokens": gu["tok"].sum(),
        "ctx_tokens": gu["ctx"].sum(),
        "gen_tokens": gu["gen"].sum(),
        "credits": gu["spent"].sum(),
        "events": gu.size(),
        "active_days": gu["event_date"].nunique(),
        "first_day": gu["event_date"].min(),
        "last_day": gu["event_date"].max(),
        "tok_agent_a": gu.apply(lambda x: x.loc[x.agent_a, "tok"].sum()),
        "tok_agent_b": gu.apply(lambda x: x.loc[x.agent_b, "tok"].sum()),
        "tok_remote": gu.apply(lambda x: x.loc[x.remote, "tok"].sum()),
        "tok_frontier": gu.apply(lambda x: x.loc[x.tier == "frontier", "tok"].sum()),
        "tok_small": gu.apply(lambda x: x.loc[x.tier == "small", "tok"].sum()),
        "n_models": gu["additional_data_model_name"].nunique(),
        "n_tools": gu["tool"].nunique(),
    })
    U["share_agent_a"] = U["tok_agent_a"] / U["tokens"]
    U["share_agent_b"] = U["tok_agent_b"] / U["tokens"]
    U["share_remote"] = U["tok_remote"] / U["tokens"]
    U["share_frontier"] = U["tok_frontier"] / U["tokens"]
    U["gen_ratio"] = U["gen_tokens"] / U["tokens"]
    U["tokens_per_event"] = U["tokens"] / U["events"]
    U["tokens_per_active_day"] = U["tokens"] / U["active_days"]
    U["credits_per_mtok"] = 1e6 * U["credits"] / U["tokens"]
    U["days_since_rollout_first"] = (U["first_day"] - TELEM_START).dt.days

    weekly = us.pivot_table(index="user_id", columns="week", values="tok",
                            aggfunc="sum", fill_value=0.0)
    weekly.columns = [f"tok_w{int(c)}" for c in weekly.columns]
    weekly.to_csv(OUT + "tokens_weekly.csv")
    daily = us.groupby(["user_id", "event_date"]).agg(
        tok=("tok", "sum"), gen=("gen", "sum"), events=("tok", "size"),
        credits=("spent", "sum")).reset_index()
    daily.to_csv(OUT + "tokens_daily.csv", index=False)

    # ---------------------------------------------------------- join
    X = P.join(C).join(A).join(R).join(Y).join(U).join(weekly)
    zero_fill = [c for c in X.columns if c.startswith(("commits", "lines", "adds",
                 "dels", "prs_", "pr_lines", "pr_files", "pr_commits",
                 "pr_comments", "review_", "issues", "tokens", "ctx_", "gen_",
                 "credits", "events", "active_days", "tok_"))]
    X[zero_fill] = X[zero_fill].fillna(0.0)
    X["has_usage"] = (X["tokens"] > 0).astype(int)
    X["has_prs"] = (X["prs_authored"] > 0).astype(int)
    X["has_issues"] = (X["issues"] > 0).astype(int)
    X["has_github_commits"] = (X["commits_github"] > 0).astype(int)
    # ---------------------------------------------------------- person-week
    PW = person_week(X.index, cm, rv, yt, us)
    PW.to_csv(OUT + "person_week.csv", index=False)

    # final rule: per-person activity totals cover the five link weeks only
    # (2026-06-29..08-02), not the truncated week of Aug 3-8 or the pre-metering
    # weeks; has_usage still marks anyone metered anywhere in the extract
    lw = PW[PW.week.between(0, 4)].groupby("user_id")[[
        "commits", "commits_linked", "commits_big", "lines", "prs_authored",
        "pr_lines", "pr_files", "prs_reviewed", "review_comments",
        "pr_comments_self", "tokens", "gen_tokens", "credits", "events",
        "tok_agent_a", "tok_agent_b", "tok_frontier", "tok_remote"]].sum()
    X[lw.columns] = lw.reindex(X.index).fillna(0.0)
    X["ctx_tokens"] = X["tokens"] - X["gen_tokens"]
    X["active_days"] = PW[PW.week.between(0, 4)].groupby("user_id").ai_days.sum().reindex(X.index).fillna(0.0)
    X["commits_linked_share"] = X["commits_linked"] / X["commits"]
    X["lines_per_commit"] = X["lines"] / X["commits"]
    for s in ["agent_a", "agent_b", "remote", "frontier"]:
        X[f"share_{s}"] = X[f"tok_{s}"] / X["tokens"]
    X["gen_ratio"] = X["gen_tokens"] / X["tokens"]
    X["tokens_per_event"] = X["tokens"] / X["events"]
    X["tokens_per_active_day"] = X["tokens"] / X["active_days"]
    X["credits_per_mtok"] = 1e6 * X["credits"] / X["tokens"]
    X.to_csv(OUT + "people.csv")
    person_day(cm, us).to_csv(OUT + "person_day.csv", index=False)

    # ---------------------------------------------------------- report
    print("people:", len(X), "| with usage:", X.has_usage.sum(),
          "| with PRs authored:", X.has_prs.sum(),
          "| with issues:", X.has_issues.sum(),
          "| with github commits:", X.has_github_commits.sum())
    print("usage & PRs:", ((X.has_usage == 1) & (X.has_prs == 1)).sum(),
          "usage & issues:", ((X.has_usage == 1) & (X.has_issues == 1)).sum())
    print(X[["commits", "lines", "prs_authored", "prs_reviewed", "issues",
             "issues_resolved", "tokens", "credits", "active_days"]]
          .describe().T.to_string())
    print("weekly token cols:", list(weekly.columns))
    print("commits: raw", len(cm_raw), "| mirror", int(cm_raw.mirror.sum()),
          "| predated", int((~cm_raw.mirror & (cm_raw.d < WINDOW_START)).sum()),
          "| kept", len(cm))
    print("person-week rows:", len(PW), "weeks:", sorted(PW.week.unique()))
    print("linked commit share (all commits):",
          round(cm.linked.mean(), 3), "| PR linked share:",
          round(rv.pr_linked.mean(), 3))
    print("issues resolved share:", round(yt.resolved.mean(), 3),
          "closed_nofix:", round(yt.closed_nofix.mean(), 3))


if __name__ == "__main__":
    main()
