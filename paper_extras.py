"""Numbers the paper quotes that no pipeline script writes out.

Recomputes, from the companion pipeline's tables (people.csv, person_week.csv,
person_day.csv) and the extract, the numbers the paper quotes that no other script
writes out. Writes out/extras.json. Run after the companion pipeline (it reads
out/companion/work).
"""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
W = HERE / "out" / "companion" / "work"
RAW = Path(os.environ.get("TOKEN_SPEND_DATA", HERE / "data"))
SURVEY = RAW / "survey" / "survey_responses.csv"
E = {}

# the companion pipeline's estimators, taken verbatim from analyze_panel.py
_src = (HERE / "companion" / "analyze_panel.py").read_text()
_ns = {"np": np, "pd": pd}
for node in ast.parse(_src).body:
    if isinstance(node, ast.FunctionDef) and node.name in {"fe_poisson", "codes", "week_dummies", "ols", "sb", "output_terms"}:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "analyze_panel.py", "exec"), _ns)
fe_poisson, codes, week_dummies, output_terms = (_ns[k] for k in ["fe_poisson", "codes", "week_dummies", "output_terms"])

X = pd.read_csv(W / "analysis/out/people.csv", index_col=0)
PW = pd.read_csv(W / "analysis/out/person_week.csv")
PD = pd.read_csv(W / "analysis/out/person_day.csv", parse_dates=["d"])
adopters = X.index[X.has_usage == 1]
any_rec = lambda d: (d.tokens > 0) | (d.commits > 0)
ALL = PW[PW.user_id.isin(adopters) & PW.week.isin(range(5))].copy()          # every adopter-week
P = ALL[any_rec(ALL)].reset_index(drop=True)                                  # the panel
P["tok_m"] = P.tokens / 1e6
P["lines_k"] = P.lines / 1e3
P["pr_lines_k"] = P.pr_lines / 1e3
P["output_any"] = (P.commits > 0).astype(int)
COUNTS, SIZES = ["commits"], ["lines_k"]


def ceiling(df, cols, form):
    df = df.reset_index(drop=True)
    kn = df.loc[df.commits > 0, cols].median().to_numpy() if form == "spline" else None
    S = output_terms(df, cols, form, kn)
    Z = np.column_stack([S, week_dummies(df.week)])
    g = codes(df.user_id)
    b, c, mu = fe_poisson(df.tok_m.values, Z, g)
    Z0 = Z.copy(); Z0[:, :S.shape[1]] = 0.0
    f0 = np.where(np.isfinite(c[g]), np.exp(Z0 @ b + np.where(np.isfinite(c), c, 0)[g]), 0.0)
    zero = df.commits == 0
    return dict(share=float(f0.sum() / df.tok_m.sum()),
                calib=float(df.tok_m[zero].sum() / mu[zero].sum())), f0, mu


# ---------------------------------------------------------------- data section
E["panel_empty_weeks"] = int((~any_rec(ALL)).sum())
E["panel_people"] = int(P.user_id.nunique())
E["panel_weeks"] = int(len(P))
E["panel_share_metered"] = float((P.tokens > 0).mean())
E["panel_share_zero_tokens"] = float((P.tokens == 0).mean())
act = P[P.tokens > 0]
E["active_weeks"] = int(len(act))
E["active_weeks_no_output"] = int((act.output_any == 0).sum())
E["active_share_no_output"] = float((act.output_any == 0).mean())
E["median_active_commits"] = float(act.commits.median())
E["median_active_tokens_m"] = float(act.tokens.median() / 1e6)
E["p10_active_tokens_m"] = float(act.tokens.quantile(.1) / 1e6)
E["p90_active_tokens_m"] = float(act.tokens.quantile(.9) / 1e6)
# five-week totals per engineer, as multiples of their median (Figure 0, right panel; no absolute volumes in the paper)
_five = P.groupby("user_id").tokens.sum(); _five = _five[_five > 0]
E["five_week_n"] = int(len(_five))
E["five_week_rel_p10"] = float(_five.quantile(.1) / _five.median())
E["five_week_rel_p90"] = float(_five.quantile(.9) / _five.median())
E["share_weeks_with_pr"] = float((P.prs_authored > 0).mean())
# commits that name an issue, and the share of those that name a bug (analysis weeks; the issue-data limitation)
E["share_commits_issue_linked"] = float(P.commits_linked.sum() / P.commits.sum())
E["share_linked_commits_bug"] = float(P.commits_bug.sum() / P.commits_linked.sum())
E["people_ge2_pr_weeks"] = int((P[P.prs_authored > 0].groupby("user_id").size() >= 2).sum())
E["team_paths_panel"] = int(X.loc[P.user_id.unique(), "team"].nunique())
E["team_paths_all"] = int(X.team.nunique())
users, non = X[X.has_usage == 1], X[X.has_usage == 0]
E["n_users"], E["n_nonusers"], E["n_people"] = len(users), len(non), len(X)
_tu = X.loc[P.user_id.unique(), "tokens"].sort_values(ascending=False)   # five link weeks, panel people
E["top10_token_share_users"] = float(_tu.head(int(np.ceil(len(_tu) * .1))).sum() / _tu.sum())
from inequality.gini import Gini as _Gini
E["gini_tokens_users"] = float(_Gini(_tu.to_numpy(float)).g)
E["share_nonusers"] = len(non) / len(X)
_pre = PW[PW.week.between(-7, -1)].groupby("user_id").commits.sum()
E["pre_commits_median_users"] = float(_pre[X.loc[_pre.index, "has_usage"] == 1].median())
E["pre_commits_median_nonusers"] = float(_pre[X.loc[_pre.index, "has_usage"] == 0].median())
E["pre_commits_mw_p"] = float(stats.mannwhitneyu(_pre[X.loc[_pre.index, "has_usage"] == 1], _pre[X.loc[_pre.index, "has_usage"] == 0]).pvalue)
E["adoption_by_role"] = {r: float(g.has_usage.mean()) for r, g in X.groupby("role") if len(g) >= 20}
E["first_day_users"] = int((pd.to_datetime(X.first_day) == pd.Timestamp("2026-06-29")).sum())
E["first_week_users"] = int(pd.to_datetime(X.first_day).between("2026-06-26", "2026-07-05").sum())

# raw telemetry: events, tool shares, local runs, generation share, sessions
u = pd.read_csv(RAW / "usages.csv", usecols=["user_id", "event_timestamp", "additional_data_tokens",
                                            "additional_data_tokens_context", "additional_data_tokens_generation",
                                            "spent", "tool", "tool_run_type"])
E["events_total"] = int(len(u))
E["event_users"] = int(u.user_id.nunique())
tok = u.additional_data_tokens
E["tool_token_share"] = {t: float(v) for t, v in (u.groupby("tool").additional_data_tokens.sum() / tok.sum()).items()}
_feat = pd.read_csv(RAW / "usages.csv", usecols=["feature_id"]).feature_id.astype(str)
E["agent_feature_token_share"] = float(tok[_feat.str.startswith("agent_route").values].sum() / tok.sum())   # gateway routes of coding agents
E["share_local_events"] = float(u.tool_run_type.eq("local").mean())
E["gen_share"] = float(u.additional_data_tokens_generation.sum() / tok.sum())
E["median_event_context"] = float(u.additional_data_tokens_context.median())
E["median_event_generation"] = float(u.additional_data_tokens_generation.median())
u["ts"] = pd.to_datetime(u.event_timestamp)
u = u.sort_values(["user_id", "ts"])
gap = u.groupby("user_id").ts.diff().dt.total_seconds() / 60
u["sess"] = (gap.isna() | (gap > 30)).cumsum()
sl = u.groupby("sess").agg(start=("ts", "min"), end=("ts", "max"), n=("ts", "size"))
sl["min"] = (sl.end - sl.start).dt.total_seconds() / 60
E["median_session_minutes"] = float(sl["min"].median())
E["median_session_events"] = float(sl.n.median())
E["share_events_long_sessions"] = float(u.sess.map(sl["min"] > 120).mean())
per = u.groupby("user_id").agg(tok=("additional_data_tokens", "sum"), cred=("spent", "sum"))
E["rho_tokens_credits"] = float(stats.spearmanr(per.tok, per.cred).correlation)
del u

# raw artifacts
cm = pd.read_csv(RAW / "commits.csv", usecols=["user_id", "source", "date", "additions", "deletions", "issue_ids"])
E["commits_raw"] = int(len(cm))
E["commits_internal_share"] = float(cm.source.eq("internal").mean())
rv = pd.read_csv(RAW / "reviews.csv")
E["pr_rows"] = int(len(rv))
E["issue_rows"] = int(len(pd.read_csv(RAW / "issues.csv", usecols=[0])))
au = rv[rv.roles.astype(str).str.contains("author")].copy()
au["d"] = pd.to_datetime(au.created)
auw = au[(au.d >= "2026-06-29") & (au.d <= "2026-08-02")]
linked = auw.issue_ids.notna() & auw.issue_ids.astype(str).str.len().gt(0)
E["authored_prs_window"] = int(len(auw))
E["authored_prs_naming_issue"] = float(linked.mean())

# ---------------------------------------------------------------- survey section (technical agent evaluations)
S = pd.read_csv(SURVEY); S.columns = [c.strip() for c in S.columns]
S["agent"] = S["Evaluated tool"].where(S["Evaluated tool"].isin(["Agent A", "Agent B"]), "other")
T = S[(S.Role == "Tech") & S.agent.isin(["Agent A", "Agent B"])]
E["tech_agent_evals"] = int(len(T))
E["tech_agent_a"] = int(T.agent.eq("Agent A").sum())
E["tech_agent_b"] = int(T.agent.eq("Agent B").sum())
E["tech_agent_people"] = int(T["Response ID"].nunique())
E["tech_agent_people_div_top"] = int(T.drop_duplicates("Response ID").Department.eq(X.team2.value_counts().idxmax()).sum())
E["tech_agent_evals_div_top"] = int(T.Department.eq(X.team2.value_counts().idxmax()).sum())
E["evals_by_tool"] = {k: int(v) for k, v in S["Evaluated tool"].value_counts().items()}
unmetered = S["Tool type"].isin(["chatbot", "enterprise search"])
E["evals_unmetered_chat_search"] = int(unmetered.sum())
E["evals_unmetered_chatbots_only"] = int(S["Tool type"].eq("chatbot").sum())
eff_q = {"quality": "Effect_quality of work output", "load": "Effect_cognitive load"}
for k, c in eff_q.items():
    v = T[c].value_counts(); n = v.sum()
    E[f"tech_agent_{k}_pos"] = float((v.get("Very positive", 0) + v.get("Somewhat positive", 0)) / n)
    E[f"tech_agent_{k}_neg"] = float((v.get("Very negative", 0) + v.get("Somewhat negative", 0)) / n)
# team leads vs ICs exposed to colleagues' AI output (assigned evaluations, answered)
A = S.drop_duplicates("Response ID")
ex = A[A["Exposure to AI generated materials"].notna()]
_lead = ex["Status.1"].astype(str)
E["lead_status_values"] = sorted(_lead.unique().tolist())
E["exposed_by_lead_status"] = {k: float(g["Exposure to AI generated materials"].eq("Yes").mean()) for k, g in ex.groupby(_lead)}

# ---------------------------------------------------------------- 5.1 coupling
pos = P[(P.commits > 0) & (P.tokens > 0)].copy()
pos = pos[pos.groupby("user_id").week.transform("size") >= 2]
for v, s in [("lc", "commits"), ("lt", "tokens")]:
    pos[v] = np.log(pos[s])
    pos[v + "_w"] = pos[v] - pos.groupby("user_id")[v].transform("mean")
    pos[v + "_w"] -= pos.groupby("week")[v + "_w"].transform("mean")
E["within_corr"] = float(np.corrcoef(pos.lc_w, pos.lt_w)[0, 1])
E["slope_commits_on_tokens"] = float(np.polyfit(pos.lt_w, pos.lc_w, 1)[0])
lc_all = np.log(pos.commits)
E["share_var_log_commits_within"] = float(pos.lc_w.var() / lc_all.var())
E["within_sd_log_commits"] = float(pos.lc_w.std())
E["share_weeks_double_or_half"] = float((pos.lc_w.abs() >= np.log(2)).mean())
r = pos.copy(); r["rt"] = np.exp(r.lt_w); r["rc"] = np.exp(r.lc_w)
E["rel_commits_when_spend_2x"] = float(r.loc[r.rt >= 2, "rc"].median())
E["rel_commits_when_spend_half"] = float(r.loc[r.rt <= .5, "rc"].median())


def ppml_d(df, out="commits"):
    d = df[(df[out] > 0)].copy()
    d = d[d.groupby("user_id").blk.transform("size") >= 2].reset_index(drop=True)
    b, _, _ = fe_poisson(d.tokens.values / 1e6, np.column_stack([np.log(d[out].values), week_dummies(d.blk)]), codes(d.user_id))
    return float(b[0])


D = PD[PD.user_id.isin(adopters) & PD.d.between("2026-06-29", "2026-08-02")].copy()
D["day"] = (D.d - pd.Timestamp("2026-06-29")).dt.days
D = D[(D.tokens > 0) | (D.commits > 0)]
for w, lab in [(1, "day"), (3, "3day")]:
    B = D.assign(blk=D.day // w).groupby(["user_id", "blk"], as_index=False)[["commits", "tokens"]].sum()
    B = B[B.tokens > 0]
    E[f"d_{lab}"] = ppml_d(B)

# instrumenting commits with lines (within person and week)
iv = pos[pos.lines > 0].copy()
iv["ll"] = np.log(iv.lines); iv["ll_w"] = iv.ll - iv.groupby("user_id").ll.transform("mean"); iv["ll_w"] -= iv.groupby("week").ll_w.transform("mean")
E["d_iv_lines"] = float(np.cov(iv.ll_w, iv.lt_w)[0, 1] / np.cov(iv.ll_w, iv.lc_w)[0, 1])

pm = P.groupby("user_id").agg(c=("commits", "mean"), t=("tokens", "mean"))
pos["cm"] = pos.user_id.map(pm.c); pos["tm"] = pos.user_id.map(pm.t)
E["slope_by_commit_level"] = {}
for lab, lo, hi in [("<5", 0, 5), ("5-12", 5, 12), ("12-25", 12, 25), (">25", 25, 1e9)]:
    g = pos[(pos.cm >= lo) & (pos.cm < hi)].assign(blk=lambda d: d.week)
    E["slope_by_commit_level"][lab] = ppml_d(g)
E["corr_by_token_tier"] = {}
for lab, lo, hi in [("<30M", 0, 30e6), ("30-100M", 30e6, 100e6), ("100-300M", 100e6, 300e6), (">300M", 300e6, 1e18)]:
    g = pos[(pos.tm >= lo) & (pos.tm < hi)]
    E["corr_by_token_tier"][lab] = float(np.corrcoef(g.lc_w, g.lt_w)[0, 1])
    if lab == ">300M":
        E["commits_on_tokens_top_tier"] = float(np.polyfit(g.lt_w, g.lc_w, 1)[0])
# reverse direction by Poisson, as the forward model: commits are the count outcome on log tokens,
# with person and week effects, on the forward model's weeks (both tokens and commits); the
# sensitivity check keeps the weeks with tokens and no commits
def ppml_rev(df):
    d = df[df.tokens > 0]
    d = d[d.groupby("user_id").week.transform("size") >= 2].reset_index(drop=True)
    b, _, _ = fe_poisson(d.commits.values.astype(float), np.column_stack([np.log(d.tokens.values), week_dummies(d.week)]), codes(d.user_id))
    return float(b[0]), len(d)
_P2 = P[P.commits > 0]
E["rev_ppml"], E["rev_ppml_n"] = ppml_rev(_P2)
E["rev_ppml_zeros"], E["rev_ppml_zeros_n"] = ppml_rev(P)
_rng = np.random.default_rng(20261015); _g = {u: d for u, d in _P2.groupby("user_id")}; _ids = np.array(list(_g)); _rb = []
for _ in range(999):
    _s = _rng.choice(_ids, len(_ids))
    _rb.append(ppml_rev(pd.concat([_g[u].assign(user_id=i) for i, u in enumerate(_s)]))[0])
E["rev_ppml_ci"] = [float(x) for x in np.percentile(_rb, [2.5, 97.5])]
E["rev_ppml_top_tier"] = ppml_rev(_P2[_P2.user_id.isin(pm.index[pm.t >= 300e6])])[0]
# flattening: tokens relative to the person's mean at a third and at 2.5x the usual commits
bins = pd.cut(np.exp(pos.lc_w), [0, .45, .75, 1.33, 2.0, 3.2, 1e9])
E["rel_tokens_by_rel_commits"] = {str(k): float(v) for k, v in np.exp(pos.lt_w).groupby(bins, observed=True).mean().items()}

# between people
Ab = P[(P.tokens > 0) & (P.commits > 0)].copy()
Ab = Ab[Ab.groupby("user_id").week.transform("size") >= 2]
Ab["ly"], Ab["lx"] = np.log10(Ab.tokens), np.log10(Ab.commits)
for v in ["ly", "lx"]:
    Ab[v] -= Ab.groupby("week")[v].transform("mean") - Ab[v].mean()
bp = Ab.groupby("user_id").agg(lx=("lx", "mean"), ly=("ly", "mean"), n=("week", "size"))
E["between_slope_unweighted"] = float(np.polyfit(bp.lx, bp.ly, 1)[0])
E["between_slope_weighted"] = float(np.polyfit(bp.lx, bp.ly, 1, w=bp.n)[0])
E["between_commits_on_tokens"] = float(np.polyfit(bp.ly, bp.lx, 1)[0])
# Mundlak: within slope and the between-minus-within difference, person bootstrap
def _mund(d):
    d = d.copy(); d["xm"] = d.groupby("user_id").lx.transform("mean"); d["xw"] = d.lx - d.xm
    M = np.column_stack([np.ones(len(d)), d.xw, d.xm]); b = np.linalg.lstsq(M, d.ly.to_numpy(), rcond=None)[0]
    return b[2] - b[1]
E["between_minus_within"] = float(_mund(Ab))
_bb = []
_rngb = np.random.default_rng(20261006); _idb = bp.index.to_numpy()
for _ in range(1000):
    _pk = bp.loc[_rngb.choice(_idb, len(_idb))]
    _bb.append(np.polyfit(_pk.lx, _pk.ly, 1)[0])
E["between_slope_ci"] = [float(np.percentile(_bb, 2.5)), float(np.percentile(_bb, 97.5))]
_rng = np.random.default_rng(20261005); _ids = Ab.user_id.unique(); _g = {u_: d_ for u_, d_ in Ab.groupby("user_id")}; _bs = []
for _ in range(300):
    pick = _rng.choice(_ids, len(_ids))
    _bs.append(_mund(pd.concat([_g[u_].assign(user_id=f"{u_}#{j}") for j, u_ in enumerate(pick)])))
E["between_minus_within_ci"] = [float(np.percentile(_bs, 2.5)), float(np.percentile(_bs, 97.5))]
# Poisson check on the between-person slopes: fits the mean, so heavy spenders weigh more
_pb = Ab.assign(lc=Ab.lx * np.log(10), lt=Ab.ly * np.log(10)).groupby("user_id").agg(
    lc=("lc", "mean"), lt=("lt", "mean"), t=("tokens", "mean"), c=("commits", "mean")).sort_values("t", ascending=False)
_one = lambda n_: np.zeros(n_, dtype=int)
_pf = lambda d: float(fe_poisson(d["t"].values / 1e6, d[["lc"]].values, _one(len(d)))[0][0])
E["between_ppml"] = _pf(_pb)
E["between_ppml_rev"] = float(fe_poisson(_pb["c"].values, _pb[["lt"]].values, _one(len(_pb)))[0][0])
E["between_top5_spend_share"] = float(_pb["t"].head(5).sum() / _pb["t"].sum())
E["between_ppml_drop5"] = _pf(_pb.iloc[5:])
E["between_ols_drop5"] = float(np.polyfit(_pb["lc"].iloc[5:], _pb["lt"].iloc[5:], 1)[0])
def _mund_ppml(d):
    d = d.copy(); d["lc"] = np.log(d.commits); d["m"] = d.groupby("user_id").lc.transform("mean")
    b = fe_poisson(d.tokens.values / 1e6, np.column_stack([d.lc - d.m, d.m, week_dummies(d.week)]), _one(len(d)))[0]
    return float(b[1] - b[0])
E["between_minus_within_ppml"] = _mund_ppml(Ab)
_bs2 = []
for _ in range(499):
    pick = _rng.choice(_ids, len(_ids))
    _bs2.append(_mund_ppml(pd.concat([_g[u_].assign(user_id=f"{u_}#{j}") for j, u_ in enumerate(pick)])))
E["between_minus_within_ppml_ci"] = [float(np.percentile(_bs2, 2.5)), float(np.percentile(_bs2, 97.5))]

# ---------------------------------------------------------------- 5.2 redundancy
pl = P[P.tokens > 0].groupby("user_id").agg(lspend=("tokens", lambda s: np.log(s.mean())), tok=("tokens", "median")).dropna()   # active weeks
pl = pl.join(X[["role", "manager", "team2", "team3"]])
E["median_weekly_tokens_by_role_m"] = {k: float(v / 1e6) for k, v in pl.groupby("role").tok.median().items()}
dev = pl[pl.role == "dev"].tok
E["dev_iqr_ratio"] = float(dev.quantile(.75) / dev.quantile(.25))


def r2_cv(y, M, folds=10, seed=20261005):
    M = np.column_stack([np.ones(len(y)), M]) if M.size else np.ones((len(y), 1))
    b = np.linalg.lstsq(M, y, rcond=None)[0]; ins = 1 - ((y - M @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(seed); f = rng.permutation(len(y)) % folds; sse = sst = 0.0
    for k in range(folds):
        tr, te = f != k, f == k
        bb = np.linalg.lstsq(M[tr], y[tr], rcond=None)[0]
        sse += ((y[te] - M[te] @ bb) ** 2).sum(); sst += ((y[te] - y[tr].mean()) ** 2).sum()
    return float(ins), float(1 - sse / sst)


dm = lambda s: pd.get_dummies(s, drop_first=True).astype(float).to_numpy()
y = pl.lspend.to_numpy()
E["r2_role_mgr"] = r2_cv(y, np.column_stack([dm(pl.role), pl[["manager"]].to_numpy(float)]))
E["r2_divisions_raw"] = r2_cv(y, dm(pl.team2))
big2 = pl.team2.where(pl.team2.map(pl.team2.value_counts()) >= 10, "pooled")
big3 = pl.team3.where(pl.team3.map(pl.team3.value_counts()) >= 10, "pooled")
E["n_divisions"] = int(pl.team2.nunique())
E["n_divisions_ge10"] = int((pl.team2.value_counts() >= 10).sum())
E["n_ecosystems_ge10"] = int((pl.team3.value_counts() >= 10).sum())
E["r2_divisions_ge10"] = r2_cv(y, dm(big2))
E["r2_ecosystems_ge10"] = r2_cv(y, dm(big3))
E["r2_all_groupings"] = r2_cv(y, np.column_stack([dm(big2), dm(big3)]))
for lvl, col in [("division", "team2"), ("ecosystem", "team3"), ("team", None)]:
    grp = X.loc[pl.index, "team"] if col is None else pl[col]
    s, n = pl.lspend.groupby(grp).transform("sum"), pl.lspend.groupby(grp).transform("size")
    loo = (s - pl.lspend) / (n - 1)
    ok = n > 1
    E[f"loo_corr_{lvl}"] = float(np.corrcoef(pl.lspend[ok], loo[ok])[0, 1])
tm = pl.groupby(X.loc[pl.index, "team3"]).tok.agg(["median", "size"])
tm = tm[tm["size"] >= 10]
E["ecosystem_median_tokens_m"] = {k: float(v / 1e6) for k, v in tm["median"].items()}
E["ecosystem_median_min"] = [tm["median"].idxmin(), float(tm["median"].min() / 1e6)]
E["ecosystem_median_max"] = [tm["median"].idxmax(), float(tm["median"].max() / 1e6)]

# ---------------------------------------------------------------- 5.3 dark spend
for form in ["counts", "sqrt", "log1p", "spline"]:
    E[f"ceiling_away_in_{form}"], _, _ = ceiling(ALL.assign(tok_m=ALL.tokens / 1e6, lines_k=ALL.lines / 1e3,
                                                             pr_lines_k=ALL.pr_lines / 1e3)
                                                  .loc[lambda d: d.groupby("user_id").tokens.transform("sum") > 0],
                                                  COUNTS + SIZES, form)
    # the same fit on the panel (away weeks dropped), so the two differ only in the away weeks
    E[f"ceiling_away_out_{form}"], _, _ = ceiling(P.assign(tok_m=P.tokens / 1e6, lines_k=P.lines / 1e3,
                                                           pr_lines_k=P.pr_lines / 1e3)
                                                   .loc[lambda d: d.groupby("user_id").tokens.transform("sum") > 0],
                                                   COUNTS + SIZES, form)
# relative prediction for weeks with no commit (the figure's fit: commits and PRs)
P["m_i"] = P.groupby("user_id").tokens.transform("mean") / 1e6
for form in ["counts", "sqrt", "log1p", "spline"]:
    _, f0, mu = ceiling(P, COUNTS, form)
    mh = pd.Series(mu).groupby(P.user_id).transform("mean").to_numpy()
    z = (P.commits == 0).to_numpy()
    E[f"zero_commit_week_pred_{form}"] = float((mu / mh)[z].mean())
E["zero_commit_week_obs"] = float((P.tok_m / P.m_i)[P.commits == 0].mean())
# spline component per person and its concentration
_, f0, _ = ceiling(P, COUNTS, "spline")          # the figures' specification (commits and PRs)
P["f0"] = f0
pp = P.groupby("user_id").agg(f0=("f0", "sum"), y=("tok_m", "sum"))
pp = pp[pp.y > 0]; pp["s"] = pp.f0 / pp.y
E["person_spline_share_q"] = {str(q): float(pp.s.quantile(q)) for q in [.1, .5, .9]}
top = pp.sort_values("f0", ascending=False).head(int(np.ceil(len(pp) * .1)))
E["top10_component_share"] = float(top.f0.sum() / pp.f0.sum())
E["top10_component_tokens_share"] = float(top.y.sum() / pp.y.sum())
for lab, wk in [("w0_1", [0, 1]), ("w2_4", [2, 3, 4])]:
    Q = P[P.week.isin(wk)]
    Q = Q[Q.groupby("user_id").tokens.transform("sum") > 0]
    E[f"ceiling_spline_{lab}"] = ceiling(Q, COUNTS, "spline")[0]["share"]
E["markup_spline"] = float(P.f0.sum() / (P.tok_m.sum() - P.f0.sum()))
_X2 = X.loc[P.user_id.unique()]
_G = P.join(_X2[["role", "manager", "team2"]], on="user_id")
_G["usage"] = _G.user_id.map(pd.qcut(_X2.tokens, 3, labels=["lightest", "middle", "heaviest"]))
_dom = pd.Series(np.where(_X2.share_agent_a > 0.9, "Agent A only", np.where(_X2.share_agent_b > 0.9, "Agent B only", "both")), index=_X2.index)
_G["tool"] = _G.user_id.map(_dom)
E["group_share"] = {}
for col in ["role", "manager", "team2", "usage", "tool"]:
    for lab_, g_ in _G.groupby(col, observed=True):
        E["group_share"][f"{col}:{lab_}"] = [float(g_.f0.sum() / g_.tok_m.sum()), float(g_.tok_m.sum() / P.tok_m.sum())]
# next-day rebound after heavy / light / empty / productive weekdays
days = pd.date_range("2026-06-29", "2026-08-02")
grid = pd.MultiIndex.from_product([adopters, days], names=["user_id", "d"])
DG = PD.set_index(["user_id", "d"]).reindex(grid, fill_value=0).reset_index()
DG = DG[DG.d.dt.dayofweek < 5].sort_values(["user_id", "d"])
DG["n1"] = DG.groupby("user_id").commits.shift(-1)
DG["pm"] = DG.groupby("user_id").commits.transform("mean")
DG = DG.dropna(subset=["n1"]); DG = DG[DG.pm > 0]
thr = DG.loc[DG.tokens > 0, "tokens"].median()
cat = np.select([(DG.commits == 0) & (DG.tokens > thr), (DG.commits == 0) & (DG.tokens > 0),
                 (DG.commits == 0) & (DG.tokens == 0), DG.commits > 0],
                ["heavy", "light", "nothing", "productive"], "other")
E["next_day_rel_commits"] = {k: float(v) for k, v in (DG.n1 / DG.pm).groupby(cat).mean().items()}

# ---------------------------------------------------------------- 5.5 stability: top-decile and the rank example
odd = P[P.week.isin([1, 3])].groupby("user_id")[["tokens", "commits"]].sum()
even = P[P.week.isin([0, 2, 4])].groupby("user_id")[["tokens", "commits"]].sum()
both = odd.join(even, lsuffix="_o", rsuffix="_e").dropna()
both = both[(both.commits_o > 0) & (both.commits_e > 0) & (both.tokens_o > 0) & (both.tokens_e > 0)]
both["tpc_o"] = both.tokens_o / both.commits_o; both["tpc_e"] = both.tokens_e / both.commits_e
k = int(np.ceil(len(both) * .1))
to, te = set(both.tpc_o.nlargest(k).index), set(both.tpc_e.nlargest(k).index)
E["tpc_n"], E["tpc_top_k"], E["tpc_top_overlap"] = len(both), k, len(to & te)
ro, re_ = both.tpc_o.rank(ascending=False), both.tpc_e.rank(ascending=False)
jump = (re_ - ro)[ro <= 5].sort_values(ascending=False)
E["rank_example"] = {"odd_rank": int(ro[jump.index[0]]), "even_rank": int(re_[jump.index[0]]), "n": len(both)}
_ex = P[(P.user_id == jump.index[0]) & P.week.isin([1, 3])]
E["rank_example_odd_weeks"] = [[int(c_), float(t_ / 1e6)] for c_, t_ in zip(_ex.commits, _ex.tokens)]
wtok = P.pivot_table(index="user_id", columns="week", values="tokens")
E["week_pair_rho"] = {f"{i}-{j}": float(stats.spearmanr(*wtok[[i, j]].dropna().where(lambda d: (d > 0).all(axis=1)).dropna().T.values).correlation)
                      for i in range(5) for j in range(i + 1, 5)}

# ---------------------------------------------------------------- 5.6 heavy users' output, pre vs telemetry
tert = pd.qcut(X.loc[P.user_id.unique(), "tokens"], 3, labels=["light", "medium", "heavy"])
heavy = tert[tert == "heavy"].index
pre = PW[PW.week.between(-7, -1)].groupby("user_id").commits.mean()
post = PW[PW.week.between(0, 4)].groupby("user_id").commits.mean()
E["heavy_mean_commits_pre"], E["heavy_mean_commits_post"] = float(pre[heavy].mean()), float(post[heavy].mean())
E["heavy_median_pre"], E["heavy_median_post"] = float(pre[heavy].median()), float(post[heavy].median())
nonu = X.index[X.has_usage == 0]
E["non_median_pre"], E["non_median_post"] = float(pre[nonu].median()), float(post[nonu].median())
E["heavy_share_rose"] = float((post[heavy] > pre[heavy]).mean())
E["non_share_rose"] = float((post[nonu] > pre[nonu]).mean())

# ---------------------------------------------------------------- the two data corrections
cm["d"] = pd.to_datetime(cm.date)
sig = (cm.user_id + "|" + cm.date + "|" + cm.additions.astype(str) + "|" + cm.deletions.astype(str) + "|" + cm.issue_ids.fillna(""))
_k = sig.groupby([sig, cm.source]).cumcount()
cm["mirror"] = cm.source.eq("github") & (_k < sig.map(sig[cm.source.eq("internal")].value_counts()).fillna(0))
E["mirror_n"] = int(cm.mirror.sum())
_pt = pd.read_csv(RAW / "commits.csv", usecols=["product_tag"]).product_tag
cm["direct"] = _pt.eq("direct").values
_gd = cm.source.eq("github") & cm.direct & (cm.d >= "2026-06-01")
E["mirror_share_github_direct"] = float(cm.loc[_gd, "mirror"].mean())
_rng = np.random.default_rng(20261005)
_sh = cm.copy()
_sh["date"] = _sh.groupby("user_id").date.transform(lambda v: v.sample(frac=1, random_state=_rng.integers(1 << 30)).values)
_sig = (_sh.user_id + "|" + _sh.date + "|" + _sh.additions.astype(str) + "|" + _sh.deletions.astype(str) + "|" + _sh.issue_ids.fillna(""))
_k2 = _sig.groupby([_sig, _sh.source]).cumcount()
_m2 = _sh.source.eq("github") & (_k2 < _sig.map(_sig[_sh.source.eq("internal")].value_counts()).fillna(0))
E["mirror_share_chance"] = float(_m2[_gd].mean())
_internal_sigs = set(sig[cm.source.eq("internal")])
E["twin_exists_github_direct"] = float(sig[_gd].isin(_internal_sigs).mean())
E["twin_exists_chance"] = float(_sig[_gd].isin(set(_sig[_sh.source.eq("internal")])).mean())
_gh = cm[cm.source.eq("github") & (cm.d >= "2026-05-11")]
_ghw = _gh.groupby((_gh.d - pd.Timestamp("2026-06-29")).dt.days // 7).size()
E["github_weekly_commits"] = {int(k): int(v) for k, v in _ghw.items()}
dedup = cm[~cm.mirror]
_day = dedup.groupby("d").size()
_ref = _day[(_day.index >= "2026-05-18") & (_day.index <= "2026-06-28")]
_dow = _ref.groupby(_ref.index.dayofweek).mean()
_S = {}
for kk in range(1, 61):
    ds = [pd.Timestamp("2026-05-10") - pd.Timedelta(days=kk + j) for j in (-1, 0, 1)]
    _S[kk] = sum(_day.get(x_, 0) for x_ in ds) / sum(_dow[x_.dayofweek] for x_ in ds)
_S = pd.Series(_S).clip(upper=1).cummin()
E["lag_survival_7d"] = float(_S[7])
# sensitivity: skip Friday 1 May (public holiday in much of Europe) when measuring delays
_HOL = {pd.Timestamp("2026-05-01")}
_Sh = {}
for kk in range(1, 61):
    ds = [pd.Timestamp("2026-05-10") - pd.Timedelta(days=kk + j) for j in (-1, 0, 1)]
    ds = [x_ for x_ in ds if x_ not in _HOL] or ds
    _Sh[kk] = sum(_day.get(x_, 0) for x_ in ds) / sum(_dow[x_.dayofweek] for x_ in ds)
_Sh = pd.Series(_Sh).clip(upper=1).cummin()
_end = pd.Timestamp("2026-08-08")
def _compl(w):
    dd = [pd.Timestamp("2026-06-29") + pd.Timedelta(days=7 * w + j) for j in range(7)]
    v = [1 - _S.get((_end - x_).days, 0.0) if (_end - x_).days >= 1 else 1 - _S.iloc[0] for x_ in dd if x_ <= _end]
    return float(np.mean(v))
E["week_completeness"] = [_compl(w) for w in range(6)]
_S_main, _S = _S, _Sh
E["week_completeness_no_may1"] = [_compl(w) for w in range(6)]
_S = _S_main
# the two automated-looking accounts
_pdc = cm[cm.d.between("2026-05-10", "2026-08-08")].groupby(["user_id", "d"]).size()
_bulk_user, _bulk_day = _pdc.idxmax()
E["bulk_day_commits"] = int(_pdc.max()); E["bulk_day_weekday"] = pd.Timestamp(_bulk_day).day_name()
_nxt = _pdc.loc[_bulk_user]; _nxt = _nxt[_nxt.index > _bulk_day]
E["bulk_next_day_commits"] = int(_nxt.iloc[0]); E["bulk_next_day_weekday"] = pd.Timestamp(_nxt.index[0]).day_name()
_wtot = PW.groupby("user_id").tokens.sum()
_agent_user = _wtot.idxmax()
_aw = PW[PW.user_id == _agent_user]
E["agent_account_pre_weekly_commits"] = float(_aw[_aw.week < 0].commits.mean())
E["agent_account_post_weekly_commits"] = float(_aw[_aw.week.between(0, 4)].commits.mean())
E["agent_account_max_day"] = int(_pdc.loc[_agent_user].max())
E["agent_account_is_top_spender"] = bool(_agent_user == X.loc[P.user_id.unique(), "tokens"].idxmax())
_Aw = P[(P.tokens > 0) & (P.commits > 0)].copy(); _Aw = _Aw[_Aw.groupby("user_id").week.transform("size") >= 2]
def _bw(d):
    d = d.copy(); d["ly"], d["lx"] = np.log10(d.tokens), np.log10(d.commits)
    for v in ["ly", "lx"]:
        d[v] -= d.groupby("week")[v].transform("mean") - d[v].mean(); d[v] -= d.groupby("user_id")[v].transform("mean")
    return float((d.ly * d.lx).sum() / (d.lx ** 2).sum())
E["within_slope_fig1"] = _bw(_Aw)
E["within_slope_fig1_without_two"] = _bw(_Aw[~_Aw.user_id.isin([_bulk_user, _agent_user])])

# 5.6: heavy users' commit rise relative to non-adopters (Fig. 6B's series), with and
# without the two automated-looking accounts, at the third metered week
_grp = pd.Series(np.nan, index=X.index, dtype=object); _grp[nonu] = "non-adopter"; _grp[tert.index] = tert.astype(str)
_idx = PW.assign(group=PW.user_id.map(_grp), pre=PW.user_id.map(pre)); _idx = _idx[(_idx.pre > 0) & _idx.group.notna()]
def _rel(fr):
    gm = fr.groupby(["group", "week"]).commits.sum().unstack(0)
    gp = fr[fr.week < 0].groupby(["group", "week"]).commits.sum().unstack(0).mean()
    r = gm / gp; return r.div(r["non-adopter"], axis=0)
E["rel_heavy_week3"] = float(_rel(_idx).loc[3, "heavy"])
E["rel_heavy_week3_without_two"] = float(_rel(_idx[~_idx.user_id.isin([_bulk_user, _agent_user])]).loc[3, "heavy"])

# the commit counts of Sec. corrections, after the mirror removal
_keep = ~cm.mirror
# the 90-day telemetry window starts 2026-05-10; the five link weeks sit inside it
E["commits_pre_window"] = int((_keep & (cm.d < "2026-05-10")).sum())
E["commits_from_window_start"] = int((_keep & (cm.d >= "2026-05-10")).sum())
E["commits_naming_issue_share"] = float(cm.loc[_keep & (cm.d >= "2026-05-10"), "issue_ids"].notna().mean())
# the new model's share of spend
_u2 = pd.read_csv(RAW / "usages.csv", usecols=["event_date", "spent", "additional_data_model_name"], parse_dates=["event_date"])
_o5 = _u2.additional_data_model_name.astype(str).eq("model_02")      # the new frontier model
E["new_model_first_day"] = str(_u2.loc[_o5, "event_date"].min().date())
_wkv = (_u2.event_date - pd.Timestamp("2026-06-29")).dt.days // 7
E["new_model_share_by_week"] = {int(k): float(v) for k, v in (_u2.spent.where(_o5, 0).groupby(_wkv).sum() / _u2.spent.groupby(_wkv).sum()).items() if k >= 0}
del _u2

# ---------------------------------------------------------------- survey: technical agent users, with the companion coding
_cwd = os.getcwd(); os.chdir(W)
_ns2 = {}
exec((W / "analysis" / "survey_crossq.py").read_text().split("outcomes = [")[0], _ns2)
os.chdir(_cwd)
TT = _ns2["T"]
act_cols = [c for c in TT.columns if c.startswith("PWeek_Act_") and not c.endswith("Other")]
val_cols = [c for c in TT.columns if c.startswith("PWeek_Value_") and not c.endswith("Other")]
DQ = TT[TT[val_cols + act_cols].notna().any(axis=1)]
E["agent_past_week_n"] = int(len(DQ))
E["agent_effect_positive"] = {}
for _c in ["Effect_overall productivity", "Effect_efficiency in completing complex tasks", "Effect_quality of work output", "Effect_cognitive load"]:
    _v = TT[_c].value_counts(); E["agent_effect_positive"][_c] = float((_v.get("Very positive", 0) + _v.get("Somewhat positive", 0)) / _v.sum())
E["agent_dir"] = {k: float(v) for k, v in DQ.Net_effect.value_counts(normalize=True).items()}
sv = TT.net_hours[TT.net_hours > 0]
E["agent_hours_savers_median"] = float(sv.median()); E["agent_hours_savers_iqr"] = [float(sv.quantile(.25)), float(sv.quantile(.75))]
E["agent_impossible_share_savers"] = float(TT.loc[TT.Net_effect.eq("Saved me time"), "Hours saved"].astype(str).eq("Impossible to tell").mean())
E["agent_hours_median_a"] = float(TT.loc[TT.is_agent_a == 1, "net_hours"].median())
E["agent_hours_median_b"] = float(TT.loc[TT.is_agent_a == 0, "net_hours"].median())
E["agent_act_share"] = {c.replace("PWeek_Act_", ""): float(DQ[c].notna().mean()) for c in act_cols}
E["agent_any_noncoding"] = float((DQ[[c for c in act_cols if "Coding" not in c]].notna().sum(axis=1) > 0).mean())
# the same share on assigned evaluations only (one per respondent; the chosen second tool is self-selected)
_S = _ns2["S"].assign(occ=lambda d: d.groupby("Response ID").cumcount() + 1)
_S["key"] = _S["Response ID"].astype(str) + "-" + _S.agent
assert not _S.loc[_S.agent.isin(["A", "B"]), "key"].duplicated().any()
_DA = DQ[DQ["key"].map(_S.drop_duplicates("key").set_index("key").occ).eq(1)]
E["agent_any_noncoding_assigned"] = float((_DA[[c for c in act_cols if "Coding" not in c]].notna().sum(axis=1) > 0).mean())
E["agent_past_week_n_assigned"] = int(len(_DA))
E["agent_n_activities_median"] = float(DQ[act_cols].notna().sum(axis=1).median())
E["agent_val_share"] = {c.replace("PWeek_Value_", ""): float(DQ[c].notna().mean()) for c in val_cols}
only = DQ["PWeek_Value_Experiments and learning"].notna() & DQ[[v for v in val_cols if "Experiments" not in v]].isna().all(axis=1)
E["agent_value_only_experiments"] = float(only.mean())
E["agent_enab_share"] = float(TT.enab_month.mean())
E["agent_hours_by_noncoding"] = {str(k): float(v) for k, v in TT.groupby(pd.cut(TT.n_noncoding, [-1, 0, 1, 99]), observed=True).net_hours.median().items()}
E["agent_hours_by_colleagues"] = {str(k): float(v) for k, v in TT.groupby(pd.cut(TT.colleagues_n, [-1, 0, 1, 2, 99]), observed=True).net_hours.median().items()}
E["agent_rho_conf_hours"] = float(TT[["confidence", "net_hours"]].corr("spearman").iloc[0, 1])
E["agent_wrong_prod"] = {str(k): float(v) for k, v in TT.groupby("L_WRONG").E_productivity.mean().items()}
E["agent_wrong_hours"] = {str(k): float(v) for k, v in TT.groupby("L_WRONG").net_hours.median().items()}

# ---------------------------------------------------------------- survey: restored from the companion draft
import sys as _sys
_sys.path.insert(0, str(HERE / "survey"))
import common as _sc
_df = _sc.load()
_A = _df[_df.occurrence == 1]
_ex = _A[_A["Exposure to AI generated materials"].eq("Yes") & _A.W_r.notna()]
E["rework_share_by_lead"] = {k: float((g.W_r > 0).mean()) for k, g in _ex.groupby("TeamLead")}
E["rework_n_by_lead"] = {k: int(len(g)) for k, g in _ex.groupby("TeamLead")}
# assigned vs chosen, raw means, without answers at the 30-hour cap
_p = _df[_df["Response ID"].isin(_df.loc[_df.is_second_tool, "Response ID"])]
_h = _p.pivot_table(index="Response ID", columns="occurrence", values="H_net", aggfunc="first")
_t = _p.pivot_table(index="Response ID", columns="occurrence", values="H_net_topcoded", aggfunc="first").astype(float)
_k = _h.notna().all(axis=1) & (_t.reindex(_h.index).fillna(0).sum(axis=1) == 0)
E["chosen_raw_nocap_n"] = int(_k.sum())
E["chosen_raw_nocap_assigned"], E["chosen_raw_nocap_chosen"] = float(_h.loc[_k, 1].mean()), float(_h.loc[_k, 2].mean())
E["chosen_raw_nocap_diff"] = float((_h.loc[_k, 2] - _h.loc[_k, 1]).mean())

# overlap of the title-based role "team leads and managers" with the directory's manager flag (token users)
_lm = X[(X.has_usage == 1) & (X.role == "lead_mgmt")]
E["lead_mgmt_flagged_manager_share"] = float(_lm.manager.eq(1).mean())

(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "extras.json").write_text(json.dumps(E, indent=1, default=float))
for kk, vv in E.items():
    print(kk, json.dumps(vv, default=float)[:160])
