"""Person-week analysis of the dated extract.

Reads analysis/out/person_week.csv and people.csv (build_dataset.py) and
writes analysis/out/results_panel.json and residuals_panel.csv.

Panel: adopters x telemetry weeks 0..4 (Mon 2026-06-29 .. Sun 2026-08-02).
Week 5 (Aug 3-8) has six days and loses about a third of its commits to
right truncation, so it is excluded except where stated; weeks -7..-1 carry
artifacts but no telemetry and supply pre-adoption covariates.

Person terms enter as fixed effects, concentrated out of the Poisson
likelihood (the plan's correlated random effect gives the same within-person
slopes; the fixed-effect version is what numpy alone supports exactly). Week
terms absorb the common right-truncation of commit counts.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glm import irls_poisson, poisson_deviance  # noqa: E402

OUT = "analysis/out/"
# Panel rule (agreed 2026-10-06, PRs dropped 2026-10-23): a person-week is in
# the panel only if it has anything recorded: tokens or a commit. Output is
# commits only; GitHub PRs cover a minority of engineers (most code review
# happens on the internal host, which the extract lacks). Weeks with neither are weeks the
# person was away; modelling them as zero-output
# working weeks pulls the fitted spend at zero output down and breaks the
# calibration check. PANEL_RULE=all restores the earlier behaviour.
RULE = os.environ.get("PANEL_RULE", "any")
def any_recorded(df):
    return (df.tokens > 0) | (df.commits > 0)
rng = np.random.default_rng(20261005)
B = 300                                       # bootstrap replicates (persons)
WEEKS = [0, 1, 2, 3, 4]


# ================================================================ estimators

def fe_poisson(y, X, g, max_iter=100, tol=1e-10):
    """Poisson with group fixed effects concentrated out (conditional /
    multinomial likelihood; exact for Poisson). X must not contain an
    intercept. Returns beta, group effects c (log scale, -inf for all-zero
    groups), and fitted mu."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    g = np.asarray(g)
    G = g.max() + 1
    k = X.shape[1]
    n_g = np.bincount(g, weights=y, minlength=G)
    b = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ b
        m = np.full(G, -np.inf)
        np.maximum.at(m, g, eta)
        w = np.exp(eta - m[g])
        s = np.bincount(g, weights=w, minlength=G)
        p = w / s[g]
        grad = X.T @ (y - n_g[g] * p)
        xbar = np.column_stack([np.bincount(g, weights=p * X[:, j], minlength=G)
                                for j in range(k)])
        H = (X * (n_g[g] * p)[:, None]).T @ X - (xbar * n_g[:, None]).T @ xbar
        step = np.linalg.solve(H + 1e-10 * np.eye(k), grad)
        b = b + step
        if np.max(np.abs(step)) < tol:
            break
    eta = X @ b
    s = np.bincount(g, weights=np.exp(eta), minlength=G)
    with np.errstate(divide="ignore"):
        c = np.log(n_g) - np.log(s)
    mu = np.where(np.isfinite(c[g]), np.exp(eta + np.where(np.isfinite(c), c, 0)[g]), 0.0)
    return b, c, mu


def codes(s):
    return pd.factorize(s)[0]


def week_dummies(w, ref=None):
    w = np.asarray(w)
    lv = sorted(set(w))
    ref = lv[0] if ref is None else ref
    return np.column_stack([(w == v).astype(float) for v in lv if v != ref]) \
        if len(lv) > 1 else np.zeros((len(w), 0))


def ols(y, X):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return b, 1 - e.var() / y.var(), e


def boot_persons(df, fn, reps=B):
    """Person-cluster bootstrap: resample persons, relabel duplicates."""
    ids = df.user_id.unique()
    groups = {u: d for u, d in df.groupby("user_id")}
    vals = []
    for r in range(reps):
        pick = rng.choice(ids, len(ids), replace=True)
        bd = pd.concat([groups[u].assign(user_id=f"{u}#{j}")
                        for j, u in enumerate(pick)], ignore_index=True)
        try:
            vals.append(fn(bd))
        except Exception:
            continue
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def sb(r):
    return 2 * r / (1 + r)


# ================================================================ data

X = pd.read_csv(OUT + "people.csv", index_col=0)
PW = pd.read_csv(OUT + "person_week.csv")
PW["tok_m"] = PW["tokens"] / 1e6
PW["lines_k"] = PW["lines"] / 1e3
PW["pr_lines_k"] = PW["pr_lines"] / 1e3
PW["output_any"] = (PW.commits > 0).astype(int)

# pre-adoption covariates from weeks -7..-1 (no telemetry, artifacts dated)
pre = PW[PW.week < 0].groupby("user_id").agg(
    pre_commits=("commits", "mean"), pre_lines=("lines", "sum"),
    pre_bug=("commits_bug", "sum"), pre_prs=("prs_authored", "mean"),
    pre_zero_weeks=("commits", lambda v: (v == 0).mean()))
pre["pre_lines_per_commit"] = pre.pre_lines / pre.pre_commits.mul(7).replace(0, np.nan)
pre["pre_bug_share"] = pre.pre_bug / pre.pre_commits.mul(7).replace(0, np.nan)
X = X.join(pre)

adopters = X.index[X.has_usage == 1]
P = PW[PW.user_id.isin(adopters) & PW.week.isin(WEEKS)].copy()
if RULE == "any":
    # one rule everywhere: drop only person-weeks with nothing recorded. A person
    # whose metered use fell outside the window keeps their output weeks; their
    # zero-token weeks add nothing to any token sum or likelihood.
    P = P[any_recorded(P)].reset_index(drop=True)
else:
    # drop adopters with no metered use in weeks 0..4 (their FE is -inf)
    keep = P.groupby("user_id").tokens.transform("sum") > 0
    P = P[keep].reset_index(drop=True)
R = {}
R["panel"] = dict(
    n_adopters=int(len(adopters)), n_people_panel=int(P.user_id.nunique()),
    n_person_weeks=int(len(P)), weeks=WEEKS,
    share_active=float((P.tokens > 0).mean()),
    median_commits_active_week=float(P.loc[P.tokens > 0, "commits"].median()),
    share_zero_commit_active=float((P.loc[P.tokens > 0, "commits"] == 0).mean()),
    share_zero_output_active=float((P.loc[P.tokens > 0, "output_any"] == 0).mean()),
    median_pos_commit_weeks=float(P[(P.commits > 0) & (P.tokens > 0)]
                                  .groupby("user_id").size().median()),
    share_weeks_with_pr=float((P.prs_authored > 0).mean()),
    people_ge2_pr_weeks=int((P[(P.prs_authored > 0) & (P.tokens > 0)]
                             .groupby("user_id").size() >= 2).sum()),
    weekly_mean_commits_all584=PW.groupby("week").commits.mean().round(2).to_dict(),
)


# ================================================================ RQ1

def output_terms(df, cols, form, knots=None):
    S = df[cols].to_numpy(float)
    if form == "log1p":
        return np.log1p(S)
    if form == "sqrt":
        return np.sqrt(S)
    if form == "spline":                       # linear spline in log1p, knot at
        L = np.log1p(S)                        # the median of positive weeks
        return np.column_stack([L, np.maximum(L - np.log1p(knots), 0)])
    return S


COUNTS = ["commits"]
SIZES = ["lines_k"]
KN = P.loc[P.commits > 0, COUNTS + SIZES].median().to_numpy()


def ceiling(df, cols, form, extra=None):
    kn = df.loc[df.commits > 0, cols].median().to_numpy() if form == "spline" else None
    S = output_terms(df, cols, form, kn)
    Z = np.column_stack([S, week_dummies(df.week)] + ([extra] if extra is not None else []))
    g = codes(df.user_id)
    b, c, mu = fe_poisson(df.tok_m.values, Z, g)
    Z0 = Z.copy()
    Z0[:, :S.shape[1]] = 0.0
    if extra is not None:
        pass                                   # tool mix is not output: kept as observed
    f0 = np.exp(Z0 @ b + c[g])
    share = float(f0.sum() / df.tok_m.sum())
    zero = df.commits == 0
    calib = float(df.tok_m[zero].sum() / mu[zero].sum()) if zero.any() else np.nan
    return dict(share=share, calib_obs_over_fit_zero_weeks=calib,
                coef=[float(v) for v in b[:S.shape[1]]]), c, mu, f0


rq1 = {}
for form in ["counts", "log1p", "sqrt", "spline"]:
    for step, cols in [("step1", COUNTS), ("step2", COUNTS + SIZES)]:
        res, c, mu, f0 = ceiling(P, cols, form)
        res["ci"] = boot_persons(P, lambda d, cols=cols, form=form:
                                 ceiling(d, cols, form)[0]["share"], reps=B)
        rq1[f"{step}_{form}"] = res
# tool-mix-adjusted step 2 (Agent A and Agent B shares of the week's tokens)
tm = np.column_stack([np.where(P.tokens > 0, P.tok_agent_a / P.tokens.replace(0, 1), 0),
                      np.where(P.tokens > 0, P.tok_agent_b / P.tokens.replace(0, 1), 0),
                      np.where(P.tokens > 0, P.tok_frontier / P.tokens.replace(0, 1), 0)])
for form in ["counts", "log1p"]:
    rq1[f"step2_{form}_toolmix"] = ceiling(P, COUNTS + SIZES, form, extra=tm)[0]
# truncation sensitivity: weeks 0..3 only; and weeks 0..5
for lab, wk in [("w0_3", [0, 1, 2, 3]), ("w0_5", [0, 1, 2, 3, 4, 5])]:
    Q = PW[PW.user_id.isin(P.user_id.unique()) & PW.week.isin(wk)].copy()
    Q = Q[Q.groupby("user_id").tokens.transform("sum") > 0]
    if RULE == "any":
        Q = Q[any_recorded(Q)]
    for form in ["counts", "log1p"]:
        rq1[f"step2_{form}_{lab}"] = ceiling(Q, COUNTS + SIZES, form)[0]
# commits from the internal host only (no mirror ambiguity)
Q = P.assign(commits=P.commits_internal)
rq1["step2_log1p_internal_only"] = ceiling(Q, COUNTS + SIZES, "log1p")[0]

# step 3: decompose the person term (log1p step 2) on person characteristics
_, c_hat, _, _ = ceiling(P, COUNTS + SIZES, "log1p")
pid = pd.Series(c_hat, index=pd.factorize(P.user_id)[1])
pid = pid[np.isfinite(pid)]           # the person effect exists only where estimable
D = X.loc[pid.index].assign(c=pid.values)
D["log_pre_commits"] = np.log1p(D.pre_commits.fillna(0))
D["log_pre_lpc"] = np.log1p(D.pre_lines_per_commit.fillna(D.pre_lines_per_commit.median()))
D["pre_bug_share"] = D.pre_bug_share.fillna(0)
blocks = {
    "role": pd.get_dummies(D.role, drop_first=True).astype(float).to_numpy(),
    "manager": D[["manager"]].to_numpy(float),
    "team2": pd.get_dummies(D.team2, drop_first=True).astype(float).to_numpy(),
    "pre_mix": D[["log_pre_commits", "log_pre_lpc", "pre_bug_share"]].to_numpy(float),
}


def cv_r2(y, Z, folds=10):
    n = len(y)
    perm = rng.permutation(n)
    sse = sst = 0.0
    for k in range(folds):
        te = perm[k::folds]
        tr = np.setdiff1d(perm, te)
        b = np.linalg.lstsq(np.c_[np.ones(len(tr)), Z[tr]], y[tr], rcond=None)[0]
        p = np.c_[np.ones(len(te)), Z[te]] @ b
        sse += ((y[te] - p) ** 2).sum()
        sst += ((y[te] - y[tr].mean()) ** 2).sum()
    return float(1 - sse / sst)


step3 = {}
for name, Z in list(blocks.items()) + [("all", np.column_stack(list(blocks.values())))]:
    _, r2, _ = ols(D.c.values, np.c_[np.ones(len(D)), Z])
    step3[name] = dict(r2=float(r2), cv_r2=cv_r2(D.c.values, Z))
    if name == "role":
        # seniority (from title keywords) was dropped as too noisy; burn the
        # draw its block used so later cross-validation folds are unchanged
        rng.permutation(len(D))
step3["sd_c"] = float(D.c.std())
step3["decile_ratio_exp_c"] = float(np.exp(D.c.quantile(0.9) - D.c.quantile(0.1)))
rq1["step3_person_term_decomposition"] = step3

# distribution of the step-2 output-independent component across people
res, c, mu, f0 = ceiling(P, COUNTS + SIZES, "log1p")
per = pd.DataFrame({"u": P.user_id, "f0": f0, "y": P.tok_m}).groupby("u").sum()
per["share"] = per.f0 / per.y
rq1["step2_log1p_person_share_quantiles"] = per.share.quantile(
    [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
rq1["step2_log1p_gini_of_component"] = gini(per.f0)

# ---------------------------------------------------------------- floor
nz_med = P.loc[P.tokens > 0, "tokens"].median()


def floor(df, width):
    """Share of tokens in windows of `width` weeks with substantial spend and
    no commit. Windows are consecutive blocks from week 0;
    an incomplete trailing block is dropped."""
    d = df.copy()
    d["blk"] = (d.week - d.week.min()) // width
    nb = d.groupby("blk").week.nunique()
    d = d[d.blk.isin(nb[nb == width].index)]
    W = d.groupby(["user_id", "blk"]).agg(tok=("tokens", "sum"),
                                          out=("output_any", "max")).reset_index()
    thr = W.loc[W.tok > 0, "tok"].median()
    hi = (W.tok > thr) & (W.out == 0)
    # within-person cut: upper half of the person's own nonzero windows
    pm = W[W.tok > 0].groupby("user_id").tok.median()
    hi_w = (W.tok > W.user_id.map(pm)) & (W.out == 0) & (W.tok > 0)
    return dict(share=float(W.tok[hi].sum() / W.tok.sum()),
                share_within_cut=float(W.tok[hi_w].sum() / W.tok.sum()),
                share_any_zero_output=float(W.tok[W.out == 0].sum() / W.tok.sum()),
                n_windows=int(len(W)), n_floor_windows=int(hi.sum()))


rq1["floor"] = {f"{w}wk": floor(P, w) for w in [1, 2, 4]}
rq1["floor"]["1wk_w0_3"] = floor(P[P.week <= 3], 1)
rq1["floor"]["1wk_commits_only"] = floor(P.assign(output_any=(P.commits > 0).astype(int)), 1)
rq1["floor"]["1wk_ci"] = boot_persons(P, lambda d: floor(d, 1)["share"])
# PR open-to-close durations (authored PRs created in the window, closed in it)
rv = pd.read_csv("data/reviews.csv")
rv["d"] = pd.to_datetime(rv.created)
au = rv[rv.roles.str.contains("author") & (rv.d >= "2026-05-10")]
open_through = (pd.Timestamp("2026-08-08") - au.d).dt.days + 1
closed = au[au.toiw < open_through - 1.01]
rq1["pr_open_days_quantiles"] = closed.toiw.quantile([0.25, 0.5, 0.75, 0.9]).round(2).to_dict()
rq1["pr_share_closed_within_7d"] = float((closed.toiw <= 7).mean())
rq1["pr_n_closed"] = int(len(closed))

# ---------------------------------------------------------------- lag test
# zero-output weeks t in 0..3; outcome = commits at t+1 (t+2) over the
# person's mean weekly commits across weeks -7..4 (pre-period supplies the
# baseline). Control: zero-output weeks with low but nonzero spend (the
# person was at work and using the tools), per the plan's absence concern.
base = PW[PW.week.between(-7, 4)].groupby("user_id").commits.mean()
Pi = P.set_index(["user_id", "week"])


def lag_test(df, lag):
    d = df.sort_values(["user_id", "week"]).copy()
    d["next"] = d.groupby("user_id").commits.shift(-lag)
    d = d[(d.output_any == 0) & d.next.notna()]
    d["rel"] = d.next / d.user_id.str.split("#").str[0].map(base).replace(0, np.nan)
    d = d[d.rel.notna()]
    hi = d[d.tokens > nz_med]
    lo = d[(d.tokens > 0) & (d.tokens <= nz_med)]
    off = d[d.tokens == 0]
    return dict(n_high=int(len(hi)), n_low=int(len(lo)), n_nouse=int(len(off)),
                rebound_high=float(hi.rel.mean()), rebound_low=float(lo.rel.mean()),
                rebound_nouse=float(off.rel.mean()),
                diff_high_minus_low=float(hi.rel.mean() - lo.rel.mean()),
                mw_p=float(stats.mannwhitneyu(hi.rel, lo.rel).pvalue)
                if len(hi) > 4 and len(lo) > 4 else np.nan)


rq1["lag_test"] = {f"lag{l}": lag_test(P, l) for l in [1, 2]}
rq1["lag_test"]["lag1_diff_ci"] = boot_persons(P, lambda d: lag_test(d, 1)["diff_high_minus_low"])
# reference: rebound after ordinary (positive-output) weeks
d = P[(P.output_any == 1) & (P.week <= 3)]
nxt = np.array([Pi.commits.get((u, w + 1), np.nan) for u, w in zip(d.user_id, d.week)])
rq1["lag_test"]["rebound_after_output_weeks"] = float(np.nanmean(
    nxt / d.user_id.map(base).replace(0, np.nan).to_numpy()))

# ---------------------------------------------------------------- concentration
Xa = X.loc[P.user_id.unique()]   # adopters with something recorded in the window
rq1["concentration"] = dict(
    gini_tokens_adopters=gini(Xa.tokens), gini_tokens_all=gini(X.tokens),
    gini_commits_dedup_all=gini(X.commits), gini_commits_dedup_adopters=gini(Xa.commits),
    gini_lines_all=gini(X.lines), gini_prs_all=gini(X.prs_authored),
    gini_issues_resolved_all=gini(X.issues_resolved),
    gini_person_week_tokens_active=gini(P.loc[P.tokens > 0, "tokens"]),
    gini_person_week_commits=gini(P.commits))
R["rq1"] = rq1

# ================================================================ RQ3
rq3 = {}


def d_fe(df, out, agg=None):
    d = df[(df[out] > 0) & (df.tokens > 0)]
    d = d[d.groupby("user_id").week.transform("nunique") >= 2]
    Z = np.column_stack([np.log(d[out].to_numpy(float)), week_dummies(d.week)])
    b, _, _ = fe_poisson(d.tok_m.values, Z, codes(d.user_id))
    return float(b[0]), d


def d_fe_ols(df, out):
    d = df[(df[out] > 0) & (df.tokens > 0)]
    d = d[d.groupby("user_id").week.transform("nunique") >= 2].copy()
    for v, s in [("ly", np.log(d.tokens)), ("lx", np.log(d[out]))]:
        d[v] = s
        d[v] = d[v] - d.groupby("user_id")[v].transform("mean")
        d[v] = d[v] - d.groupby("week")[v].transform("mean")
    return float((d.ly * d.lx).sum() / (d.lx ** 2).sum()), float(np.corrcoef(d.ly, d.lx)[0, 1])


for out in ["commits", "lines", "prs_authored", "commit_days"]:
    est, d = d_fe(P, out)
    o, r = d_fe_ols(P, out)
    dropped = P[(P.tokens > 0) & ~P.index.isin(d.index)]
    rq3[f"within_{out}"] = dict(
        d_ppml=est, d_ols=o, within_corr=r, n_weeks=int(len(d)),
        n_people=int(d.user_id.nunique()),
        ci=boot_persons(P, lambda q, out=out: d_fe(q, out)[0]),
        dropped_active_weeks=int(len(dropped)),
        dropped_median_tokens_m=float(dropped.tok_m.median()) if len(dropped) else np.nan,
        kept_median_tokens_m=float(d.tok_m.median()))

# two-week aggregation: blocks (0,1), (2,3); week 4 dropped
A2 = P[P.week <= 3].assign(week=lambda d: d.week // 2).groupby(
    ["user_id", "week"], as_index=False)[["tokens", "tok_m", "commits", "lines",
                                           "prs_authored"]].sum()
for out in ["commits", "lines"]:
    est, d = d_fe(A2, out)
    rq3[f"within_{out}_2wk"] = dict(d_ppml=est, n_people=int(d.user_id.nunique()),
                                    ci=boot_persons(A2, lambda q, out=out: d_fe(q, out)[0]))
# one-week on the same weeks 0..3 for a like-for-like comparison
rq3["within_commits_w0_3"] = dict(d_ppml=d_fe(P[P.week <= 3], "commits")[0])
# between-person on the same panel window (person totals over weeks 0..4)
T = P.groupby("user_id")[["tokens", "commits", "lines", "prs_authored"]].sum()
for out in ["commits", "lines"]:
    t = T[(T[out] > 0) & (T.tokens > 0)]
    b, r2, _ = ols(np.log(t.tokens), np.c_[np.ones(len(t)), np.log(t[out])])
    rq3[f"between_{out}_same_window"] = dict(d_ols=float(b[1]), r2=float(r2), n=int(len(t)))
# between-person using pre-period output (spend vs output measured before it)
t = X[(X.has_usage == 1) & (X.pre_commits > 0)]
t = t.join(T.commits.rename("tw_commits"))
t = t[(t.tw_commits > 0) & (t.tokens > 0)]
b_pre, r2_pre, _ = ols(np.log(t.tokens), np.c_[np.ones(len(t)), np.log(t.pre_commits)])
b_tw, r2_tw, _ = ols(np.log(t.tokens), np.c_[np.ones(len(t)), np.log(t.tw_commits)])
b_both, r2_both, _ = ols(np.log(t.tokens), np.c_[np.ones(len(t)), np.log(t.pre_commits),
                                                np.log(t.tw_commits)])
rq3["between_pre_vs_telemetry_output"] = dict(
    n=int(len(t)), d_on_pre_commits=float(b_pre[1]), r2_pre=float(r2_pre),
    d_on_telemetry_commits=float(b_tw[1]), r2_tw=float(r2_tw),
    joint_pre=float(b_both[1]), joint_tw=float(b_both[2]), r2_joint=float(r2_both),
    spearman_tokens_pre=float(stats.spearmanr(t.tokens, t.pre_commits).correlation),
    spearman_tokens_tw=float(stats.spearmanr(t.tokens, t.tw_commits).correlation),
    spearman_pre_tw=float(stats.spearmanr(t.pre_commits, t.tw_commits).correlation))
# output change vs spend: log(telemetry-week commit rate / pre rate)
t["chg"] = np.log(t.tw_commits / 5) - np.log(t.pre_commits)
rq3["output_change_vs_tokens"] = dict(
    spearman=float(stats.spearmanr(t.tokens, t.chg).correlation),
    p=float(stats.spearmanr(t.tokens, t.chg).pvalue), n=int(len(t)))
na = X[(X.has_usage == 0) & (X.pre_commits > 0)].join(
    PW[PW.week.isin(WEEKS)].groupby("user_id").commits.sum().rename("tw_commits"))
na = na[na.tw_commits > 0]
rq3["output_change_adopters_vs_non"] = dict(
    median_adopters=float(t.chg.median()), median_non=float(
        (np.log(na.tw_commits / 5) - np.log(na.pre_commits)).median()),
    n_non=int(len(na)),
    mw_p=float(stats.mannwhitneyu(t.chg, np.log(na.tw_commits / 5) - np.log(na.pre_commits)).pvalue))

# tool-homogeneous person-weeks (>90% Agent A / Agent B)
for tool, col in [("agent_a", "tok_agent_a"), ("agent_b", "tok_agent_b")]:
    Q = P[(P.tokens > 0) & (P[col] / P.tokens.replace(0, 1) > 0.9)]
    est, d = d_fe(Q, "commits")
    rq3[f"within_commits_{tool}_weeks"] = dict(d_ppml=est, n_people=int(d.user_id.nunique()),
                                                n_weeks=int(len(d)))
# credits and generation tokens as outcomes
for m in ["credits", "gen_tokens"]:
    Q = P.assign(tok_m=P[m] / P[m].mean())
    rq3[f"within_commits_{m}"] = dict(d_ppml=d_fe(Q, "commits")[0])

# cost-per-commit vs volume, within person
d = P[(P.commits > 0) & (P.tokens > 0)].copy()
d["lr"] = np.log(d.tokens / d.commits)
d["lc"] = np.log(d.commits)
for v in ["lr", "lc"]:
    d[v] = d[v] - d.groupby("user_id")[v].transform("mean")
rq3["cost_per_commit_vs_volume_within_spearman"] = float(stats.spearmanr(d.lr, d.lc).correlation)
rq3["tokens_per_commit_p90_p10_person_weeks"] = float(np.exp(
    np.log(P.loc[(P.commits > 0) & (P.tokens > 0)].eval("tokens/commits")).quantile([0.1, 0.9]).diff().iloc[-1]))

# ---------------------------------------------------------------- stability
# d_i per person is not estimable (median 4 positive-output weeks, the plan's
# gate is ~20). What remains: reliability of each person's conversion LEVEL
# a_i (person effect at a common d) and of tokens-per-commit, across
# interleaved weeks; and of spend itself on the same weeks.
d_all, _ = d_fe(P, "commits")
stab = dict(d_common=d_all, gate_median_pos_weeks=R["panel"]["median_pos_commit_weeks"],
            d_i_estimable=False)


def level(df):
    d = df[(df.commits > 0) & (df.tokens > 0)]
    tt = d.groupby("week").apply(lambda q: np.log(q.tokens).mean() - d_all * np.log(q.commits).mean())
    a = (np.log(d.tokens) - d_all * np.log(d.commits) - d.week.map(tt)).groupby(d.user_id).mean()
    return a


odd = P[P.week.isin([1, 3])]
even = P[P.week.isin([0, 2, 4])]
a_o, a_e = level(odd), level(even)
j = a_o.index.intersection(a_e.index)
r = stats.spearmanr(a_o[j], a_e[j]).correlation
stab["a_i_oddeven_spearman"] = float(r)
stab["a_i_oddeven_sb"] = float(sb(r))
stab["a_i_n"] = int(len(j))
tpc = lambda q: (q.groupby("user_id").tokens.sum() / q.groupby("user_id").commits.sum()).replace(np.inf, np.nan)
to, te = tpc(odd).dropna(), tpc(even).dropna()
j2 = to.index.intersection(te.index)
j2 = j2[(to[j2] > 0) & (te[j2] > 0)]
r = stats.spearmanr(to[j2], te[j2]).correlation
stab["tokens_per_commit_oddeven_spearman"] = float(r)
stab["tokens_per_commit_oddeven_sb"] = float(sb(r))
so, se_ = odd.groupby("user_id").tokens.sum(), even.groupby("user_id").tokens.sum()
r = stats.spearmanr(so[j], se_[j]).correlation
stab["tokens_oddeven_spearman_same_people"] = float(r)
stab["tokens_oddeven_sb_same_people"] = float(sb(r))
co, ce = odd.groupby("user_id").commits.sum(), even.groupby("user_id").commits.sum()
r = stats.spearmanr(co[j], ce[j]).correlation
stab["commits_oddeven_spearman_same_people"] = float(r)
stab["commits_oddeven_sb_same_people"] = float(sb(r))
# rank correlation between a_i (RQ3) and c_i (RQ1)
ci = pd.Series(c_hat, index=pd.factorize(P.user_id)[1])
a_full = level(P)
jj = a_full.index.intersection(ci.index)
stab["spearman_a_i_c_i"] = float(stats.spearmanr(a_full[jj], ci[jj]).correlation)
# top-decile overlap of tokens-per-commit across halves (a dashboard's view)
qo, qe = to[j2] >= to[j2].quantile(0.9), te[j2] >= te[j2].quantile(0.9)
stab["tokens_per_commit_top_decile_overlap"] = float((qo & qe).sum() / qo.sum())
rq3["stability"] = stab
R["rq3"] = rq3

# ================================================================ RQ2
rq2 = {}
res, c, mu, f0 = ceiling(P, COUNTS + SIZES, "counts")      # baseline as written
P["mu"] = mu
P["resid"] = P.tok_m - P.mu
P["resid"] = P.resid - P.resid.mean()                      # explicit centering
P["lres"] = np.where((P.tok_m > 0) & (P.mu > 0), np.log(P.tok_m / P.mu.clip(lower=1e-9)), np.nan)
_, _, mu_l, _ = ceiling(P, COUNTS + SIZES, "log1p")
P["resid_log1p"] = P.tok_m - mu_l
P["resid_log1p"] -= P.resid_log1p.mean()

signals = {
    "bug_commits": "commits_bug",
    "bug_commit_share": None,
    "pr_open_days": "pr_toiw_mean",
    "pr_reviewers": "pr_reviewers_mean",
    "pr_self_comments": "pr_comments_self",
}
P["bug_commit_share"] = np.where(P.commits > 0, P.commits_bug / P.commits.replace(0, 1), np.nan)
signals["bug_commit_share"] = "bug_commit_share"
Pi = P.set_index(["user_id", "week"])


def resid_vs_next(rcol, lag=1):
    out = {}
    d = P[P.week <= 4 - lag].copy()
    for name, col in signals.items():
        nxt = np.array([Pi[col].get((u, w + lag), np.nan) for u, w in zip(d.user_id, d.week)])
        q = d.assign(nxt=nxt).dropna(subset=["nxt", rcol])
        # within-person: demean both
        for v in [rcol, "nxt"]:
            q[v + "_w"] = q[v] - q.groupby("user_id")[v].transform("mean")
        q = q[q.groupby("user_id").week.transform("size") >= 2]
        if len(q) < 30:
            out[name] = dict(n=int(len(q)))
            continue
        r = stats.spearmanr(q[rcol + "_w"], q["nxt_w"])
        out[name] = dict(spearman=float(r.correlation), p=float(r.pvalue), n=int(len(q)),
                         n_people=int(q.user_id.nunique()))
    return out


rq2["resid_counts_vs_next_week"] = resid_vs_next("resid")
rq2["resid_log1p_vs_next_week"] = resid_vs_next("resid_log1p")
rq2["lres_vs_next_week"] = resid_vs_next("lres")

# mode mix: high-residual weeks vs the person's other active weeks
A = P[P.tokens > 0].copy()
A["gen_ratio"] = A.gen_tokens / A.tokens
A["tok_per_event"] = A.tokens / A.events
A["share_agent_a"] = A.tok_agent_a / A.tokens
A["share_agent_b"] = A.tok_agent_b / A.tokens
A["share_frontier"] = A.tok_frontier / A.tokens
A["share_remote"] = A.tok_remote / A.tokens
A["cred_per_tok"] = A.credits / A.tokens
A["lres_w"] = A.lres - A.groupby("user_id").lres.transform("mean")
A = A[A.groupby("user_id").week.transform("size") >= 2]
mode = {}
for col in ["gen_ratio", "tok_per_event", "events", "ai_days", "n_models", "share_agent_a",
            "share_agent_b", "share_frontier", "share_remote", "cred_per_tok"]:
    v = A[col] - A.groupby("user_id")[col].transform("mean")
    r = stats.spearmanr(A.lres_w, v)
    mode[col] = dict(spearman_within=float(r.correlation), p=float(r.pvalue))
rq2["mode_mix_within_person"] = mode
rq2["mode_mix_n_weeks"] = int(len(A))
# person-level residual (mean log ratio) for the sampling frame
pr = A.groupby("user_id").lres.mean()
U = X.loc[pr.index].assign(resid=pr)
U["dominant_tool"] = np.select([U.share_agent_a > 0.9, U.share_agent_b > 0.9],
                               ["agent_a", "agent_b"], "mixed")
U["resid_tercile"] = pd.qcut(U.resid, 3, labels=["low", "mid", "high"])
rq2["sampling_frame"] = pd.crosstab(U.resid_tercile, U.dominant_tool).to_dict()
rq2["resid_by_role"] = {k: [int(len(g)), float(g.median())] for k, g in U.groupby("role").resid}
rq2["resid_by_manager"] = {str(k): [int(len(g)), float(g.median())]
                           for k, g in U.groupby("manager").resid}
# within-person: how much of the week-level log residual is person-level?
rq2["lres_var_between_share"] = float(pr.var() / A.lres.var())
rq2["lres_week_sd_within"] = float(A.lres_w.std())
R["rq2"] = rq2
A[["user_id", "week", "tokens", "commits", "lres", "lres_w"]].to_csv(
    OUT + "residuals_panel.csv", index=False)
U[["resid", "resid_tercile", "dominant_tool"]].to_csv(OUT + "residuals_person_panel.csv")

# ================================================================ RQ4
rq4 = {}
# The profile the extract supports for every engineer: commits, lines changed,
# bug-linked commits, days with a commit. GitHub PR and review fields (PRs
# authored and reviewed, review comments, PR open time) are recorded for the
# minority of engineers who work on GitHub and are zero by construction for
# the rest, so they are not in the profile (dropped 2026-10-23).
OUTPUT = ["commits", "lines_k"]
NONOUT = ["commits_bug", "commit_days"]
PROFILE = OUTPUT + NONOUT


def zs(M):
    M = np.asarray(M, float)
    sd = M.std(0)
    return (M - M.mean(0)) / np.where(sd > 0, sd, 1)


def feat(df, cols):
    return zs(np.log1p(df[cols].to_numpy(float)))


def within_share(df, base_cols, add_cols, folds=5):
    """Incremental deviance share of add_cols over person FE + week FE (+
    base_cols); CV holds out a random week per person in each fold."""
    y = df.tok_m.to_numpy()
    g = codes(df.user_id)
    Wd = week_dummies(df.week)
    Zb = np.column_stack([Wd] + ([feat(df, base_cols)] if base_cols else []))
    Zf = np.column_stack([Zb, feat(df, add_cols)])
    _, _, mb = fe_poisson(y, Zb, g)
    _, _, mf = fe_poisson(y, Zf, g)
    ins = 1 - poisson_deviance(y, mf) / poisson_deviance(y, mb)
    fold = df.groupby("user_id").cumcount().to_numpy()
    shift = rng.integers(0, folds, g.max() + 1)
    fold = (fold + shift[g]) % folds
    dvb = dvf = 0.0
    for k in range(folds):
        tr, te = fold != k, fold == k
        for Z, acc in [(Zb, "b"), (Zf, "f")]:
            b, c, _ = fe_poisson(y[tr], Z[tr], g[tr])
            ok = np.isfinite(c[g[te]])
            m = np.exp(Z[te][ok] @ b + c[g[te]][ok])
            dv = poisson_deviance(y[te][ok], m)
            if acc == "b":
                dvb += dv
            else:
                dvf += dv
    return dict(in_sample=float(ins), cv=float(1 - dvf / dvb))


def between_share(df, base, add, folds=10):
    """Incremental deviance share of `add` over week FE (+ base); no person
    term; CV holds out whole persons."""
    y = df.tok_m.to_numpy()
    Wd = week_dummies(df.week)
    one = np.ones((len(df), 1))
    Zb = np.column_stack([one, Wd] + ([base] if base is not None else []))
    Zf = np.column_stack([Zb, add])
    bb, bf = irls_poisson(Zb, y), irls_poisson(Zf, y)
    ins = 1 - poisson_deviance(y, np.exp(np.clip(Zf @ bf, -30, 30))) / \
        poisson_deviance(y, np.exp(np.clip(Zb @ bb, -30, 30)))
    ids = df.user_id.unique()
    perm = rng.permutation(len(ids))
    pf = pd.Series(perm % folds, index=ids)
    fo = df.user_id.map(pf).to_numpy()
    dvb = dvf = 0.0
    for k in range(folds):
        tr, te = fo != k, fo == k
        bb, bf = irls_poisson(Zb[tr], y[tr]), irls_poisson(Zf[tr], y[tr])
        dvb += poisson_deviance(y[te], np.exp(np.clip(Zb[te] @ bb, -30, 30)))
        dvf += poisson_deviance(y[te], np.exp(np.clip(Zf[te] @ bf, -30, 30)))
    return dict(in_sample=float(ins), cv=float(1 - dvf / dvb))


rq4["within_profile"] = within_share(P, None, PROFILE)
rq4["within_output"] = within_share(P, None, OUTPUT)
rq4["within_profile_beyond_output"] = within_share(P, OUTPUT, NONOUT)
# person + week FE alone vs week FE alone: how much is person heterogeneity
y = P.tok_m.to_numpy()
_, _, m_pw = fe_poisson(y, week_dummies(P.week), codes(P.user_id))
m_w = P.groupby("week").tok_m.transform("mean").to_numpy()
rq4["person_fe_share_over_week_fe"] = float(1 - poisson_deviance(y, m_pw) / poisson_deviance(y, m_w))

pc = X.loc[P.user_id]
role = np.column_stack([pd.get_dummies(pc.role, drop_first=True).astype(float).to_numpy(),
                        pc[["manager"]].to_numpy(float),
                        pd.get_dummies(pc.team2, drop_first=True).astype(float).to_numpy()])
issue_cols = ["issues", "issues_resolved", "issues_bug", "issue_toiw_mean"]
issues = zs(np.log1p(pc[issue_cols].fillna(0).to_numpy(float)))
premix = zs(np.column_stack([np.log1p(pc.pre_commits.fillna(0)),
                             np.log1p(pc.pre_lines_per_commit.fillna(pc.pre_lines_per_commit.median())),
                             pc.pre_bug_share.fillna(0)]))
prof = feat(P, PROFILE)
outp = feat(P, OUTPUT)
rq4["between_profile"] = between_share(P, None, prof)
rq4["between_profile_plus_issues"] = between_share(P, None, np.c_[prof, issues])
rq4["between_output"] = between_share(P, None, outp)
rq4["between_profile_beyond_output"] = between_share(P, outp, feat(P, NONOUT))
rq4["between_role_team"] = between_share(P, None, role)
rq4["between_role_beyond_profile"] = between_share(P, np.c_[prof, issues], role)
rq4["between_premix_beyond_profile"] = between_share(P, np.c_[prof, issues], premix)
rq4["between_profile_beyond_premix"] = between_share(P, premix, np.c_[prof, issues])
# dispersion in units of spread: person-week log tokens, active weeks
A = P[P.tokens > 0].copy()
lt = np.log(A.tokens.to_numpy())
Zp = np.c_[np.ones(len(A)), week_dummies(A.week), feat(A, PROFILE),
           zs(np.log1p(X.loc[A.user_id, issue_cols].fillna(0).to_numpy(float)))]
_, r2p, e = ols(lt, Zp)
rq4["log_r2_profile_person_weeks"] = float(r2p)
rq4["decile_ratio_tokens_person_weeks"] = float(np.exp(np.percentile(lt, 90) - np.percentile(lt, 10)))
rq4["decile_ratio_resid_profile"] = float(np.exp(np.percentile(e, 90) - np.percentile(e, 10)))
# within commit quintile of the person-week
q = pd.qcut(A.commits.rank(method="first"), 5, labels=False)
rq4["within_commit_quintile_decile_ratio"] = {
    int(k): float(np.exp(np.log(g).quantile(0.9) - np.log(g).quantile(0.1)))
    for k, g in A.groupby(q).tokens}
R["rq4"] = rq4

with open(OUT + "results_panel.json", "w") as f:
    json.dump(R, f, indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
