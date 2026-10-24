from pathlib import Path
"""Figures for the results walkthrough. Run from the repo root."""
import sys, pathlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from paperstyle import (BLUE, ORANGE, AQUA, YEL, GREY, INK, MUTED, LIGHTBLUE, PALEGREY,
                        FULL, COL, TEXT, TICK, polish, figure, save)

OUT = "figs/"

def fe_poisson(y, X, g, max_iter=100, tol=1e-10):
    y = np.asarray(y, float); X = np.asarray(X, float); g = np.asarray(g)
    G = g.max() + 1; k = X.shape[1]
    n_g = np.bincount(g, weights=y, minlength=G); b = np.zeros(k)
    for _ in range(max_iter):
        eta = X @ b; m = np.full(G, -np.inf); np.maximum.at(m, g, eta)
        w = np.exp(eta - m[g]); s = np.bincount(g, weights=w, minlength=G); p = w / s[g]
        grad = X.T @ (y - n_g[g] * p)
        xbar = np.column_stack([np.bincount(g, weights=p * X[:, j], minlength=G) for j in range(k)])
        H = (X * (n_g[g] * p)[:, None]).T @ X - (xbar * n_g[:, None]).T @ xbar
        step = np.linalg.solve(H + 1e-10 * np.eye(k), grad); b = b + step
        if np.max(np.abs(step)) < tol: break
    eta = X @ b; s = np.bincount(g, weights=np.exp(eta), minlength=G)
    with np.errstate(divide="ignore"): c = np.log(n_g) - np.log(s)
    mu = np.where(np.isfinite(c[g]), np.exp(eta + np.where(np.isfinite(c), c, 0)[g]), 0.0)
    return b, c, mu

def codes(s): return pd.factorize(s)[0]
def rel_axis(ax, which, lo, hi, label, pooled_low=False, step=None, powers=False):
    """Log10 axis of a quantity divided by its median: ticks 10^lo..10^hi labelled 0.01, 0.1, 1, 10 ...
    (token volumes are shown relative to a median so that no absolute counts are revealed)."""
    step = step or (1 if hi - lo <= 5 else 2)        # every decade, or every other one on long axes (0 always labelled)
    fmt = (lambda k: "1" if k == 0 else ("10" if k == 1 else f"$10^{{{k}}}$")) if powers else (lambda k: f"{10.0 ** k:g}")
    ks = [k for k in range(lo, hi + 1) if k % step == 0]; lab = [fmt(k) for k in ks]
    if pooled_low:                                   # the first bin pools everything below 10^lo
        if ks[0] != lo: ks, lab = [lo] + ks, [fmt(lo)] + lab
        lab[0] = "\u2264" + lab[0]
    (ax.set_xticks if which == "x" else ax.set_yticks)(ks)
    (ax.set_xticklabels if which == "x" else ax.set_yticklabels)(lab)
    (ax.set_xlabel if which == "x" else ax.set_ylabel)(label)
def week_dummies(w):
    w = np.asarray(w); lv = sorted(set(w))
    return np.column_stack([(w == v).astype(float) for v in lv[1:]])

X = pd.read_csv("analysis/out/people.csv", index_col=0)
PW = pd.read_csv("analysis/out/person_week.csv")
PW["tok_m"] = PW.tokens / 1e6
PW["output_any"] = ((PW.commits > 0) | (PW.prs_authored > 0)).astype(int)
WEEKS = [0, 1, 2, 3, 4]
adopters = X.index[X.has_usage == 1]
P = PW[PW.user_id.isin(adopters) & PW.week.isin(WEEKS)].copy()
# one rule everywhere: a person-week is in iff it has anything recorded
P = P[(P.tokens > 0) | (P.commits > 0) | (P.prs_authored > 0)].reset_index(drop=True)
pre = PW[PW.week < 0].groupby("user_id").commits.mean().rename("pre_commits")

# ===================================================== Fig 0: spend distribution (relative to the median)
fig0, axes0 = figure(COL, 1.75, ncols=2)
for ax, vals, title in [(axes0[0], P.loc[P.tokens > 0, "tokens"], "Person-weeks"),
                        (axes0[1], P.groupby("user_id").tokens.sum().pipe(lambda s: s[s > 0]), "People, five-week total")]:
    polish(ax)
    lv = np.log10(vals / vals.median()); lv = lv.clip(lower=-3)        # values below 0.001 x median pooled
    ax.hist(lv, bins=np.arange(-3, lv.max() + 0.25, 0.25), color=BLUE, edgecolor="white", lw=0.6)
    ax.axvline(0, color=INK, lw=1.0)
    for qv in np.percentile(lv, [10, 90]): ax.axvline(qv, color=INK, lw=0.9, ls=":")
    _hi0 = int(np.ceil(lv.max()))
    rel_axis(ax, "x", -3, _hi0, "tokens (median = 1)", step=2, powers=True)   # every other decade labelled (the pooled first bin is stated in the caption)
    ax.set_xticks(range(-3, _hi0 + 1), minor=True)
    ax.set_title(title, loc="left", fontweight="bold")
    print("fig0", title, "n", len(vals), "p90/p10", round(float(np.percentile(vals, 90) / np.percentile(vals, 10)), 1),
          "p10, p90 rel. to median", round(float(np.percentile(vals, 10) / vals.median()), 3), round(float(np.percentile(vals, 90) / vals.median()), 2))
save(fig0, OUT + "fig0-spend-distribution")

# ===================================================== Fig 1: two slopes
fig, axes = figure(COL, 1.9, ncols=2)
figB, axB = figure(COL, 1.9)
A = P[(P.tokens > 0) & (P.commits > 0)].copy()
A = A[A.groupby("user_id").week.transform("size") >= 2]
A["ly"], A["lx"] = np.log10(A.tokens), np.log10(A.commits)
for v in ["ly", "lx"]:
    A[v] -= A.groupby("week")[v].transform("mean") - A[v].mean()
pm = A.groupby("user_id")[["lx", "ly"]].mean()
pm["ly"] -= pm.ly.median()                      # tokens relative to the median engineer (slope unchanged)
b = np.polyfit(pm.lx, pm.ly, 1)
ax = axes[0]; polish(ax)
ax.scatter(pm.lx, pm.ly, s=4, color=BLUE, alpha=0.45, lw=0)
_vb = pd.qcut(pm.lx.rank(method="first"), 20, labels=False)
_vm = pm.groupby(_vb)[["lx", "ly"]].mean()
ax.scatter(_vm.lx, _vm.ly, s=8, color=INK, marker="s", zorder=5, label="ventile means")
xs = np.linspace(pm.lx.min(), pm.lx.max(), 10)
ax.plot(xs, np.polyval(b, xs), color=ORANGE, lw=1.5, label=f"slope {b[0]:.2f}")
ax.plot(xs, (xs - pm.lx.mean()) + pm.ly.mean(), color=INK, lw=0.8, ls=":", label="slope 1")
ax.set_xlabel("commits/week, mean")
_lo, _hi = int(np.floor(pm.ly.min())), int(np.ceil(pm.ly.max()))
ax.set_ylim(None, _hi + 1.0); rel_axis(ax, "y", _lo, _hi, "tokens/week, mean\n(median engineer = 1)")
ax.set_title("A  Between people"); ax.legend(loc="upper left")
# pre-period
T = P.groupby("user_id")[["tokens", "commits"]].sum()
t2 = X.loc[adopters].join(pre); t2 = t2.join(T.commits.rename("tw"))
t2 = t2[(t2.pre_commits > 0) & (t2.tokens > 0) & (t2.tw > 0)]
r_pre = stats.spearmanr(t2.tokens, t2.pre_commits).correlation
r_tw = stats.spearmanr(t2.tokens, t2.tw).correlation
ax = axB; polish(ax)
lx2, ly2 = np.log10(t2.pre_commits), np.log10(t2.tokens / t2.tokens.median())
b2 = np.polyfit(lx2, ly2, 1)
ax.scatter(lx2, ly2, s=4, color=GREY, alpha=0.5, lw=0)
xs = np.linspace(lx2.min(), lx2.max(), 10)
ax.plot(xs, np.polyval(b2, xs), color=ORANGE, lw=1.5, label=f"slope {b2[0]:.2f}")
ax.set_xlabel("commits/week, weeks −7 to −1 (log10)")
_lo2, _hi2 = int(np.floor(ly2.min())), int(np.ceil(ly2.max()))
h, _ = ax.get_legend_handles_labels()
h += [Line2D([], [], ls="none", label=f"Spearman ρ = {r_pre:.2f}"),
      Line2D([], [], ls="none", label=f"(concurrent output: ρ = {r_tw:.2f})")]
ax.set_ylim(None, _hi2 + 2.0); rel_axis(ax, "y", _lo2, _hi2, "tokens, weeks 0–4\n(median engineer = 1)")
ax.legend(handles=h, loc="upper left")
# within
for v in ["ly", "lx"]:
    A[v] -= A.groupby("user_id")[v].transform("mean")
bw = (A.ly * A.lx).sum() / (A.lx ** 2).sum()
ax = axes[1]; polish(ax)
ax.scatter(A.lx, A.ly, s=3, color=BLUE, alpha=0.25, lw=0)
A["bin"] = pd.qcut(A.lx.rank(method="first"), 20, labels=False)
vm = A.groupby("bin")[["lx", "ly"]].mean()
ax.scatter(vm.lx, vm.ly, s=8, color=INK, marker="s", zorder=5, label="ventile means")
xs = np.linspace(-1.1, 1.1, 10)
ax.plot(xs, bw * xs, color=ORANGE, lw=1.5, label=f"slope {bw:.2f}")
ax.plot(xs, xs, color=INK, lw=0.8, ls=":", label="slope 1")
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-2.2, 3.1); ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_xlabel("commits vs own mean")
ax.set_ylabel("tokens vs own mean")
ax.set_title("B  Within person"); ax.legend(loc="upper left")
save(fig, OUT + "fig1-two-slopes")
save(figB, OUT + "fig1b-pre-metering")
print("fig1 slopes: between", b[0], "pre", b2[0], "within", bw, "rho pre", r_pre, "rho tw", r_tw)

# ===================================================== Fig 2: cost per commit
fig, axes = figure(FULL, 2.2, ncols=2)
pp = X.loc[adopters]; pp = pp[(pp.commits > 0) & (pp.tokens > 0)]
ax = axes[0]; polish(ax)
lx, ly = np.log10(pp.commits), np.log10(pp.tokens / pp.commits)
ax.scatter(lx, ly, s=4, color=BLUE, alpha=0.4, lw=0)
pp = pp.assign(lx=lx, ly=ly, dec=pd.qcut(lx.rank(method="first"), 10, labels=False))
dm = pp.groupby("dec")[["lx", "ly"]].median()
ax.plot(dm.lx, dm.ly, "s-", color=INK, ms=2.5, lw=1, label="decile medians")
rho_b = stats.spearmanr(pp.commits, pp.tokens / pp.commits).correlation
ax.set_xlabel("commits in 90-day window (log10)"); ax.set_ylabel("tokens per commit (log10)")
ax.set_title("A  Between people"); ax.legend(loc="upper right")
ax.text(0.03, 0.04, f"Spearman ρ = {rho_b:.2f}", transform=ax.transAxes, fontsize=TEXT, color=MUTED)
d = P[(P.commits > 0) & (P.tokens > 0)].copy()
d["lr"] = np.log10(d.tokens / d.commits); d["lc"] = np.log10(d.commits)
for v in ["lr", "lc"]: d[v] -= d.groupby("user_id")[v].transform("mean")
rho_w = stats.spearmanr(d.lr, d.lc).correlation
ax = axes[1]; polish(ax)
ax.scatter(d.lc, d.lr, s=3, color=BLUE, alpha=0.25, lw=0)
d["bin"] = pd.qcut(d.lc.rank(method="first"), 12, labels=False)
bm = d.groupby("bin")[["lc", "lr"]].median()
ax.plot(bm.lc, bm.lr, "s-", color=INK, ms=2.5, lw=1, label="bin medians")
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-2.2, 2.2)
ax.set_xlabel("commits this week vs own mean (log10)")
ax.set_ylabel("tokens per commit vs own mean (log10)")
ax.set_title("B  Within person"); ax.legend(loc="upper right")
ax.text(0.03, 0.04, f"Spearman ρ = {rho_w:.2f}", transform=ax.transAxes, fontsize=TEXT, color=MUTED)
plt.close(fig)   # fig2 (cost per commit) is no longer in the paper; its correlations are still printed for the checker
print("fig2 rho between", rho_b, "within", rho_w)

# ===================================================== Fig 3: dark spend
COUNTS = ["commits", "prs_authored"]
def terms(S, form, kn=None):
    if form == "log1p": return np.log1p(S)
    if form == "sqrt": return np.sqrt(S)
    if form == "spline":
        L = np.log1p(S); return np.column_stack([L, np.maximum(L - np.log1p(kn), 0)])
    return S
kn = P.loc[P.commits > 0, COUNTS].median().to_numpy()
g = codes(P.user_id); WD = week_dummies(P.week)
fits = {}
for form in ["counts", "sqrt", "log1p", "spline"]:
    S = terms(P[COUNTS].to_numpy(float), form, kn)
    Z = np.column_stack([S, WD]); b, c, mu = fe_poisson(P.tok_m.values, Z, g)
    Z0 = Z.copy(); Z0[:, :S.shape[1]] = 0
    f0 = np.exp(Z0 @ b + c[g]); share = f0.sum() / P.tok_m.sum()
    zero = (P.commits == 0) & (P.prs_authored == 0)
    calib = P.tok_m[zero].sum() / mu[zero].sum()
    fits[form] = dict(b=b[:S.shape[1]], share=share, calib=calib)
    print(form, "share", round(share, 3), "calib", round(calib, 2))
# observed vs predicted tokens, each relative to the person's own mean week, by commit bin.
# In the fixed-effects Poisson fit each person's fitted total equals their observed total, so
# mu_it / mean_t(mu_it) and y_it / mean_t(y_it) are on the same footing.
edges = [-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, 40.5, 80.5, 1e9]
labels = ["0", "1–2", "3–5", "6–10", "11–20", "21–40", "41–80", "80+"]
P["cbin"] = pd.cut(P.commits, edges, labels=labels)
P["m_i"] = P.groupby("user_id").tokens.transform("mean") / 1e6
P["rel_obs"] = P.tok_m / P.m_i
res = P.groupby("cbin").agg(n=("rel_obs", "size"), obs=("rel_obs", "mean"))
for form in ["counts", "sqrt", "log1p", "spline"]:
    S = terms(P[COUNTS].to_numpy(float), form, kn); Z = np.column_stack([S, WD]); b, c, mu = fe_poisson(P.tok_m.values, Z, g)
    mhat = pd.Series(mu).groupby(P.user_id.values).transform("mean").values
    res[form] = pd.Series(mu / mhat).groupby(P.cbin.values).mean()
fig, ax = figure(COL, 1.6); polish(ax)
xi = np.arange(len(labels))
colors = {"counts": ORANGE, "sqrt": YEL, "log1p": AQUA, "spline": BLUE}
names = {"counts": "raw counts", "sqrt": "square root", "log1p": "log(1+x)", "spline": "spline in log(1+x)"}
for form in ["counts", "sqrt", "log1p", "spline"]:
    ax.plot(xi, res[form], "-o", ms=2.5, lw=1.3, color=colors[form],
            label=f"{names[form]} ({fits[form]['share']:.0%})")
ax.scatter(xi, res.obs, s=np.sqrt(res.n) * 1.6, color=INK, zorder=6, lw=0, label="observed")
ax.axhline(1, color=INK, lw=0.5, ls=":")
ax.set_xticks(xi); ax.set_xticklabels(labels)
ax.set_ylim(0.4, 1.5); ax.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
ax.set_xlabel("commits in the week"); ax.set_ylabel("tokens ÷ own mean week")
ax.legend(loc="lower right", markerscale=0.8, ncol=2, columnspacing=0.8, handlelength=1.6)
save(fig, OUT + "fig3-dark-spend")
# floor series by window width, commits only, from person-day (plotted in fig 8B)
PDd = pd.read_csv("analysis/out/person_day.csv", parse_dates=["d"])
dgrid = pd.MultiIndex.from_product([P.user_id.unique(), pd.date_range("2026-06-29", "2026-08-02")], names=["user_id", "d"])
DD = PDd.set_index(["user_id", "d"]).reindex(dgrid, fill_value=0).reset_index()
DD["day"] = (DD.d - DD.d.min()).dt.days
floors = {}
for lab, w in [("1 day", 1), ("3 days", 3), ("1 week", 7), ("2 weeks", 14), ("4 weeks", 28)]:
    W = DD.assign(k=DD.day // w).groupby(["user_id", "k"], as_index=False)[["commits", "tokens"]].sum()
    W = W[W.k < (35 // w)]                                   # complete windows only
    thr = W.loc[W.tokens > 0, "tokens"].median()
    floors[lab] = float(W.tokens[(W.tokens > thr) & (W.commits == 0)].sum() / W.tokens.sum())
print("floor series:", {k: round(v, 3) for k, v in floors.items()})

# ===================================================== Fig 4: redundancy
figA, axA = figure(COL, 1.6)
act = P[P.tokens > 0].copy()
act["q"] = pd.qcut(act.commits.rank(method="first"), 5, labels=False)
# relative to the median active week, so the axis reveals no absolute token volumes (ratios are unchanged)
med_week = act.tokens.median()
ax = axA; polish(ax)
rngj = np.random.default_rng(1); tl = []
for q in range(5):
    v = np.log10(act.loc[act.q == q, "tokens"] / med_week)
    ax.scatter(q + rngj.uniform(-0.28, 0.28, len(v)), v, s=2.5, color=BLUE, alpha=0.25, lw=0)
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    ax.plot([q - 0.36, q + 0.36], [p50, p50], color=INK, lw=1.3)
    ax.plot([q - 0.36, q + 0.36], [p10, p10], color=INK, lw=0.7, ls="--")
    ax.plot([q - 0.36, q + 0.36], [p90, p90], color=INK, lw=0.7, ls="--")
    ax.text(q, 2.26, f"{10**(p90-p10):.0f}×", ha="center", va="center", fontsize=TEXT, color=INK, fontweight="bold")
    lo, hi = act.loc[act.q == q, "commits"].min(), act.loc[act.q == q, "commits"].max()
    tl.append(f"Q{q+1}\n{lo:.0f}–{hi:.0f}" if q < 4 else f"Q{q+1}\n{lo:.0f}+")
ax.set_xticks(range(5)); ax.set_xticklabels(tl)
ax.set_ylim(-3.4, 2.7)
ax.set_yticks([-3, -2, -1, 0, 1, 2]); ax.set_yticklabels(["0.001", "0.01", "0.1", "1", "10", "100"])
ax.set_ylabel("tokens in week\n(median week = 1)")
ax.set_xlabel("commit quintile (commits in week)")
save(figA, OUT + "fig4a-spread")
fig, ax = figure(COL, 1.4); polish(ax, "x")
# values from analyze_panel.py's results (were hard-coded before 2026-10-07)
import json as _json
_R4 = _json.load(open("analysis/out/results_panel.json"))["rq4"]
rows = [("which person", _R4["person_fe_share_over_week_fe"], INK),
        ("commits and lines", _R4["between_output"]["cv"], BLUE),
        ("+ bug-linked, commit-days", _R4["between_profile"]["cv"], BLUE),
        ("same, within person", _R4["within_profile"]["cv"], LIGHTBLUE),
        ("those beyond output", _R4["between_profile_beyond_output"]["cv"], ORANGE),
        ("work mix beyond all four", _R4["between_premix_beyond_profile"]["cv"], ORANGE),
        ("role, manager, team", _R4["between_role_team"]["cv"], ORANGE)]
ys = np.arange(len(rows))[::-1]
for y, (lab, v, col) in zip(ys, rows):
    ax.barh(y, v, color=col, height=0.7)
    ax.text(max(v, 0) + 0.015, y, f"{v:+.2f}" if v < 0 else f"{v:.2f}", va="center", fontsize=TEXT, color=INK)
ax.axvline(0, color=INK, lw=0.6)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(-0.15, 0.95); ax.set_xlabel("share of variation explained")
save(fig, OUT + "fig4b-explained")

# ===================================================== Fig 5: stability
TW = pd.read_csv("analysis/out/tokens_weekly.csv", index_col=0)
fig, axes = figure(FULL, 2.3, ncols=2, gridspec_kw=dict(width_ratios=[1, 1.35]))
ax = axes[0]; ax.grid(False)
NW = 5   # the five complete link weeks; the truncated week 5 is out, as everywhere
M = np.zeros((NW, NW))
for i in range(NW):
    for j in range(NW):
        a, bb = TW[f"tok_w{i}"], TW[f"tok_w{j}"]; ok = (a > 0) & (bb > 0)
        M[i, j] = stats.spearmanr(a[ok], bb[ok]).correlation
im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f3f7fc", "#cde2fb", "#5598e7", "#0d366b"]), vmin=0.3, vmax=1)
for i in range(NW):
    for j in range(NW):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=TICK, color="white" if M[i, j] > 0.72 else INK)
ax.set_xticks(range(NW)); ax.set_yticks(range(NW)); ax.set_xticklabels([f"w{i}" for i in range(NW)]); ax.set_yticklabels([f"w{i}" for i in range(NW)])
ax.set_title("A  Rank agreement between weeks"); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax = axes[1]; polish(ax)
odd, even = P[P.week.isin([1, 3])], P[P.week.isin([0, 2, 4])]
tpc = lambda q: (q.groupby("user_id").tokens.sum() / q.groupby("user_id").commits.sum()).replace(np.inf, np.nan)
to, te = tpc(odd).dropna(), tpc(even).dropna()
j = to.index.intersection(te.index); j = j[(to[j] > 0) & (te[j] > 0)]
_m = np.median(np.concatenate([to[j].values, te[j].values]))
to, te = np.log10(to[j] / _m), np.log10(te[j] / _m)
qo, qe = to >= to.quantile(0.9), te >= te.quantile(0.9)
ax.scatter(to, te, s=4, color=BLUE, alpha=0.4, lw=0)
ax.scatter(to[qo & qe], te[qo & qe], s=7, color=ORANGE, lw=0, label=f"top decile in both halves ({(qo&qe).sum()})")
ax.scatter(to[qo ^ qe], te[qo ^ qe], s=7, facecolors="none", edgecolors=ORANGE, lw=0.6, label=f"top decile in one half only ({(qo^qe).sum()})")
ax.axvline(to.quantile(0.9), color=ORANGE, lw=0.6, ls=":"); ax.axhline(te.quantile(0.9), color=ORANGE, lw=0.6, ls=":")
r = stats.spearmanr(to, te).correlation
_lo5, _hi5 = int(np.floor(min(to.min(), te.min()))), int(np.ceil(max(to.max(), te.max())))
rel_axis(ax, "x", _lo5, _hi5, "tokens per commit, weeks 1 and 3\n(median = 1)")
rel_axis(ax, "y", _lo5, _hi5, "tokens per commit, weeks 0, 2, 4\n(median = 1)")
ax.set_title("B  Top decile, odd vs even weeks"); ax.legend(loc="upper left")
ax.text(0.98, 0.03, f"Spearman ρ = {r:.2f}\ntop-decile overlap = {(qo&qe).sum()/qo.sum():.0%}", transform=ax.transAxes, fontsize=TEXT, color=MUTED, ha="right", va="bottom")
save(fig, OUT + "fig5-stability")
print("fig5 tpc rho", r, "overlap", (qo & qe).sum() / qo.sum(), "adjacent week corr", [round(M[i, i+1], 2) for i in range(NW - 1)])

# ===================================================== Fig 6: output rise
terc = pd.qcut(X.loc[P.user_id.unique(), "tokens"], 3, labels=["light", "medium", "heavy"])
X["group"] = "non-adopter"; X.loc[adopters, "group"] = np.nan   # adopters with nothing in the window are out
X.loc[terc.index, "group"] = terc.astype(str)
W = PW[PW.week <= 4].join(X[["group"]], on="user_id")
order = ["non-adopter", "light", "medium", "heavy"]; cols = {"non-adopter": GREY, "light": "#b7d3f6", "medium": "#5598e7", "heavy": "#0d366b"}
n = X.group.value_counts()
fig, axes = figure(FULL, 2.0, ncols=3)
# A: five-week spend against the change in the engineer's commit rate since the pre-period
ax = axes[0]; polish(ax)
_T = P.groupby("user_id")[["commits"]].sum().rename(columns={"commits": "tw"})
_c = X.loc[adopters, ["tokens"]].join(pre).join(_T)
_c = _c[(_c.pre_commits > 0) & (_c.tokens > 0) & (_c.tw > 0)]
_c["chg"] = np.log2(_c.tw / 5) - np.log2(_c.pre_commits); _c["ltok"] = np.log10(_c.tokens / _c.tokens.median())
_rho = stats.spearmanr(_c.tokens, _c.chg).correlation
ax.scatter(_c.ltok, _c.chg, s=3, color=BLUE, alpha=0.35, lw=0)
_vb = pd.qcut(_c.ltok.rank(method="first"), 20, labels=False); _vm = _c.groupby(_vb)[["ltok", "chg"]].mean()
ax.scatter(_vm.ltok, _vm.chg, s=8, color=INK, marker="s", zorder=5, label="ventile means")
ax.axhline(0, color=INK, lw=0.6, ls=":")
ax.set_ylim(-4.5, 4.5)
rel_axis(ax, "x", int(np.floor(_c.ltok.min())), int(np.ceil(_c.ltok.max())), "tokens, weeks 0–4 (median engineer = 1)")
ax.set_ylabel("change in commit rate (log2)")
h, _ = ax.get_legend_handles_labels(); h += [Line2D([], [], ls="none", label=f"Spearman ρ = {_rho:.2f}")]
ax.legend(handles=h, loc="upper left")
ax.set_title("A  Spend and change in output")
ax = axes[1]; polish(ax)
idx = W.assign(pre=W.user_id.map(pre)); idx = idx[idx.pre > 0]
def rel_series(frame):
    gm = frame.groupby(["group", "week"]).commits.sum().unstack(0)
    gp = frame[frame.week < 0].groupby(["group", "week"]).commits.sum().unstack(0).mean()
    r = gm / gp; return r.div(r["non-adopter"], axis=0)
rel = rel_series(idx)
for gname in order[1:]:
    ax.plot(rel.index, rel[gname], "o-", ms=2, lw=1.2, color=cols[gname], label=gname)
# heavy without top-10 contributors to the rise
hv = idx[idx.group == "heavy"]
rise = hv[hv.week.between(0, 4)].groupby("user_id").commits.mean() - hv[hv.week < 0].groupby("user_id").commits.mean()
top10 = rise.sort_values(ascending=False).index[:10]
rel2 = rel_series(idx[~idx.user_id.isin(top10)])
ax.plot(rel2.index, rel2["heavy"], "o--", ms=2, lw=1, color="#0d366b", alpha=0.6, label="heavy, without top 10 risers")
ax.axhline(1, color=INK, lw=0.7, ls=":"); ax.axvspan(-0.5, 4.5, color=BLUE, alpha=0.06, lw=0)
ax.set_xlim(-7.3, 4.5); ax.set_ylim(0.6, 2.9)
ax.set_xlabel("week (0 = first week of token data)"); ax.set_ylabel("commits vs own pre-period,\nrelative to non-users")
ax.set_title("B  By usage third"); ax.legend(loc="upper left")
# C: late adopters' event study, all weeks and active weeks only
import json as _json3
_RT = _json3.load(open(Path(__file__).resolve().parents[1] / "out" / "telemetry" / "results.json"))
ax = axes[2]; polish(ax)
for key, col, lab, dx in [("es_dynamic_commits", ORANGE, "all weeks", -0.12), ("es_dynamic_commits_present", BLUE, "active weeks only", 0.12)]:
    es = pd.DataFrame(_RT[key]); es["k"] = es.term.astype(int); es = es.sort_values("k")
    ax.errorbar(es.k + dx, es.estimate, yerr=[es.estimate - es["conf.low"], es["conf.high"] - es.estimate],
                fmt="o", ms=2.5, lw=0.9, color=col, label=lab)
ax.axhline(0, color=INK, lw=0.6, ls=":"); ax.axvline(-0.5, color=MUTED, lw=0.6, ls="--")
ax.set_xlabel("weeks since first metered use"); ax.set_ylabel("log commits vs non-users")
ax.set_title("C  Late adopters"); ax.legend(loc="upper left")
save(fig, OUT + "fig6-output-change")
print("fig6 rho change", _rho, len(_c))
print("fig6 heavy rel", rel["heavy"].round(2).to_dict()); print("without top10", rel2["heavy"].round(2).to_dict())
print("share of heavy rise from top10:", float(rise[top10].sum() / rise.sum()))

# ===================================================== Fig 7: lag
PD = pd.read_csv("analysis/out/person_day.csv", parse_dates=["d"])
days = pd.date_range("2026-06-29", "2026-08-02")
grid = pd.MultiIndex.from_product([adopters, days], names=["user_id", "d"])
D = PD.set_index(["user_id", "d"]).reindex(grid, fill_value=0).reset_index()
D = D[D.d.dt.dayofweek < 5]
D["ltok"] = np.log1p(D.tokens / 1e6); D["lcom"] = np.log1p(D.commits)
for v in ["ltok", "lcom"]:
    D[v] -= D.groupby("user_id")[v].transform("mean"); D[v] -= D.groupby("d")[v].transform("mean")
D = D.sort_values(["user_id", "d"])
xc = {}
for k in range(-10, 11):
    s = D.groupby("user_id").lcom.shift(-k); ok = s.notna()
    xc[k] = float(np.corrcoef(D.ltok[ok], s[ok])[0, 1])
xc = pd.Series(xc)
Dr = PD[PD.user_id.isin(adopters) & PD.d.between("2026-06-29", "2026-08-02")]
tok_days = Dr[Dr.tokens > 0]
nocommit = 1 - tok_days.loc[tok_days.commits > 0, "tokens"].sum() / tok_days.tokens.sum()
fig, ax = figure(COL, 2.0); polish(ax)
ax.bar(xc.index, xc.values, color=[ORANGE if k == 0 else (BLUE if k > 0 else LIGHTBLUE) for k in xc.index], width=0.72)
ax.axhline(0, color=INK, lw=0.7)
ax.set_xlabel("k: commits on weekday t+k vs tokens on day t"); ax.set_ylabel("within-person correlation")
ax.text(-10, 0.27, "commits before\nthe spend", fontsize=TEXT, color=MUTED, va="center")
ax.text(3.2, 0.27, "commits after\nthe spend", fontsize=TEXT, color=MUTED, va="center")
save(fig, OUT + "fig7-lag")
print("lag", xc.round(2).to_dict(), "share tokens on no-commit days", nocommit)

# ===================================================== Fig 8: dark-spend share by group, by window width, per engineer
def fit_f0(df, form="spline", cols=COUNTS):
    df = df.reset_index(drop=True); knn = df.loc[df.commits > 0, cols].median().to_numpy()
    S = terms(df[cols].to_numpy(float), form, knn); Z = np.column_stack([S, week_dummies(df.week)]); gg = codes(df.user_id)
    b, c, mu = fe_poisson(df.tok_m.values, Z, gg); Z0 = Z.copy(); Z0[:, :S.shape[1]] = 0
    return np.exp(Z0 @ b + c[gg])
P["f0"] = fit_f0(P)
X2 = X.loc[P.user_id.unique()]
G = P.join(X2[["role", "manager", "team2"]], on="user_id")
G["usage"] = G.user_id.map(pd.qcut(X2.tokens, 3, labels=["lightest third of users", "middle third", "heaviest third of users"]))
dom = pd.Series(np.where(X2.share_agent_a > 0.9, "Agent A only", np.where(X2.share_agent_b > 0.9, "Agent B only", "both agents")), index=X2.index)   # agents anonymized
G["tool"] = G.user_id.map(dom)
G["rolelab"] = G.role.map({"dev": "developers", "lead_mgmt": "team leads and managers", "qa": "QA engineers", "intern": "interns", "other": None})
G["mgr"] = G.manager.map({0: "non-managers", 1: "managers"})
_top2 = X.team2.value_counts().index[:2]          # the two largest divisions, by headcount
G["divl"] = G.team2.where(G.team2.isin(_top2), None).map({_top2[0]: "largest division", _top2[1]: "second-largest division"})
rows = []
for col, head in [("rolelab", "role"), ("mgr", "manager flag"), ("divl", "division"), ("usage", "usage level"), ("tool", "dominant tool")]:
    s = G.dropna(subset=[col]).groupby(col).agg(tok=("tok_m", "sum"), f0=("f0", "sum"), n=("user_id", "nunique"))
    for lab, r in s.sort_values("tok", ascending=False).iterrows():
        rows.append((head, lab, r.f0 / r.tok, r.tok / P.tok_m.sum(), int(r.n)))
pooled = P.f0.sum() / P.tok_m.sum()
# window-width series (commits only, person-day)
PDd = pd.read_csv("analysis/out/person_day.csv", parse_dates=["d"])
dgrid = pd.MultiIndex.from_product([P.user_id.unique(), pd.date_range("2026-06-29", "2026-08-02")], names=["user_id", "d"])
DD = PDd.set_index(["user_id", "d"]).reindex(dgrid, fill_value=0).reset_index(); DD["day"] = (DD.d - DD.d.min()).dt.days
widths = [1, 3, 7, 14, 28]; series = {k: [] for k in ["floor", "counts", "sqrt", "log1p", "spline"]}
for w in widths:
    W = DD.assign(k=DD.day // w).groupby(["user_id", "k"], as_index=False)[["commits", "tokens"]].sum(); W = W[W.k < 35 // w]
    W = W[W.groupby("user_id").tokens.transform("sum") > 0].reset_index(drop=True); W["tok_m"] = W.tokens / 1e6; W["week"] = W.k
    W = W[(W.tokens > 0) | (W.commits > 0)].reset_index(drop=True)   # panel rule at every width: a block with nothing recorded is out
    thr = W.loc[W.tokens > 0, "tokens"].median(); series["floor"].append(W.tokens[(W.tokens > thr) & (W.commits == 0)].sum() / W.tokens.sum())
    for form in ["counts", "sqrt", "log1p", "spline"]:
        if w == 28 and form != "floor":
            series[form].append(np.nan); continue      # one window per person: no within-person fit
        series[form].append(fit_f0(W, form, ["commits"]).sum() / W.tok_m.sum())
per = P.groupby("user_id").agg(f0=("f0", "sum"), y=("tok_m", "sum")); per["s"] = per.f0 / per.y
print("fig8 pooled", round(pooled, 3), "widths", {k: np.round(v, 2).tolist() for k, v in series.items()}, "per-person median", round(per.s.median(), 2))

fig, ax = figure(COL, 2.0); polish(ax, "x")      # the group panel only (window widths and per-engineer shares are in the text)
ypos = []; y = 0; last = None; labs = []
for head, lab, share, tshare, n in rows:
    if head != last and last is not None: y += 0.4
    last = head; ypos.append(y); labs.append(lab); y += 1
ypos = -np.array(ypos)  # top to bottom
for (head, lab, share, tshare, n), yy in zip(rows, ypos):
    ax.plot([0, share], [yy, yy], color=PALEGREY, lw=0.7, zorder=1)
    ax.scatter(share, yy, s=6 + 110 * tshare, color=BLUE, alpha=0.85, lw=0, zorder=3)
    ax.annotate(f"{share:.0%}", (share, yy), xytext=(np.sqrt(6 + 110 * tshare) / 2 + 2.5, 0), textcoords="offset points", va="center", fontsize=TICK, color=INK, zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
ax.axvline(pooled, color=ORANGE, lw=1, ls="--", zorder=2)
ax.text(pooled + 0.008, ypos.max() + 1.25, f"all users {pooled:.0%}", color=ORANGE, fontsize=TICK, va="center", ha="left")
ax.set_yticks(ypos); ax.set_yticklabels(labs); ax.set_xlim(0, 0.6); ax.set_ylim(ypos.min() - 0.6, ypos.max() + 1.6)
ax.set_xlabel("share of group's tokens not tied to output")
save(fig, OUT + "fig8-dark-by-group")
