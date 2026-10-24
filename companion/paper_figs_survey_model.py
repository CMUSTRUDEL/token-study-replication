"""One pre-specified model of reported hours saved, technical Agent A / Agent B evaluations.

Replaces the pairwise scan of survey_crossq.py as the reported analysis. The outcome is
net hours saved last week (asinh-transformed, so losses stay negative). The predictors are
fixed in advance by the argument the survey serves: two reach measures (the agent enabled
otherwise-skipped work; number of non-coding activity types), one speed measure (used it
for coding), one measurement check (colleagues otherwise needed), and two controls
(tool, division).
Confidence in the estimate is left out: it is missing for a third of the answers and is
treated as a measurement check in the hours section instead. Standardised coefficients with 95% bootstrap intervals,
resampling respondents. Writes figs/fig11-survey-model.{png,pdf} and figs/fig11b-colleagues.{png,pdf}. Run from the repo root.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = open("analysis/survey_crossq.py").read().split("outcomes = [")[0]
exec(src)  # builds T, the 197 technical Agent A / Agent B evaluations with derived columns

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from paperstyle import BLUE, ORANGE, GREY, INK, MUTED, FULL, COL, TICK, polish, figure, save
OUT = "figs/"

T["y"] = np.arcsinh(T.net_hours); T["log_coll"] = np.log1p(T.colleagues_n)
preds = ["enab_month", "n_noncoding", "coding", "V_experiments_only", "log_coll", "is_agent_a", "div_top"]
labels = {"enab_month": "enabled skipped work (yes)", "n_noncoding": "non-coding activity types (count)",
          "coding": "used it for coding (yes)", "V_experiments_only": "only experiments so far (yes)",
          "log_coll": "colleagues needed (log count)", "confidence": "confidence in the estimate (1–10)",
          "is_agent_a": "Agent A rather than Agent B", "div_top": "largest division"}
d = T[["y"] + preds + ["Response ID"]].dropna()
Z = (d[preds] - d[preds].mean()) / d[preds].std(ddof=0); X = np.column_stack([np.ones(len(Z)), Z.values]); y = d.y.values
b = np.linalg.lstsq(X, y, rcond=None)[0]; r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
rng = np.random.default_rng(20261006); ids = d["Response ID"].values; uid = np.unique(ids); bs = []
for _ in range(3000):
    idx = np.concatenate([np.where(ids == u)[0] for u in rng.choice(uid, len(uid))])
    bs.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
coef = pd.DataFrame({"beta": b[1:], "lo": lo[1:], "hi": hi[1:]}, index=preds)
print(f"n={len(d)} evaluations, {len(uid)} respondents, R2={r2:.2f}"); print(coef.round(2).to_string())

# Two stacked panels with different left margins (B's long tick labels), so the axes are
# placed by hand: each is grown to fill its own slot of the figure, decorations included.
fig = plt.figure(figsize=(COL, 2.8))
axA = fig.add_axes([0.15, 0.62, 0.8, 0.3]); axB = fig.add_axes([0.5, 0.1, 0.45, 0.3])
def fit(ax, slot, pad=0.01):
    """Move and resize ax so its tight bbox (ticks, labels, title) fills slot = (x0, y0, x1, y1) in figure fractions."""
    for _ in range(2):
        fig.canvas.draw()
        tb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()); p = ax.get_position()
        dl, db = slot[0] + pad - tb.x0, slot[1] + pad - tb.y0
        dr, dt = slot[2] - pad - tb.x1, slot[3] - pad - tb.y1
        ax.set_position([p.x0 + dl, p.y0 + db, p.width - dl + dr, p.height - db + dt])
def strip(ax, col, bins, lab, title):
    g = pd.cut(T[col], bins, labels=lab); dd = T[["net_hours"]].assign(g=g).dropna()
    for i, l in enumerate(lab):
        v = dd.net_hours[dd.g == l]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=5, color=BLUE, alpha=0.45, lw=0)
        ax.plot([i - 0.28, i + 0.28], [v.median()] * 2, color=ORANGE, lw=1.6, solid_capstyle="butt")
        ax.annotate(f"n={len(v)}", (i, 0), xycoords=("data", "axes fraction"), xytext=(0, -11), textcoords="offset points",
                    ha="center", va="top", fontsize=TICK, color=MUTED)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab); ax.set_yscale("symlog", linthresh=2, linscale=0.35)
    ax.set_yticks([-2, 0, 2, 5, 10, 20, 40, 80]); ax.set_yticklabels(["−2", "0", "2", "5", "10", "20", "40", "80"]); ax.minorticks_off()
    ax.set_ylim(-5, 90); ax.set_ylabel("hours saved last week"); polish(ax)
    if title: ax.set_title(title)
    ax.plot([], [], color=ORANGE, lw=1.6, label="median"); ax.legend(loc="upper left")
ax = axA; strip(ax, "n_noncoding", [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"], "A  By non-coding activities")
ax.set_xlabel("non-coding activity types ticked", labelpad=8)
ax = axB; polish(ax, "x")
order = preds[::-1]; yy = np.arange(len(order))
for i, p in enumerate(order):
    c = coef.loc[p]; col = BLUE if (c.lo > 0 or c.hi < 0) else GREY
    ax.plot([c.lo, c.hi], [i, i], color=col, lw=1.2); ax.plot(c.beta, i, "o", color=col, ms=3.5)
ax.axvline(0, color=INK, lw=0.6, ls=":"); ax.set_yticks(yy); ax.set_yticklabels([labels[p] for p in order])
ax.set_xlabel("standardised coefficient (95% CI)"); ax.set_xlim(-0.6, 0.65)
ax.set_title("B  Others held fixed")
fit(axA, (0, 0.5, 1, 1)); fit(axB, (0, 0, 1, 0.5))
save(fig, OUT + "fig11-survey-model")
figB, axB = figure(COL, 1.6)
strip(axB, "colleagues_n", [-1, 0, 1, 2, 99], ["0", "1", "2", "3+"], None)
axB.set_xlabel("colleagues otherwise needed", labelpad=8)
save(figB, OUT + "fig11b-colleagues")
