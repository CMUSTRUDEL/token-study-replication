import sys, pathlib, re
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from paperstyle import BLUE, ORANGE, INK, PALEGREY, COL, TEXT, TICK, polish, figure, save
OUT = "figs/"
S = pd.read_csv("data/survey/survey_responses.csv"); S.columns=[c.strip() for c in S.columns]
S["agent"] = S["Evaluated tool"].where(S["Evaluated tool"].isin(["Agent A", "Agent B"]), "other")
T = S[(S.Role=="Tech") & S.agent.isin(["Agent A", "Agent B"])]
act = [c for c in S.columns if c.startswith("PWeek_Act_") and not c.endswith("Other")]; val=[c for c in S.columns if c.startswith("PWeek_Value_") and not c.endswith("Other")]
D = T[T[val+act].notna().any(axis=1)]
short = {"Coding and development-related activities":"coding and development", "Processing and analyzing data":"processing and analysing data", "Brainstorming and exploring ideas":"brainstorming and exploring ideas",
         "Doing research on specific topics or projects":"research on a topic or project", "Planning and structuring work":"planning and structuring work", "Finding and summarizing information":"finding and summarising information",
         "Learning and education":"learning", "Writing, editing, or improving long-form text":"writing or editing text", "Onboarding":"onboarding", "Creating communication materials (emails, messages, presentations, meeting notes, etc.)":"emails, messages, presentations"}
a = D[act].notna().mean().rename(lambda c: short[c.replace("PWeek_Act_","")]).sort_values()
eff = ["Effect_overall productivity","Effect_efficiency in completing routine or boilerplate tasks","Effect_efficiency in completing complex tasks","Effect_ efficiency in finding information","Effect_quality of work output","Effect_cognitive load"]
elab = ["overall productivity","routine or boilerplate tasks","complex tasks","finding information","quality of work output","cognitive load (positive = lower)"]
E = []
for c in eff:
    v = T[c].value_counts(); n=v.sum()
    E.append(((v.get("Very positive",0)+v.get("Somewhat positive",0))/n, v.get("No effect",0)/n, (v.get("Very negative",0)+v.get("Somewhat negative",0))/n))
E = np.array(E)
# fig 9a: activities (single column)
fig, ax = figure(COL, 1.95); polish(ax, "x")
ax.barh(a.index, a.values, color=BLUE, height=0.65)
for i,(k,v) in enumerate(a.items()): ax.text(v+0.01, i, f"{v:.0%}", va="center", fontsize=TICK, color=INK)
ax.set_xlim(0, 0.8); ax.set_xticks([0, .2, .4, .6, .8]); ax.set_xticklabels(["0", "20%", "40%", "60%", "80%"])
ax.set_xlabel("share of evaluations naming it"); ax.set_ylim(-0.6, len(a) - 0.4)
for t in ax.get_yticklabels():
    if t.get_text()=="coding and development": t.set_fontweight("bold"); t.set_color(INK)
fig.suptitle(f"Activities last week (n={len(D)})", x=0.01, ha="left", fontsize=7.5, fontweight="bold", color=INK)
save(fig, OUT + "fig9a-activities")
# fig 9b: reported effects (single column)
fig, ax = figure(COL, 1.55); polish(ax, "x")
y = np.arange(len(elab))[::-1]
ax.barh(y, E[:,0], color=BLUE, height=0.65, label="positive"); ax.barh(y, E[:,1], left=E[:,0], color=PALEGREY, height=0.65, label="no effect"); ax.barh(y, E[:,2], left=E[:,0]+E[:,1], color=ORANGE, height=0.65, label="negative")
for yy,(p,z,ng) in zip(y,E):
    ax.text(p/2, yy, f"{p:.0%}", va="center", ha="center", fontsize=TICK, color="white")
    ax.text(p+z/2, yy, f"{z:.0%}", va="center", ha="center", fontsize=TICK if z >= 0.15 else TICK - 1.5, color=INK)
    if ng>=0.2: ax.text(p+z+ng/2, yy, f"{ng:.0%}", va="center", ha="center", fontsize=TICK, color="white")
    elif ng>=0.08: ax.text(1.02, yy, f"{ng:.0%}", va="center", ha="left", fontsize=TICK, color=ORANGE)
ax.set_yticks(y); ax.set_yticklabels(elab); ax.set_xlim(0,1.13); ax.spines["bottom"].set_bounds(0, 1); ax.set_xticks([0,.25,.5,.75,1]); ax.set_xticklabels(["0","25%","50%","75%","100%"])
ax.set_ylim(-0.6, len(elab) - 0.4)
fig.legend(loc="outside upper right", ncol=3, handlelength=1, columnspacing=1)
save(fig, OUT + "fig9b-effects")
print("activities", a.round(2).to_dict()); print("effects", E.round(2).tolist())
