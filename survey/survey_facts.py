"""Survey facts the paper quotes that survey_numbers.py does not write out.

Each value is computed directly from the export with the coding in common.py.
Writes out/survey/numbers_descriptive.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

df = common.load()
A = df[df.occurrence == 1].copy()
N = {}

# respondents, evaluations, rounds
N["respondents"] = int(df["Response ID"].nunique())
N["evaluations"] = int(len(df))
N["one_eval"] = int((df.groupby("Response ID").size() == 1).sum())
N["two_evals"] = int((df.groupby("Response ID").size() == 2).sum())
N["tools"] = int(df["Evaluated tool"].nunique())
N["vendors"] = int(df["Vendor"].nunique())
agent = df["Evaluated tool"].isin(["Agent A", "Agent B"])
N["agent_evals"] = int(agent.sum())
N["share_technical"] = float(A.Role.eq("Tech").mean())
N["share_team_leads"] = float(A.TeamLead.eq("Manager").mean())

# the two rounds are selected differently: the same people's assigned and chosen
# evaluations, paired, each measure on the people who answered it twice
two = df[df["Response ID"].isin(df.loc[df.is_second_tool, "Response ID"])]
for col, key in [("Enablement past month", "enab"), ("H_net", "hours"),
                 ("Effect_overall productivity_num", "prod")]:
    pv = two.pivot_table(index="Response ID", columns="occurrence", values=col, aggfunc="first").dropna()
    N[f"{key}_assigned_paired"] = float(pv[1].mean())
    N[f"{key}_chosen_paired"] = float(pv[2].mean())
    N[f"{key}_paired_n"] = int(len(pv))

# all numeric evaluations, assigned and chosen
_h = df.H_net.dropna()
N["all_evals_numeric"] = int(len(_h)); N["all_evals_mean"] = float(_h.mean()); N["all_evals_median"] = float(_h.median())
# reported hours by team-lead status (assigned evaluations)
N["hours_mean_leads"] = float(A.loc[A.TeamLead.eq("Manager"), "H_net"].mean())
N["hours_mean_ics"] = float(A.loc[A.TeamLead.eq("IC"), "H_net"].mean())
N["burden_mean_leads"] = float((A.W_m + A.W_r)[A.TeamLead.eq("Manager")].mean())
N["burden_mean_ics"] = float((A.W_m + A.W_r)[A.TeamLead.eq("IC")].mean())

# censoring: assigned evaluations (the basis of every mean) and all evaluations
N["n_over_30_assigned"] = int(A.H_net_topcoded.sum())
N["n_impossible_assigned"] = int(A.H_net_impossible.sum())
N["n_over_30_all"] = int(df.H_net_topcoded.sum())
N["n_impossible_all"] = int(df.H_net_impossible.sum())

# reach and attrition (assigned evaluations; the last block is asked once per person)
N["reached_hours"] = float(A.Net_effect.notna().mean())
N["reached_burden"] = float(A["Exposure to AI generated materials"].notna().mean())
quit_ = A[A.Status.ne("Complete")]; done = A[A.Status.eq("Complete")]
saved = lambda g: g.Net_effect.dropna().eq("Saved me time")
N["saved_share_quitters"] = float(saved(quit_).mean())
N["saved_share_finishers"] = float(saved(done).mean())
tab = [[int(saved(quit_).sum()), int((~saved(quit_)).sum())],
       [int(saved(done).sum()), int((~saved(done)).sum())]]
N["quit_fisher_p"] = float(stats.fisher_exact(tab)[1])
N["quit_hours_gap"] = float(done.H_net.mean() - quit_.H_net.mean())

# components: the hours figure against the six effect items
# across all 525 evaluations
num = ["H_net"] + [c + "_num" for c in common.EFFECT_ITEMS]
cm = df[num].corr(method="spearman")
N["rho_hours_items"] = {lab: float(cm.loc["H_net", c + "_num"]) for c, lab in common.EFFECT_ITEMS.items()}
items = [c + "_num" for c in common.EFFECT_ITEMS]
off = df[items].corr(method="spearman").where(lambda m: m < 1).mean()
N["item_offdiag_min"] = float(off.min())
N["item_offdiag_max"] = float(off.max())

# response IDs missing from the export
ids = sorted(df["Response ID"].astype(int).unique())
gaps = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))
N["missing_ids"] = len(gaps)

out = common.OUT
out.mkdir(parents=True, exist_ok=True)
(out / "numbers_descriptive.json").write_text(json.dumps(N, indent=1))
for k, v in N.items():
    print(k, round(v, 4) if isinstance(v, float) else v)
