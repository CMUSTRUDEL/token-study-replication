"""Exploratory scan of associations across survey questions, technical Agent A / Agent B evaluations.
Joins closed items, hours, checkboxes, exposure to others' AI output, and the free-text codes.
Not the reported analysis: the paper reports the one pre-specified model in paper_figs_survey_model.py,
and keeps from this scan only the comparisons of the coded free text with the closed items.
paper_figs_survey_model.py executes the data-preparation block that precedes the outcomes list."""
import numpy as np, pandas as pd, json
from scipy import stats
S = pd.read_csv("data/survey/survey_responses.csv"); S.columns=[c.strip() for c in S.columns]
S["agent"] = S["Evaluated tool"].map({"Agent A": "A", "Agent B": "B"}).fillna("other")
T = S[(S.Role=="Tech") & S.agent.isin(["A", "B"])].copy()
T["key"] = T["Response ID"].astype(str) + "-" + T.agent
code = {"Very negative":1,"Somewhat negative":2,"No effect":3,"Somewhat positive":4,"Very positive":5}
eff = {"productivity":"Effect_overall productivity","routine":"Effect_efficiency in completing routine or boilerplate tasks","complex":"Effect_efficiency in completing complex tasks","findinfo":"Effect_ efficiency in finding information","quality":"Effect_quality of work output","cogload":"Effect_cognitive load"}
for k,c in eff.items(): T["E_"+k] = T[c].map(code)
net = T["Net_effect"].map({"Saved me time":1,"No difference":0,"Caused me to lose time":-1})
_cap = {"More than 30 hours": 30, "More than 40": 40, "More than 10": 10}   # dropdown ceilings, capped as in survey/common.py, not dropped
hs = pd.to_numeric(T["Hours saved"].replace(_cap), errors="coerce"); hl = pd.to_numeric(T["Hours lost"].replace(_cap), errors="coerce")
T["net_hours"] = np.where(net==1, hs, np.where(net==0, 0.0, np.where(net==-1, -hl, np.nan)))
T["confidence"] = pd.to_numeric(T["Confidence"], errors="coerce")
act = [c for c in T.columns if c.startswith("PWeek_Act_") and not c.endswith("Other")]
T["n_activities"] = T[act].notna().sum(axis=1); T["coding"] = T["PWeek_Act_Coding and development-related activities"].notna().astype(int)
T["n_noncoding"] = T["n_activities"] - T["coding"]
for c in act: T["A_"+c.replace("PWeek_Act_","").split(" ")[0].lower()] = T[c].notna().astype(int)
for c in ["Experiments and learning","Workflow improvements","Core job execution","Organizational impact"]: T["V_"+c.split(" ")[0].lower()] = T["PWeek_Value_"+c].notna().astype(int)
T["V_experiments_only"] = ((T.V_experiments==1) & (T[["V_workflow","V_core","V_organizational"]].sum(axis=1)==0)).astype(int)
T["enab_month"] = T["Enablement past month"]
T["exposed"] = T["Exposure to AI generated materials"].map({"Yes":1,"No":0})
T["read_hours"] = pd.to_numeric(T["Reading, interpreting, or making sense of AI-generated materials"], errors="coerce")
T["rework_hours"] = pd.to_numeric(T["Improving and reworking AI-generated materials"], errors="coerce")
T["colleagues_n"] = pd.to_numeric(T["N of other colleagues needed"], errors="coerce"); T["colleagues_hours"] = pd.to_numeric(T["Colleagues time spent"], errors="coerce")
T["is_agent_a"] = (T.agent=="A").astype(int); T["div_top"] = (T.Department==pd.read_csv("data/people.csv").team.str.split(r" \\ ").str[1].value_counts().idxmax()).astype(int)   # the largest division
T["second_tool"] = (T["Status.1"].notna()).astype(int) if "Status.1" in T else 0
# free-text codes
C = pd.read_csv("analysis/out/survey_thematic_codes.csv")
fam={"ENAB":{"UNFAM":"reach","DEFER":"expand","PROTO":"expand","INVEST":"invest","BIG":"speed","TICKET":"speed","TOOLING":"expand","DOCS":"speed","PARALLEL":"parallel","QUAL":"speed"},
     "SAVE":{"DELEG":"speed","NAV":"reach","INVEST":"invest","IDEAS":"expand","REVIEW":"quality","PARALLEL":"parallel","WORKFLOW":"workflow","RESEARCH":"invest","START":"speed"},
     "VALUE":{"SPEED":"speed","BRIDGE":"reach","UPSHIFT":"workflow","INVEST":"invest","EXPAND":"expand","BUSY":"speed","QUALITY":"quality","LESSASK":"reach","QUAL":"qualified","PARALLEL":"parallel"}}
flags = {}
for _,r in C.iterrows():
    d = flags.setdefault(r.response, {})
    for c in str(r.codes).split(";"):
        if r.question=="LOSE": d["L_"+c] = 1
        else:
            f = fam[r.question].get(c)
            if f: d["B_"+f] = 1
        d["answered_"+r.question] = 1
F = pd.DataFrame.from_dict(flags, orient="index").fillna(0)
T = T.set_index("key").join(F).reset_index()
for c in F.columns: T[c] = T[c].fillna(0)
T["L_substantive"] = T[[c for c in T.columns if c.startswith("L_") and c not in ("L_NONE","L_UNCLEAR","L_substantive")]].max(axis=1)
T["n_lose_themes"] = T[[c for c in T.columns if c.startswith("L_") and c not in ("L_NONE","L_UNCLEAR","L_substantive")]].sum(axis=1)

outcomes = ["net_hours","E_productivity","E_quality","E_cogload","E_complex","E_routine","confidence","n_activities","colleagues_n"]
binaries = ["coding","V_experiments","V_experiments_only","V_core","V_workflow","enab_month","exposed","is_agent_a","div_top","L_WRONG","L_VERIFY","L_LOAD","L_TOOL","L_PROMPT","L_OVER","L_WAIT","L_SELF","L_COMPLEX","L_OTHERS","L_NONE","B_reach","B_expand","B_speed","B_invest","B_workflow","B_parallel","B_quality","B_qualified","A_coding","A_learning","A_brainstorming","A_processing","A_planning","A_onboarding"]
ordinals = ["n_activities","n_noncoding","n_lose_themes","read_hours","rework_hours","colleagues_n","colleagues_hours","confidence","E_cogload","E_quality","E_productivity","net_hours"]
rows = []
for y in outcomes:
    for x in binaries:
        if x==y: continue
        d = T[[x,y]].dropna(); a = d[d[x]==1][y]; b = d[d[x]==0][y]
        if len(a)<8 or len(b)<8: continue
        p = stats.mannwhitneyu(a,b).pvalue
        rows.append(dict(y=y, x=x, kind="binary", n1=len(a), n0=len(b), stat_1=round(a.median(),2), stat_0=round(b.median(),2), mean_1=round(a.mean(),2), mean_0=round(b.mean(),2), cliff=round(2*stats.mannwhitneyu(a,b).statistic/(len(a)*len(b))-1,2), p=p))
    for x in ordinals:
        if x==y: continue
        d = T[[x,y]].dropna()
        if len(d)<20: continue
        r,p = stats.spearmanr(d[x], d[y]); rows.append(dict(y=y, x=x, kind="spearman", n=len(d), rho=round(r,2), p=p))
R = pd.DataFrame(rows).drop_duplicates(subset=["y","x"])
# BH
R = R.sort_values("p"); m=len(R); R["q"] = (R.p*m/np.arange(1,m+1)).cummin()[::-1].cummin()[::-1] if False else np.minimum.accumulate((R.p.values*m/np.arange(1,m+1))[::-1])[::-1]
R.to_csv("analysis/out/survey_crossq.csv", index=False)
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
print("tests:", m, " q<0.10:", int((R.q<0.10).sum()), " p<0.05:", int((R.p<0.05).sum()))
print(R[R.p<0.05].to_string(index=False))
