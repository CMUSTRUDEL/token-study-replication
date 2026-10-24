"""Survey numbers the paper reports: the hours-saved measure, its reliability and
selection, non-response, burden, and group differences.

Analysis set: assigned evaluations, one per respondent (occurrence 1), except where
the design uses both evaluations of the 125 respondents who chose a second tool.
Intervals are cluster bootstraps over respondents (common.cluster_boot_ci) or Wilson
intervals for proportions. Writes out/survey/numbers_02_findings.json.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
from common import cluster_boot_ci, prop_ci  # noqa: E402

df = common.load()
A = df[df.occurrence == 1].copy()          # assigned evaluations, one per person
A["W_burden"] = A.W_m + A.W_r

# ---------------------------------------------------------------- hours saved
# signed net hours: saved positive, lost negative, no difference zero
h = cluster_boot_ci(A, "H_net")
q = A["H_net"].quantile([.25, .5, .75, .9])
s1 = cluster_boot_ci(A, "H_net")                                  # "more than 30 hours" capped at 30
s2 = cluster_boot_ci(A[~A.H_net_topcoded], "H_net")               # those answers dropped
A3 = A.copy(); A3.loc[A.H_net_impossible, "H_net"] = 0.0
s3 = cluster_boot_ci(A3, "H_net")                                 # "impossible to tell" as no effect

v = A["H_net"].dropna(); pos = v[v > 0].sort_values(); tot = pos.sum()
allv = v.clip(lower=0).sort_values()      # every respondent with a usable figure; a loss counts as zero hours claimed


def top_share_all(f):
    k = max(1, int(round(len(allv) * f)))
    return allv.iloc[-k:].sum() / allv.sum()


def top_share(f):
    k = max(1, int(round(len(pos) * f)))
    return pos.iloc[-k:].sum() / tot


def gini(x):
    from inequality.gini import Gini          # pysal's inequality package
    return float(Gini(np.asarray(x, float)).g)


d = A[A.Net_effect.notna()]
n_los = int(d.Net_effect.eq("Caused me to lose time").sum())

# ---------------------------------------------------------------- enabled work
k, n = int(A["Enablement past month"].sum()), int(A["Enablement past month"].notna().sum())
# value types of enabled work against the same person's past-week work (exact McNemar)
ev = [f"Enab_Value_{c}" for c in common.ENAB_VALUE if c != "Other"]
pw = [f"PWeek_Value_{c}" for c in common.ENAB_VALUE if c != "Other"]
ey = df[df["Enablement past month"].eq(1) & df[ev].sum(axis=1).gt(0)]
pa = df[df[pw].sum(axis=1).gt(0)]
both = df.loc[sorted(set(ey.index) & set(pa.index))]
rows, pvals = [], []
for c in common.ENAB_VALUE:
    if c == "Other":
        continue
    x, y = both[f"Enab_Value_{c}"].astype(bool), both[f"PWeek_Value_{c}"].astype(bool)
    pv = mcnemar([[int((x & y).sum()), int((x & ~y).sum())],
                  [int((~x & y).sum()), int((~x & ~y).sum())]], exact=True).pvalue
    pvals.append(pv)
    rows.append((c, ey[f"Enab_Value_{c}"].mean(), pa[f"PWeek_Value_{c}"].mean(),
                 x.mean() - y.mean(), pv))

# ---------------------------------------------------------------- selection: assigned vs chosen tool
pan = df[df["Response ID"].isin(df.loc[df.is_second_tool, "Response ID"])].copy()
piv = pan.pivot_table(index="Response ID", columns="occurrence",
                      values=["H_net", "Vendor"], aggfunc="first")
P = pd.DataFrame({"h1": piv[("H_net", 1)], "h2": piv[("H_net", 2)],
                  "v1": piv[("Vendor", 1)], "v2": piv[("Vendor", 2)]}).dropna()
P["delta"] = P.h2 - P.h1
ref = P.v1.value_counts().idxmax()


class _FE:
    """H = person + vendor + chosen, fitted with pyfixest on each person's two
    answers (person fixed effect, CRV3 standard errors clustered by person).
    With two answers per person the coefficients equal those of differencing.
    Exposes params / pvalues / conf_int() keyed 'const' for the chosen-tool
    effect and by vendor name, and nobs = people."""

    def __init__(self, h1, h2, v1, v2, ref):
        import pyfixest as pf
        long = pd.concat([pd.DataFrame({"person": h1.index, "H": h1.values, "vendor": v1.values, "chosen": 0}),
                          pd.DataFrame({"person": h2.index, "H": h2.values, "vendor": v2.values, "chosen": 1})],
                         ignore_index=True)
        m = pf.feols(f"H ~ chosen + i(vendor, ref='{ref}') | person", data=long, vcov={"CRV3": "person"})
        key = lambda k: "const" if k == "chosen" else k.split("::")[-1].split("[T.")[-1].rstrip("]")
        tid = m.tidy()
        self.params = tid["Estimate"].rename(index=key)
        self.pvalues = tid["Pr(>|t|)"].rename(index=key)
        self._ci = tid[["2.5%", "97.5%"]].rename(index=key).set_axis([0, 1], axis=1)
        self.nobs = long.person.nunique()
        self.model = m

    def conf_int(self):
        return self._ci


fe = _FE(P.h1, P.h2, P.v1, P.v2, ref)
fci = fe.conf_int()
sel = fe.params["const"]

# 125 agreed to a second evaluation; the regression needs an hours figure on both.
n_agreed = int(df.loc[df.is_second_tool, "Response ID"].nunique())
_hw = pan.pivot_table(index="Response ID", columns="occurrence", values="H_net",
                      aggfunc="first", dropna=False).reindex(
                          sorted(set(df.loc[df.is_second_tool, "Response ID"])))
_vw = pan.pivot_table(index="Response ID", columns="occurrence", values="Vendor",
                      aggfunc="first")
n_one = int((_hw.notna().sum(axis=1) == 1).sum())
n_none = int((_hw.notna().sum(axis=1) == 0).sum())


def _fe_const(hh):
    dd = pd.DataFrame({"h1": hh[1], "h2": hh[2], "v1": _vw[1], "v2": _vw[2]}).dropna()
    ff = _FE(dd.h1, dd.h2, dd.v1, dd.v2, dd.v1.value_counts().idxmax())
    return ff.params["const"], ff.pvalues["const"], int(ff.nobs)


# the selection effect with the respondents missing one hours figure filled in three ways
_h0 = _hw.copy(); _h0[2] = _h0[2].fillna(_h0[1]); _h0[1] = _h0[1].fillna(_h0[2])
_hz = _hw.copy(); _hz[1] = _hz[1].fillna(0.0); _hz[2] = _hz[2].fillna(0.0)
_hm = _hw.copy(); _hm[1] = _hm[1].fillna(_hw[1].median()); _hm[2] = _hm[2].fillna(_hw[2].median())
IMP = {"no change for the missing person": _fe_const(_h0),
       "missing treated as no effect": _fe_const(_hz),
       "missing filled with the column median": _fe_const(_hm)}

# ---------------------------------------------------------------- reliability of the hours figure
pv2 = P[["h1", "h2"]]
rho2 = pv2.corr(method="spearman").iloc[0, 1]
pvd = pan.pivot_table(index="Response ID", columns="occurrence", values="Net_effect",
                      aggfunc="first").dropna()
agree = (pvd[1] == pvd[2]).mean()
import pingouin as pg  # noqa: E402
_long = pv2.reset_index().melt(id_vars="Response ID", var_name="occ", value_name="h")
# Consistency ICC, two-way, single rating: the systematic shift between the
# assigned and the chosen answer (the selection effect) is removed as an
# occasion effect rather than counted as unreliability.
icc = float(pg.intraclass_corr(data=_long, targets="Response ID", raters="occ", ratings="h")
            .set_index("Type").loc["ICC(C,1)", "ICC"])
maxr = np.sqrt(max(icc, 0.0))
n_mgr = int(A.TeamLead.eq("Manager").sum()); n_ic = int(A.TeamLead.eq("IC").sum())
det = TTestIndPower().solve_power(effect_size=None, nobs1=n_mgr, ratio=n_ic / n_mgr,
                                  alpha=0.05, power=0.8) * A["H_net"].std()
obs_gap = (A.loc[A.TeamLead.eq("Manager"), "H_net"].mean()
           - A.loc[A.TeamLead.eq("IC"), "H_net"].mean())

conf = A["Confidence_hours_saved"]
hi, lo = A[conf >= 9]["H_net"].mean(), A[conf <= 3]["H_net"].mean()
rho3 = A[["Confidence_hours_saved", "H_net"]].corr(method="spearman").iloc[0, 1]

# ---------------------------------------------------------------- non-response and analytic choices
n_all, obs = len(A), A["H_net"].dropna()
lower = obs.sum() / n_all                                          # missing respondents at zero
upper = (obs.sum() + (n_all - len(obs)) * 30.0) / n_all           # missing respondents at the 30-hour cap
# Headcount: the public figure of about 2,600 employees. Respondents: 412 (400 who
# evaluated a tool, 12 non-users).
N_EMP, N_RESP = 2600, 412
rr = N_RESP / N_EMP
specs = []
for unit, top, imp, pt, st in itertools.product(
        ["assigned", "all"], ["cap30", "drop", "cap40"], ["drop", "zero", "median"],
        ["keep", "drop"], ["mean", "median"]):
    g = df[df.occurrence == 1] if unit == "assigned" else df
    if pt == "drop":
        g = g[g.Status.eq("Complete")]
    vv = g["H_net"].copy()
    if top == "drop":
        vv = vv.mask(g.H_net_topcoded)
    elif top == "cap40":
        vv = vv.mask(g.H_net_topcoded, 40.0)
    if imp == "zero":
        vv = vv.mask(g.H_net_impossible, 0.0)
    elif imp == "median":
        vv = vv.mask(g.H_net_impossible, float(vv.dropna().median()))
    vv = vv.dropna()
    if len(vv) >= 30:
        specs.append({"stat": st, "unit": unit,
                      "est": float(vv.mean() if st == "mean" else vv.median())})
S = pd.DataFrame(specs); Sm = S[S.stat == "mean"]

# ---------------------------------------------------------------- who reports more: role, team lead, vendor
m = A.dropna(subset=["H_net"]).copy()
m = m[m.Role.isin(["Tech", "Non-tech"]) & m.TeamLead.isin(["IC", "Manager"])]
m["tech"] = m.Role.eq("Tech").astype(int)
m["mgr"] = m.TeamLead.eq("Manager").astype(int)
m["vendor_a"] = m.Vendor.eq("Vendor A").astype(int)
m["vendor_b"] = m.Vendor.eq("Vendor B").astype(int)
ols = smf.ols("H_net ~ tech + mgr + vendor_a + vendor_b", data=m).fit(cov_type="HC3")
base = cluster_boot_ci(A[A.Vendor.eq(ref)], "H_net")[0]
vb_naive = cluster_boot_ci(A[A.Vendor.eq("Vendor B")], "H_net")[0] - base

# ---------------------------------------------------------------- burden of colleagues' AI output
ex = A["Exposure to AI generated materials"]
p_ex = prop_ci(int(ex.eq("Yes").sum()), int(ex.notna().sum()))
pb = cluster_boot_ci(A, "W_burden")
pm, pr = cluster_boot_ci(A, "W_m"), cluster_boot_ci(A, "W_r")
pdd = cluster_boot_ci(A, "W_d")                                   # colleague time the tool displaced
nc = A["W_d_n_colleagues"]
pdz = cluster_boot_ci(A[A.W_d > 0], "W_d")

exm = A[A["Exposure to AI generated materials"].notna()].copy()
exm["y"] = exm["Exposure to AI generated materials"].eq("Yes").astype(int)
exm = exm[exm.Role.isin(["Tech", "Non-tech"]) & exm.TeamLead.isin(["IC", "Manager"])]
exm["mgr"] = exm.TeamLead.eq("Manager").astype(int)
exm["tech"] = exm.Role.eq("Tech").astype(int)
lg = smf.logit("y ~ mgr + tech", data=exm).fit(disp=0)
lci = lg.conf_int()

# every group comparison in one family, Benjamini-Hochberg
tests = [("team leads vs ICs: exposure to others' AI output", lg.pvalues["mgr"])]
for lab, col in [("hours saved", "H_net"), ("colleague hours displaced", "W_d"),
                 ("burden borne", "W_burden")]:
    tests.append((f"team leads vs ICs: {lab}", stats.mannwhitneyu(
        A.loc[A.TeamLead.eq("Manager"), col].dropna(),
        A.loc[A.TeamLead.eq("IC"), col].dropna()).pvalue))
for lab, col in [("hours saved", "H_net"), ("burden borne", "W_burden")]:
    tests.append((f"technical vs non-technical: {lab}", stats.mannwhitneyu(
        A.loc[A.Role.eq("Tech"), col].dropna(),
        A.loc[A.Role.eq("Non-tech"), col].dropna()).pvalue))
en = A[A["Enablement past month"].notna()]
for grp, lab, lv in [("Role", "technical vs non-technical", ["Tech", "Non-tech"]),
                     ("TeamLead", "team leads vs ICs", ["Manager", "IC"])]:
    t = pd.crosstab(en[grp], en["Enablement past month"].eq(1)).reindex(
        index=lv, columns=[False, True])
    tests.append((f"{lab}: enablement rate", stats.fisher_exact(t.to_numpy())[1]))
for c, lab in common.EFFECT_ITEMS.items():
    tests.append((f"team leads vs ICs: {lab}", stats.mannwhitneyu(
        A.loc[A.TeamLead.eq("Manager"), c + "_num"].dropna(),
        A.loc[A.TeamLead.eq("IC"), c + "_num"].dropna()).pvalue))
names = [t[0] for t in tests]; raw = np.array([t[1] for t in tests])
rej, adj, _, _ = multipletests(raw, alpha=0.05, method="fdr_bh")

# ---------------------------------------------------------------- the six effect items
cog = A["Effect_cognitive load_num"].dropna()
prod = A["Effect_overall productivity_num"].dropna()
rho4 = A[["H_net", "Effect_cognitive load_num"]].corr(method="spearman").iloc[0, 1]

# respondents who use no company-paid AI tool (a separate export)
nu = pd.read_csv(common.DATA.parent / "non_users.csv")

NUM = {
  "hours_mean": h[0], "hours_mean_lo": h[1], "hours_mean_hi": h[2], "hours_n": int(h[3]),
  "hours_q25": float(q[.25]), "hours_median": float(q[.5]), "hours_q90": float(q[.9]),
  "hours_cap30": s1[0], "hours_drop_top": s2[0], "hours_impossible_zero": s3[0],
  "hours_top10_share": top_share(.10), "hours_gini": gini(pos), "hours_n_positive": int(len(pos)),
  "hours_top10_share_all": top_share_all(.10), "hours_gini_all": gini(allv), "hours_n_all": int(len(allv)),
  "dir_saved": float(d.Net_effect.eq("Saved me time").mean()), "dir_lost": float(d.Net_effect.eq("Caused me to lose time").mean()),
  "dir_none": float(d.Net_effect.eq("No difference").mean()), "dir_n_lost": n_los,
  "enab_k": k, "enab_n": n, "enab_share": k / n, "enab_lo": prop_ci(k, n)[1], "enab_hi": prop_ci(k, n)[2],
  "enab_both_n": int(len(both)), "enab_mcnemar_pmin": float(min(pvals)), "enab_mcnemar_pmax": float(max(pvals)),
  "enab_rows": [[c, float(x), float(y), float(dd), float(pv)] for c, x, y, dd, pv in rows],
  "sel": float(sel), "sel_lo": float(fci.loc["const", 0]), "sel_hi": float(fci.loc["const", 1]), "sel_p": float(fe.pvalues["const"]),
  "sel_n": int(len(P)), "sel_agreed": n_agreed, "sel_one": n_one, "sel_none": n_none,
  "sel_vb_to_va": int((P.v1.eq("Vendor B") & P.v2.eq("Vendor A")).sum()), "sel_va_to_vb": int((P.v1.eq("Vendor A") & P.v2.eq("Vendor B")).sum()),
  "sel_imp": {k_: float(v_[0]) for k_, v_ in IMP.items()}, "sel_imp_p": {k_: float(v_[1]) for k_, v_ in IMP.items()},
  "sel_h1_mean": float(P.h1.mean()), "sel_h2_mean": float(P.h2.mean()),
  "retest_rho": float(rho2), "retest_gap": float((pv2.h1 - pv2.h2).abs().mean()), "retest_agree": float(agree),
  "icc": icc, "max_r": float(maxr), "det_gap": float(det), "obs_gap_mgr_ic": float(obs_gap),
  "conf_hi_mean": float(hi), "conf_lo_mean": float(lo), "conf_rho": float(rho3),
  "nr_lower": float(lower), "nr_upper": float(upper), "nr_rate": rr, "nr_mult": 1 / rr,
  "spec_n": int(len(Sm)), "spec_min": float(Sm.est.min()), "spec_max": float(Sm.est.max()),
  "ols_vendor_a": float(ols.params["vendor_a"]), "ols_vendor_a_p": float(ols.pvalues["vendor_a"]),
  "ols_vendor_b": float(ols.params["vendor_b"]), "ols_vendor_b_p": float(ols.pvalues["vendor_b"]),
  "ols_tech": float(ols.params["tech"]), "ols_tech_p": float(ols.pvalues["tech"]),
  "ols_mgr": float(ols.params["mgr"]), "ols_mgr_p": float(ols.pvalues["mgr"]),
  "fe_vendor_a": float(fe.params["Vendor A"]), "fe_vendor_a_p": float(fe.pvalues["Vendor A"]),
  "fe_vendor_b": float(fe.params["Vendor B"]), "fe_vendor_b_p": float(fe.pvalues["Vendor B"]), "vb_naive": float(vb_naive),
  "burden_exposed": p_ex[0], "burden_exposed_lo": p_ex[1], "burden_exposed_hi": p_ex[2],
  "burden_n": int(ex.notna().sum()), "burden_unsure": float(ex.eq("I'm not sure").mean()),
  "burden_hours": pb[0], "burden_hours_lo": pb[1], "burden_hours_hi": pb[2], "burden_making_sense": pm[0], "burden_rework": pr[0],
  "colleague_hours": pdd[0], "colleague_hours_lo": pdd[1], "colleague_hours_hi": pdd[2],
  "colleague_none": int((nc == 0).sum()), "colleague_n": int(nc.notna().sum()), "colleague_hours_pos": pdz[0],
  "colleague_hours_pos_lo": pdz[1], "colleague_hours_pos_hi": pdz[2],
  "conf_colleague": float(A.Confidence_colleague_time.mean()), "conf_own": float(A.Confidence_hours_saved.mean()),
  "lead_or": float(np.exp(lg.params["mgr"])), "lead_or_lo": float(np.exp(lci.loc["mgr", 0])), "lead_or_hi": float(np.exp(lci.loc["mgr", 1])),
  "lead_p": float(lg.pvalues["mgr"]), "bh_tests": len(tests), "bh_survive": int(rej.sum()),
  "bh_survivors": [names[i] for i in range(len(names)) if rej[i]],
  "cog_harm": float((cog < 0).mean()), "prod_harm": float((prod < 0).mean()), "rho_hours_cog": float(rho4),
  "effect_harm": {lab: float((A[c + "_num"].dropna() < 0).mean()) for c, lab in common.EFFECT_ITEMS.items()},
  "nonusers": int(len(nu)),
  "n_mgr": n_mgr, "n_ic": n_ic,
}
common.OUT.mkdir(parents=True, exist_ok=True)
(common.OUT / "numbers_02_findings.json").write_text(json.dumps(NUM, indent=1, default=float))
print(f"wrote {common.OUT / 'numbers_02_findings.json'} ({len(NUM)} numbers)")
