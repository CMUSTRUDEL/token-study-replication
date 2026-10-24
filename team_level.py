"""Team-level coupling of token spend and output (Appendix B, "Teams").

Does a team's spend follow the team's output more closely than an engineer's spend follows
the engineer's? Person-weeks of the panel are summed to team-weeks, and the paper's
within-person and between-person estimators are applied to teams:

  * within team: Poisson regression of team-week tokens on log team-week commits with team and
    week fixed effects (fe_poisson from analyze_panel.py), on team-weeks with tokens and commits,
    teams with two or more such weeks; also with log(active members) as a control, because a
    week in which more members are present has both more tokens and more commits;
  * between teams: least squares of team-mean log tokens per active member on team-mean log
    commits per active member (per member, so team size does not drive the slope);
  * intervals: 95% percentile intervals from a team-cluster bootstrap;
  * like-for-like: the within-person slope on the members of the same teams;
  * aggregation check: the same within-team slope for random groups of the same sizes drawn
    from the same engineers, to separate the effect of summing from the effect of teams.

Units: "teams" are the finest team paths, "ecosystems" the level-3 units the paper calls product
ecosystems; only units with at least MIN_MEMBERS metered users enter. Writes out/team_level.json.
Run after the companion pipeline (reads out/companion/work).
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "out" / "companion" / "work"
MIN_MEMBERS = 5
B = 999            # team-cluster bootstrap replicates
R = 200            # random regroupings for the aggregation check

# the companion pipeline's estimators, taken verbatim from analyze_panel.py
_src = (HERE / "companion" / "analyze_panel.py").read_text()
_ns = {"np": np, "pd": pd}
for node in ast.parse(_src).body:
    if isinstance(node, ast.FunctionDef) and node.name in {"fe_poisson", "codes", "week_dummies"}:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "analyze_panel.py", "exec"), _ns)
fe_poisson, codes, week_dummies = _ns["fe_poisson"], _ns["codes"], _ns["week_dummies"]

X = pd.read_csv(W / "analysis/out/people.csv", index_col=0)
PW = pd.read_csv(W / "analysis/out/person_week.csv")
adopters = X.index[X.has_usage == 1]
ALL = PW[PW.user_id.isin(adopters) & PW.week.isin(range(5))].copy()
P = ALL[(ALL.tokens > 0) | (ALL.commits > 0) | (ALL.prs_authored > 0)].reset_index(drop=True)   # the panel rule
P["tok_m"] = P.tokens / 1e6
rng = np.random.default_rng(20261022)


def team_weeks(panel, unit):
    """Sum a panel to unit-weeks; 'members' counts the members with a panel week that week."""
    g = panel.groupby([unit, "week"], as_index=False).agg(
        tokens=("tokens", "sum"), commits=("commits", "sum"), members=("user_id", "nunique"))
    g["tok_m"] = g.tokens / 1e6
    return g


def within(tw, unit, control=False):
    """Within-unit Poisson slope of tokens on log commits, unit and week effects."""
    d = tw[(tw.tokens > 0) & (tw.commits > 0)]
    d = d[d.groupby(unit).week.transform("nunique") >= 2]
    cols = [np.log(d.commits.to_numpy(float))] + ([np.log(d.members.to_numpy(float))] if control else [])
    Z = np.column_stack(cols + [week_dummies(d.week)])
    b, _, _ = fe_poisson(d.tok_m.values, Z, codes(d[unit]))
    return float(b[0]), d


def within_ols(tw, unit, control=False):
    """Least squares on two-way demeaned logs; with control, partial out log(active members) first."""
    d = tw[(tw.tokens > 0) & (tw.commits > 0)].copy()
    d = d[d.groupby(unit).week.transform("nunique") >= 2]
    srcs = [("ly", np.log(d.tokens)), ("lx", np.log(d.commits)), ("lm", np.log(d.members))]
    for v, s in srcs:
        d[v] = s - s.groupby(d[unit]).transform("mean")
        d[v] = d[v] - d[v].groupby(d.week).transform("mean")
    if control:
        M = np.column_stack([d.lx, d.lm]); b = np.linalg.lstsq(M, d.ly.to_numpy(), rcond=None)[0]
        return float(b[0]), np.nan
    return float((d.ly * d.lx).sum() / (d.lx ** 2).sum()), float(np.corrcoef(d.ly, d.lx)[0, 1])


def person_within_ols(panel):
    d = panel[(panel.commits > 0) & (panel.tokens > 0)].copy()
    d = d[d.groupby("user_id").week.transform("nunique") >= 2]
    for v, s in [("ly", np.log(d.tokens)), ("lx", np.log(d.commits))]:
        d[v] = s - s.groupby(d.user_id).transform("mean")
        d[v] = d[v] - d[v].groupby(d.week).transform("mean")
    return float((d.ly * d.lx).sum() / (d.lx ** 2).sum())


def between(tw, unit, per_member=True):
    """Least squares across units of mean log tokens on mean log commits, per member by default."""
    d = tw[(tw.tokens > 0) & (tw.commits > 0)].copy()
    d = d[d.groupby(unit).week.transform("nunique") >= 2]
    k = d.members if per_member else 1.0
    d["ly"], d["lx"] = np.log(d.tokens / k), np.log(d.commits / k)
    for v in ["ly", "lx"]:
        d[v] -= d.groupby("week")[v].transform("mean") - d[v].mean()
    m = d.groupby(unit)[["lx", "ly"]].mean()
    return float(np.polyfit(m.lx, m.ly, 1)[0]), int(len(m))


def person_within(panel):
    """The paper's within-person slope (d_fe in analyze_panel.py), on the given people."""
    d = panel[(panel.commits > 0) & (panel.tokens > 0)]
    d = d[d.groupby("user_id").week.transform("nunique") >= 2]
    Z = np.column_stack([np.log(d.commits.to_numpy(float)), week_dummies(d.week)])
    b, _, _ = fe_poisson(d.tok_m.values, Z, codes(d.user_id))
    return float(b[0]), d


def boot_units(tw, unit, fn, reps=B):
    ids = tw[unit].unique(); grp = {u: d for u, d in tw.groupby(unit)}; vals = []
    for _ in range(reps):
        pick = rng.choice(ids, len(ids), replace=True)
        bd = pd.concat([grp[u].assign(**{unit: f"{u}#{j}"}) for j, u in enumerate(pick)], ignore_index=True)
        try:
            vals.append(fn(bd))
        except Exception:
            continue
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def boot_people(panel, fn, reps=B):
    ids = panel.user_id.unique(); grp = {u: d for u, d in panel.groupby("user_id")}; vals = []
    for _ in range(reps):
        pick = rng.choice(ids, len(ids), replace=True)
        vals.append(fn(pd.concat([grp[u].assign(user_id=f"{u}#{j}") for j, u in enumerate(pick)], ignore_index=True)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


OUT = {"min_members": MIN_MEMBERS, "bootstrap_reps": B, "random_regroupings": R}
for unit, col in [("team", "team"), ("ecosystem", "team3")]:
    size = X.loc[adopters, col].value_counts()
    keep = size.index[size >= MIN_MEMBERS]
    Q = P.join(X[[col]], on="user_id")
    Q = Q[Q[col].isin(keep)].rename(columns={col: "unit"}).reset_index(drop=True)
    tw = team_weeks(Q, "unit")
    est, d = within(tw, "unit")
    est_c, _ = within(tw, "unit", control=True)
    o, r = within_ols(tw, "unit")
    bt, n_b = between(tw, "unit")
    pw_est, pw_d = person_within(Q)
    # aggregation check: random groups of the same sizes from the same engineers
    users = Q.user_id.unique(); sizes = Q.groupby("unit").user_id.nunique().to_numpy()
    rand, rand_ols = [], []
    for _ in range(R):
        perm = rng.permutation(users); lab = np.repeat(np.arange(len(sizes)), sizes)
        Qr = Q.assign(unit=Q.user_id.map(dict(zip(perm, lab))))
        twr = team_weeks(Qr, "unit")
        rand.append(within(twr, "unit")[0]); rand_ols.append(within_ols(twr, "unit")[0])
    OUT[unit] = dict(
        units=int(tw.unit.nunique()), members=int(Q.user_id.nunique()),
        members_per_unit_median=float(np.median(sizes)),
        unit_weeks=int(len(d)), units_in_fit=int(d.unit.nunique()),
        within_ppml=est, within_ppml_ci=boot_units(tw, "unit", lambda q: within(q, "unit")[0]),
        within_ppml_members=est_c, within_ppml_members_ci=boot_units(tw, "unit", lambda q: within(q, "unit", True)[0]),
        within_ols=o, within_corr=r, within_ols_members=within_ols(tw, "unit", True)[0],
        within_ols_ci=boot_units(tw, "unit", lambda q: within_ols(q, "unit")[0]),
        person_within_ols_same_people=person_within_ols(Q),
        random_groups_within_ols_median=float(np.median(rand_ols)),
        random_groups_within_ols_p05_p95=[float(np.percentile(rand_ols, 5)), float(np.percentile(rand_ols, 95))],
        share_random_ols_ge_actual=float(np.mean(np.array(rand_ols) >= o)),
        between_per_member=bt, between_units=n_b, between_totals=between(tw, "unit", per_member=False)[0],
        members_size_corr=float(np.corrcoef(np.log(tw.groupby("unit").members.mean()), np.log(tw.groupby("unit").tokens.mean()))[0, 1]),
        between_per_member_ci=boot_units(tw, "unit", lambda q: between(q, "unit")[0]),
        person_within_same_people=pw_est, person_within_same_people_n=int(pw_d.user_id.nunique()),
        person_within_same_people_ci=boot_people(Q, lambda q: person_within(q)[0]),
        random_groups_within_median=float(np.median(rand)),
        random_groups_within_p05_p95=[float(np.percentile(rand, 5)), float(np.percentile(rand, 95))],
        share_random_ge_actual=float(np.mean(np.array(rand) >= est)))
    # robustness: other size cutoffs, and leaving out each unit in turn (main cutoff)
    sens = {}
    for mn in ([3, 5, 8] if unit == "team" else [3, 5, 10]):
        kk = size.index[size >= mn]
        Qs = P.join(X[[col]], on="user_id"); Qs = Qs[Qs[col].isin(kk)].rename(columns={col: "unit"}).reset_index(drop=True)
        sens[str(mn)] = dict(units=int(Qs.unit.nunique()), within_ppml=within(team_weeks(Qs, "unit"), "unit")[0],
                             person_within_same_people=person_within(Qs)[0])
    loo = [within(tw[tw.unit != x], "unit")[0] for x in d.unit.unique()]
    OUT[unit]["by_min_members"] = sens
    OUT[unit]["drop_one_unit_range"] = [float(min(loo)), float(max(loo))]
    print(unit, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in OUT[unit].items()})

(HERE / "out" / "team_level.json").write_text(json.dumps(OUT, indent=1))
print("wrote out/team_level.json")
