"""Shared loading and coding for the company's AI value survey.

Every coding decision that is not mechanical is written here, once, with a
comment saying what was decided and why. Analyses import from this module so
that no number in the outputs comes from an undocumented transformation.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TOKEN_SPEND_DATA", ROOT / "data")) / "survey" / "survey_responses.csv"
OUT = ROOT / "out" / "survey"

# Blocks asked ONCE PER PERSON rather than once per evaluated tool. The burden
# questions are about other people's AI output in general, not about the tool
# being evaluated, so the instrument asks them only in the first pass. All 299
# answers sit on occurrence-1 rows and none on occurrence-2 rows. Any rate over
# these columns must use 400 respondents as its denominator, never 525 rows.
PER_PERSON_COLS = [
    "Exposure to AI generated materials",
    "Reading, interpreting, or making sense of AI-generated materials",
    "Improving and reworking AI-generated materials",
]

# --- column groups -----------------------------------------------------------

ENAB_VALUE = ["Experiments and learning", "Workflow improvements",
              "Core job execution", "Organizational impact", "Other"]
ACTIVITIES = ["Finding and summarizing information",
              "Doing research on specific topics or projects",
              "Brainstorming and exploring ideas",
              "Planning and structuring work",
              "Writing, editing, or improving long-form text",
              "Coding and development-related activities",
              "Processing and analyzing data",
              "Learning and education",
              "Onboarding",
              "Creating communication materials (emails, messages, presentations, "
              "meeting notes, etc.)",
              "Other"]

EFFECT_ITEMS = {
    "Effect_overall productivity": "overall productivity",
    "Effect_quality of work output": "quality of work output",
    "Effect_efficiency in completing routine or boilerplate tasks": "routine-task efficiency",
    "Effect_efficiency in completing complex tasks": "complex-task efficiency",
    "Effect_cognitive load": "cognitive load (positive = reduced)",
    "Effect_ efficiency in finding information": "finding information",
}
LIKERT = {"Very negative": -2, "Somewhat negative": -1, "No effect": 0,
          "Somewhat positive": 1, "Very positive": 2}

# Dropdown answers that are values, not numbers. Kept as flags rather than
# silently dropped, because they are a substantial share of the responses
# (e.g. 63 of 434 hours-saved answers are "More than 30 hours" or "Impossible
# to tell") and dropping them quietly would bias every mean downward.
TOPCODE = {"More than 30 hours": 30.0, "More than 40": 40.0, "More than 10": 10.0}
UNKNOWN = {"Impossible to tell"}


def split_dropdown(s: pd.Series):
    """Return (numeric, is_topcoded, is_unknown) for a capped-dropdown column.

    `numeric` holds the literal number where one was given and the cap value
    where the respondent chose the top bucket ("More than 30 hours" -> 30).
    A top-coded value is therefore a *lower bound*, flagged so analyses can
    report with and without it. "Impossible to tell" stays NaN and is flagged.
    """
    raw = s.astype("object")
    num = pd.to_numeric(raw, errors="coerce")
    topc = raw.isin(TOPCODE)
    unk = raw.isin(UNKNOWN)
    num = num.where(~topc, raw.map(TOPCODE))
    return num.astype(float), topc.fillna(False), unk.fillna(False)


def load(drop_partial: bool = False) -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df.columns = [c.strip() for c in df.columns]

    df["Vendor"] = df["Vendor"].str.strip()

    # `Status.1` is the survey's "Are you a team lead?" yes/no, recoded to
    # Manager/IC. It is *not* seniority: a principal engineer with no reports is
    # an IC. It is also independent of `Role`, which asks whether the work is
    # technical and instructs anyone leading a technical team to answer
    # Technical — so a technical team lead is Tech + Manager.
    df = df.rename(columns={"Status.1": "TeamLead",
                            "Confidence": "Confidence_hours_saved",
                            "Confidence.1": "Confidence_hours_lost",
                            "Confidence.2": "Confidence_colleague_time",
                            "How confident are you in the time estimate you’ve just provided?":
                                "Confidence_burden"})

    # occurrence: 1 = the tool the instrument assigned first, 2 = the optional
    # second tool the respondent volunteered to also evaluate.
    df["occurrence"] = df.groupby("Response ID").cumcount() + 1
    df["is_second_tool"] = df["occurrence"] == 2

    # --- H_net: signed net hours for the reference work week -----------------
    # "No difference" is a real zero, not missing. Saved is positive, lost is
    # negative. Respondents who never reached the magnitude question stay NaN.
    saved, saved_top, saved_unk = split_dropdown(df["Hours saved"])
    lost, lost_top, lost_unk = split_dropdown(df["Hours lost"])
    net = pd.Series(np.nan, index=df.index, dtype=float)
    net = net.mask(df["Net_effect"].eq("Saved me time"), saved)
    net = net.mask(df["Net_effect"].eq("Caused me to lose time"), -lost)
    net = net.mask(df["Net_effect"].eq("No difference"), 0.0)
    df["H_net"] = net
    df["H_net_topcoded"] = saved_top | lost_top
    df["H_net_impossible"] = saved_unk | lost_unk

    # --- W_d: colleague time the tool substituted for ------------------------
    n_coll, n_coll_top, _ = split_dropdown(df["N of other colleagues needed"])
    w_d, w_d_top, w_d_unk = split_dropdown(df["Colleagues time spent"])
    df["W_d_n_colleagues"] = n_coll
    df["W_d_n_topcoded"] = n_coll_top
    # A respondent who said "0 colleagues" is a substantive zero for W_d even
    # though the follow-up was never shown to them.
    df["W_d"] = w_d.mask(df["N of other colleagues needed"].eq("0"), 0.0)
    df["W_d_topcoded"] = w_d_top
    df["W_d_impossible"] = w_d_unk

    # --- W_m / W_r: burden imposed by other people's AI output ---------------
    w_m, w_m_top, w_m_unk = split_dropdown(
        df["Reading, interpreting, or making sense of AI-generated materials"])
    w_r, w_r_top, w_r_unk = split_dropdown(
        df["Improving and reworking AI-generated materials"])
    # "No" exposure is a substantive zero; "I'm not sure" is left missing.
    no_exposure = df["Exposure to AI generated materials"].eq("No")
    df["W_m"] = w_m.mask(no_exposure, 0.0)
    df["W_r"] = w_r.mask(no_exposure, 0.0)
    df["W_m_topcoded"], df["W_r_topcoded"] = w_m_top, w_r_top
    df["W_m_impossible"], df["W_r_impossible"] = w_m_unk, w_r_unk

    # --- Likert items --------------------------------------------------------
    # "Not applicable" was not exported distinctly; it is blank, the same as a
    # drop-out. Both stay NaN and every Likert n is reported alongside the mean.
    for col in EFFECT_ITEMS:
        df[col + "_num"] = df[col].map(LIKERT)

    # --- multi-selects: blank means not selected, within a shown block -------
    for col in df.columns:
        if col.startswith(("Enab_Value_", "Enab_Act_", "PWeek_Value_", "PWeek_Act_")):
            if df[col].dtype == object:
                continue  # the two "Other" columns hold free text
            df[col] = df[col].fillna(0).astype(int)

    if drop_partial:
        df = df[df["Status"].eq("Complete")].copy()
    return df


def respondent_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per person: the first-assigned tool only.

    Use for any 'share of employees who...' statistic. The second-tool rows are
    a self-selected subsample (only 125 of the 299 offered accepted), so
    pooling all 525 rows would over-weight heavy multi-tool users.
    """
    return df[df["occurrence"] == 1].copy()


# --- inference ---------------------------------------------------------------

def cluster_boot_ci(df, value_col, stat=np.mean, cluster="Response ID",
                    n_boot=4000, alpha=0.05, seed=0):
    """Percentile bootstrap CI resampling *clusters* (respondents), not rows.

    125 respondents contribute two rows. Resampling rows would treat those as
    independent and understate the standard error, so we resample respondents
    and take all of their rows together. Done with scipy.stats.bootstrap: each
    respondent is the pair (sum of their values, number of their rows), the
    pair is resampled together (paired=True), and the row mean of a resample is
    sum over sum.
    """
    if stat is not np.mean:
        raise ValueError("cluster_boot_ci supports the mean only")
    from scipy.stats import bootstrap
    d = df[[cluster, value_col]].dropna(subset=[value_col])
    if d.empty:
        return (np.nan, np.nan, np.nan, 0)
    g = d.groupby(cluster)[value_col].agg(["sum", "count"])
    res = bootstrap((g["sum"].to_numpy(float), g["count"].to_numpy(float)),
                    lambda s, c, axis: s.sum(axis=axis) / c.sum(axis=axis),
                    paired=True, vectorized=True, n_resamples=n_boot,
                    confidence_level=1 - alpha, method="percentile", random_state=seed)
    point = float(d[value_col].mean())
    return (point, float(res.confidence_interval.low), float(res.confidence_interval.high), len(d))


def prop_ci(k, n, alpha=0.05):
    """Wilson score interval for a proportion (statsmodels proportion_confint)."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return (k / n, float(lo), float(hi))


def fmt_ci(point, lo, hi, pct=False, dec=2):
    if pct:
        return f"{point:.0%} [{lo:.0%}, {hi:.0%}]"
    return f"{point:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]"


def boot_diff(a, b, n_boot=20000, alpha=0.05, seed=3):
    """mean(a) - mean(b) for independent groups: percentile bootstrap CI
    (scipy.stats.bootstrap) and a two-sided permutation p-value
    (scipy.stats.permutation_test).

    Used where the two groups are disjoint sets of respondents, so there is no
    cluster structure left to preserve within the comparison.
    """
    from scipy.stats import bootstrap, permutation_test
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    diff = lambda x, y, axis: x.mean(axis=axis) - y.mean(axis=axis)
    ci = bootstrap((a, b), diff, vectorized=True, n_resamples=n_boot, confidence_level=1 - alpha,
                   method="percentile", random_state=seed).confidence_interval
    p = permutation_test((a, b), diff, vectorized=True, n_resamples=n_boot, alternative="two-sided",
                         permutation_type="independent", random_state=seed).pvalue
    return a.mean() - b.mean(), float(ci.low), float(ci.high), float(p)
