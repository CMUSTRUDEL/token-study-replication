"""Thematic coding of open-text survey answers (technical respondents, Agents A and B).
One coder, inductive, single pass. Codes per answer ID (response ID - agent) are recorded here verbatim.
Reads data/survey/survey_open_text.txt, the free-text answers tagged [QUESTION|ID-AGENT] (not released)."""
import sys, pathlib
import pandas as pd, numpy as np, re, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from paperstyle import BLUE, ORANGE, AQUA, GREY, INK, FULL, TICK, polish, figure, save
OUT = "figs/"

LOSE_BOOK = {"WRONG":"wrong, misleading or low-quality output","OVER":"over-engineering, bloat, overreach","COMPLEX":"weak on complex or legacy code","VERIFY":"verifying and reviewing its output",
  "PROMPT":"prompting, context and steering","WAIT":"waiting and context switching","TOOL":"tooling friction: setup, integrations, outages, quotas","COST":"token or credit cost",
  "LOAD":"cognitive load, flood of generated content","OTHERS":"AI-generated material from others","SELF":"faster or better to do it myself","SCOPE":"doing more than needed","NONE":"nothing or negligible","UNCLEAR":"unclear"}
LOSE = {"108-A":["NONE"],"122-B":["UNCLEAR"],"139-A":["PROMPT"],"144-B":["OVER","WRONG"],"157-B":["WRONG"],"172-B":["TOOL"],"200-A":["WRONG","SELF"],"201-A":["VERIFY","WRONG"],
 "216-A":["WRONG","TOOL"],"226-B":["COMPLEX","PROMPT","VERIFY"],"258-A":["NONE"],"259-B":["WRONG"],"260-A":["OVER"],"264-A":["WRONG","SELF"],"280-B":["PROMPT","WRONG"],"290-B":["WRONG","SCOPE"],
 "293-A":["WAIT","COST","WRONG"],"295-A":["VERIFY"],"312-B":["SCOPE"],"324-A":["NONE"],"340-A":["SCOPE","LOAD"],"342-B":["WRONG"],"357-B":["TOOL"],"389-A":["WRONG"],"395-A":["WRONG","OVER","PROMPT"],
 "397-B":["PROMPT"],"406-B":["OTHERS","WRONG"],"407-B":["PROMPT","COMPLEX","COST"],"427-B":["COMPLEX"],"432-A":["VERIFY"],"437-A":["TOOL"],"449-A":["WRONG","TOOL"],"467-A":["WRONG","COMPLEX"],
 "468-A":["WRONG","OVER"],"471-A":["NONE"],"472-B":["WAIT"],"504-B":["WRONG"],"508-B":["WRONG"],"525-B":["COST","TOOL"],"537-A":["PROMPT"],"553-B":["WRONG"],"638-B":["SELF"],"646-A":["TOOL","COMPLEX"],
 "88-A":["NONE"],"90-A":["TOOL","WAIT","PROMPT","WRONG"],"101-B":["PROMPT"],"160-A":["WRONG","WAIT"],"162-B":["WRONG","OVER","VERIFY"],"168-A":["WRONG","VERIFY"],"170-A":["VERIFY","WRONG","PROMPT","WAIT"],
 "179-A":["LOAD"],"229-A":["PROMPT"],"240-B":["WRONG"],"266-B":["TOOL"],"271-A":["WRONG"],"276-B":["TOOL"],"282-B":["OVER","WRONG","PROMPT"],"290-A":["NONE"],"357-A":["COMPLEX","WRONG"],
 "364-A":["LOAD","OTHERS"],"382-A":["OTHERS"],"407-A":["WRONG","COMPLEX"],"425-A":["WRONG","VERIFY"],"429-A":["TOOL","PROMPT"],"453-A":["WAIT","TOOL"],"491-A":["SELF","VERIFY"],"526-A":["TOOL","PROMPT"],
 "535-A":["TOOL"],"548-A":["WRONG","VERIFY"],"552-A":["TOOL"],"576-A":["WAIT"],"579-A":["TOOL"],"624-A":["WRONG","VERIFY","COMPLEX"],"629-B":["OVER","VERIFY","SCOPE"]}

ENAB_BOOK = {"UNFAM":"work in an unfamiliar codebase, stack or domain","DEFER":"tasks that would have been deferred or handed off","PROTO":"prototypes, experiments, hypothesis tests","INVEST":"investigation at scale (logs, large codebases)",
  "BIG":"large refactorings, features or test suites","TICKET":"difficult tickets or bugs closed","TOOLING":"internal tools, scripts, automation","DOCS":"documents, reports, charts","PARALLEL":"work in parallel with other tasks","QUAL":"not 'wouldn't have', just faster"}
ENAB = {"84-A":["UNFAM"],"87-B":["UNFAM","BIG"],"115-A":["TICKET"],"139-A":["UNFAM"],"159-A":["TICKET"],"201-A":["UNFAM"],"212-A":["QUAL"],"280-B":["DEFER","UNFAM"],"293-A":["UNFAM","INVEST"],
 "295-A":["DEFER","TOOLING"],"302-A":["PROTO","UNFAM"],"340-A":["BIG"],"342-B":["QUAL"],"345-B":["INVEST"],"386-B":["UNFAM","TOOLING","PROTO"],"407-B":["TOOLING","UNFAM"],"437-A":["TICKET","DOCS"],
 "470-B":["BIG","TICKET","DEFER"],"556-B":["PROTO","INVEST"],"566-A":["TOOLING","DEFER"],"575-B":["BIG"],"88-A":["PROTO","TOOLING"],"152-A":["TOOLING","DEFER"],"160-A":["INVEST"],"170-A":["TICKET","UNFAM"],
 "179-A":["PROTO","TOOLING"],"229-A":["DOCS","DEFER"],"240-B":["BIG","INVEST"],"266-B":["PROTO"],"276-B":["QUAL"],"282-B":["PROTO"],"323-A":["UNFAM","PARALLEL","TOOLING"],"346-A":["PROTO"],
 "349-A":["DEFER","DOCS"],"364-A":["DOCS","UNFAM"],"382-A":["BIG"],"390-B":["BIG"],"425-A":["QUAL","DOCS"],"446-A":["QUAL"],"460-A":["UNFAM","DEFER"],"491-A":["DEFER","INVEST"],"526-A":["BIG","TOOLING"],
 "535-A":["UNFAM","DEFER"],"552-A":["PROTO"],"576-A":["PROTO","DEFER"],"579-A":["BIG"],"616-A":["INVEST","UNFAM"],"624-A":["UNFAM"]}

SAVE_BOOK = {"DELEG":"delegating routine or well-specified implementation","NAV":"navigating unfamiliar code, frameworks, domains","INVEST":"investigation, debugging, log and data analysis","IDEAS":"brainstorming, design feedback, hypotheses",
  "REVIEW":"agent-run code review","PARALLEL":"working on several tasks in parallel","WORKFLOW":"workflow investment: skills, MCPs, plugins","RESEARCH":"research, information gathering, drafting","START":"getting started on a task","NONE":"none or qualified"}
SAVE = {"99-A":["DELEG","WORKFLOW","PARALLEL"],"108-A":["NAV","REVIEW"],"117-B":["WORKFLOW"],"136-A":["DELEG"],"139-A":["IDEAS","DELEG"],"144-B":["NAV","DELEG"],"157-B":["NAV"],"200-A":["NAV","INVEST"],"201-A":["NAV"],
 "212-A":["WORKFLOW","DELEG"],"216-A":["WORKFLOW"],"226-B":["DELEG"],"247-A":["WORKFLOW"],"258-A":["NONE"],"259-B":["INVEST","NAV"],"260-A":["DELEG"],"264-A":["NAV"],"280-B":["WORKFLOW"],"290-B":["NAV","DELEG"],
 "293-A":["NAV","DELEG"],"295-A":["NONE"],"302-A":["NAV","PARALLEL"],"312-B":["NAV"],"324-A":["NAV","DELEG"],"340-A":["NAV","REVIEW","DELEG"],"342-B":["DELEG"],"357-B":["WORKFLOW"],"362-A":["DELEG","RESEARCH"],
 "389-A":["IDEAS","NAV"],"395-A":["DELEG","IDEAS"],"406-B":["REVIEW"],"407-B":["IDEAS","DELEG"],"432-A":["INVEST","NAV"],"437-A":["DELEG","RESEARCH"],"449-A":["INVEST","DELEG"],"467-A":["DELEG","NAV"],
 "471-A":["DELEG","RESEARCH"],"504-B":["NAV","INVEST"],"508-B":["WORKFLOW"],"525-B":["WORKFLOW"],"537-A":["WORKFLOW"],"548-B":["IDEAS","REVIEW"],"553-B":["PARALLEL","DELEG","INVEST"],"558-A":["NAV","DELEG"],
 "575-B":["DELEG"],"626-A":["NAV","DELEG"],"638-B":["REVIEW"],"646-A":["DELEG","NAV"],"88-A":["NONE"],"90-A":["INVEST","RESEARCH","PARALLEL"],"101-B":["DELEG"],"160-A":["INVEST","NAV","DELEG"],"162-B":["IDEAS","DELEG"],
 "168-A":["DELEG","NONE"],"170-A":["DELEG"],"172-A":["WORKFLOW","DELEG"],"179-A":["DELEG","WORKFLOW"],"180-A":["RESEARCH","DELEG"],"219-A":["RESEARCH","IDEAS"],"229-A":["DELEG"],"240-B":["DELEG","NAV"],
 "266-B":["DELEG"],"271-A":["NAV"],"276-B":["DELEG","NONE"],"282-B":["IDEAS"],"290-A":["NONE"],"300-A":["WORKFLOW","RESEARCH"],"303-A":["NAV","REVIEW"],"323-A":["DELEG","NAV"],"349-A":["DELEG","NAV"],
 "357-A":["WORKFLOW"],"360-A":["RESEARCH"],"364-A":["RESEARCH","NAV","DELEG"],"380-A":["IDEAS","DELEG"],"382-A":["DELEG"],"390-B":["DELEG"],"404-A":["NAV","REVIEW"],"407-A":["NAV","REVIEW"],"429-A":["NONE"],
 "453-A":["IDEAS","DELEG","INVEST"],"460-A":["REVIEW","DELEG","RESEARCH"],"491-A":["START","RESEARCH","INVEST"],"526-A":["WORKFLOW"],"535-A":["DELEG","INVEST","PARALLEL"],"548-A":["WORKFLOW"],"552-A":["DELEG","NAV"],
 "576-A":["IDEAS"],"579-A":["DELEG"],"624-A":["NAV"]}

VALUE_BOOK = {"SPEED":"speed and throughput on usual tasks","BRIDGE":"bridging into unfamiliar code, domains, technologies","UPSHIFT":"role shifts to design, orchestration, review","INVEST":"investigation, debugging, triage, analysis",
  "EXPAND":"exploration, prototyping, expanding what is worth attempting","BUSY":"busywork removed, easier to start","QUALITY":"quality: tests, corner cases, review, audits","LESSASK":"less dependence on colleagues","QUAL":"qualified or negative"}
VALUE = {"99-A":["SPEED"],"136-A":["BRIDGE"],"139-A":["SPEED"],"144-B":["BRIDGE"],"157-B":["BRIDGE"],"200-A":["BUSY","BRIDGE","INVEST"],"212-A":["UPSHIFT","QUAL"],"216-A":["BRIDGE"],"226-B":["QUAL"],"259-B":["INVEST"],
 "260-A":["BUSY"],"280-B":["BUSY","EXPAND"],"290-B":["EXPAND"],"293-A":["BRIDGE","SPEED"],"295-A":["LESSASK","QUAL","EXPAND"],"302-A":["BRIDGE","LESSASK","EXPAND"],"324-A":["BRIDGE","EXPAND"],"340-A":["BRIDGE","QUALITY","BUSY"],
 "342-B":["UPSHIFT","SPEED"],"345-B":["INVEST"],"357-B":["BUSY","BRIDGE"],"362-A":["SPEED","BUSY"],"388-B":["SPEED"],"389-A":["QUAL"],"395-A":["QUALITY","BRIDGE"],"397-B":["SPEED"],"406-B":["QUALITY","BRIDGE","QUAL"],
 "407-B":["EXPAND","QUAL"],"427-B":["INVEST"],"437-A":["BUSY","UPSHIFT","QUALITY"],"449-A":["BUSY","BRIDGE","INVEST"],"467-A":["INVEST"],"468-A":["BRIDGE","QUAL"],"471-A":["BUSY","UPSHIFT","QUALITY"],"472-B":["EXPAND"],
 "484-A":["SPEED","QUAL"],"504-B":["BRIDGE","INVEST"],"508-B":["QUALITY"],"525-B":["EXPAND"],"537-A":["EXPAND","QUALITY"],"553-B":["SPEED","EXPAND"],"554-A":["QUAL"],"558-A":["BRIDGE"],"638-B":["BRIDGE","INVEST"],
 "644-B":["INVEST"],"646-A":["SPEED","QUAL"],"88-A":["SPEED"],"90-A":["INVEST","LESSASK"],"160-A":["INVEST"],"162-B":["QUALITY","SPEED"],"168-A":["BUSY","QUAL"],"179-A":["EXPAND","SPEED"],"180-A":["INVEST"],
 "219-A":["EXPAND"],"229-A":["BUSY"],"240-B":["BRIDGE","LESSASK"],"266-B":["EXPAND","SPEED"],"276-B":["SPEED"],"282-B":["SPEED","EXPAND"],"300-A":["INVEST","SPEED"],"349-A":["SPEED"],"357-A":["BUSY"],"360-A":["INVEST","EXPAND"],
 "380-A":["SPEED","BRIDGE"],"390-B":["SPEED"],"404-A":["BRIDGE"],"407-A":["BRIDGE","QUALITY","QUAL"],"425-A":["INVEST"],"429-A":["QUAL"],"453-A":["EXPAND","QUALITY"],"460-A":["BRIDGE"],"491-A":["INVEST","QUALITY"],
 "515-A":["SPEED","UPSHIFT","QUALITY"],"526-A":["SPEED"],"541-A":["INVEST"],"552-A":["SPEED"],"576-A":["EXPAND"],"624-A":["EXPAND"]}

# ---- verify coverage against the dump and count
txt = open("data/survey/survey_open_text.txt").read()
ids = {k: re.findall(rf"\[{k}\|([^\]]+)\]", txt) for k in ["LOSE","ENAB","SAVE","VALUE"]}
rows = []
res = {}
for key, codes, book in [("LOSE", LOSE, LOSE_BOOK), ("ENAB", ENAB, ENAB_BOOK), ("SAVE", SAVE, SAVE_BOOK), ("VALUE", VALUE, VALUE_BOOK)]:
    missing = set(ids[key]) - set(codes); extra = set(codes) - set(ids[key])
    assert not missing and not extra, (key, missing, extra)
    for c in codes.values():
        for code in c: assert code in book, (key, code)
    subst = {k: v for k, v in codes.items() if not set(v) <= {"NONE", "UNCLEAR", "QUAL"}}
    n = len(codes); ns = len(subst)
    counts = pd.Series([c for v in subst.values() for c in set(v)]).value_counts()
    share = (counts / ns).rename(lambda c: book[c])
    res[key] = (n, ns, share)
    for rid, c in codes.items():
        rows.append(dict(question=key, response=rid, codes=";".join(c)))
    print(f"\n== {key}: {n} answers, {ns} substantive ==")
    print(share.round(2).to_string())
pd.DataFrame(rows).to_csv("analysis/out/survey_thematic_codes.csv", index=False)
book = pd.DataFrame([(q, c, d) for q, b in [("LOSE", LOSE_BOOK), ("ENAB", ENAB_BOOK), ("SAVE", SAVE_BOOK), ("VALUE", VALUE_BOOK)] for c, d in b.items()], columns=["question", "code", "theme"])
book.to_csv("analysis/out/survey_thematic_codebook.csv", index=False)

# ---- figure: four panels of theme shares
titles = {"LOSE": "A  What cost time", "ENAB": "B  Enabled work", "SAVE": "A  What saves time", "VALUE": "B  Main value"}
colors = {"LOSE": GREY, "ENAB": AQUA, "SAVE": BLUE, "VALUE": ORANGE}
# shorter display names for the figure only (the codebook CSV keeps the full theme names)
SHORT = {"wrong, misleading or low-quality output": "wrong or low-quality output",
         "tooling friction: setup, integrations, outages, quotas": "tooling friction (setup, outages, quotas)",
         "verifying and reviewing its output": "verifying its output",
         "over-engineering, bloat, overreach": "over-engineering, bloat",
         "faster or better to do it myself": "faster to do it myself",
         "cognitive load, flood of generated content": "cognitive load, flood of content",
         "AI-generated material from others": "others' AI-generated material",
         "work in an unfamiliar codebase, stack or domain": "unfamiliar codebase, stack, domain",
         "tasks that would have been deferred or handed off": "tasks otherwise deferred or handed off",
         "prototypes, experiments, hypothesis tests": "prototypes, experiments",
         "large refactorings, features or test suites": "large refactorings, features, tests",
         "investigation at scale (logs, large codebases)": "investigation at scale",
         "difficult tickets or bugs closed": "difficult tickets or bugs",
         "work in parallel with other tasks": "parallel work",
         "not 'wouldn't have', just faster": "nothing new, just faster",
         "delegating routine or well-specified implementation": "delegating routine implementation",
         "navigating unfamiliar code, frameworks, domains": "navigating unfamiliar code",
         "workflow investment: skills, MCPs, plugins": "workflow setup: skills, MCPs",
         "brainstorming, design feedback, hypotheses": "brainstorming, design feedback",
         "investigation, debugging, log and data analysis": "investigation, debugging",
         "research, information gathering, drafting": "research, drafting",
         "working on several tasks in parallel": "several tasks in parallel",
         "speed and throughput on usual tasks": "speed on usual tasks",
         "bridging into unfamiliar code, domains, technologies": "bridging into unfamiliar code",
         "exploration, prototyping, expanding what is worth attempting": "exploration, prototyping",
         "investigation, debugging, triage, analysis": "investigation, debugging, triage",
         "quality: tests, corner cases, review, audits": "quality: tests, review, audits",
         "role shifts to design, orchestration, review": "role shifts to design, review"}
# two figures: what cost time and enabled work in the text (fig10), what saved time and main value in the appendix (fig10b)
for keys, height, stem in [(["LOSE", "ENAB"], 1.9, "fig10-themes"), (["SAVE", "VALUE"], 1.7, "fig10b-themes-rest")]:
    fig, axes = figure(FULL, height, ncols=2, sharex=True)
    for ax, key in zip(axes.flat, keys):
        n, ns, share = res[key]
        s = share.sort_values(kind="stable").rename(lambda k: SHORT.get(k, k))
        polish(ax, "x")
        ax.barh(s.index, s.values, color=colors[key], height=0.7)
        for i, (k, v) in enumerate(s.items()): ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=TICK, color=INK)
        ax.set_xlim(0, 0.65); ax.set_ylim(-0.6, len(s) - 0.4)
        ax.set_xticks([0, .2, .4, .6]); ax.set_xticklabels(["0", "20%", "40%", "60%"])
        ax.set_title(f"{titles[key]} (n={ns})")
    fig.supxlabel("share of answers coded to the theme", fontsize=7)
    save(fig, OUT + stem)
print("\nmean codes per answer:", {k: round(np.mean([len(v) for v in d.values()]), 2) for k, d in [("LOSE", LOSE), ("ENAB", ENAB), ("SAVE", SAVE), ("VALUE", VALUE)]})
