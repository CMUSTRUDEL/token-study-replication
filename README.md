# Replication package: The AI Era's Lines of Code? A Case Study on What an Engineer's Token Spend Measures

This repository holds the analysis code behind the paper's numbers and figures. It contains
no data. The study uses a company's pseudonymized telemetry (coding-agent calls, commits,
pull requests, issues) and an internal survey; neither can be shared. Anonymized data may be
made available to researchers under a non-disclosure agreement (see the paper's Data
availability section).

The scripts read an anonymized extract in which every company-specific label is replaced:

| in the extract | stands for |
|---|---|
| `Division NN`, `Ecosystem NN`, `Team NN` | the team tree; numbered so the paths sort as the real names do, which keeps bootstrap draws and dummy coding unchanged |
| `role`, `seniority`, `family` | coded from the job title before release; titles are not in the extract |
| `agent_a`, `agent_b`, ... | coding agents, ordered by tokens (Agent A and Agent B in the paper) |
| `model_NN`, `model_vendor`, `model_tier` | models, ordered by tokens; `model_02` is the frontier model first seen inside the window |
| `agent_route_NN`, `route_NN` | gateway routes (feature ids), coding-agent routes and others |
| `internal` (commit `source`) | the company's internal code host, mirrored on GitHub |
| `Agent A`, `Agent B`, `Tool NN`, `Vendor A`... | survey tools and vendors; `Tool type` gives the kind of tool |

## Running

    TOKEN_SPEND_DATA=/path/to/extract bash run_all.sh

Numbers go to `out/` (JSON, CSV and logs), figures to `figs/`. A full run takes about an hour.
The extract directory holds:

    people.csv      user_id, team, role, seniority, family, is_manager, commits (binned)
    usages.csv      one row per agent call: user_id, event_date, event_timestamp, feature_id,
                    spent, charged, additional_data_tokens(_context, _generation),
                    additional_data_model_name, model_vendor, model_tier, tool, tool_run_type
    commits.csv     user_id, source, product_tag, date (author date), additions, deletions
                    (binned), issue_ids
    reviews.csv     pull requests: user_id, roles, created, toiw, size bins, issue_ids, ...
    issues.csv      user_id, id, type, state, toiw
    survey/survey_responses.csv   one row per tool evaluation
    survey/non_users.csv          respondents who use no company-paid AI tool
    survey/survey_open_text.txt   free-text answers tagged [QUESTION|ID-AGENT], for the coding

Environment of the verified run: Python 3.10 with pandas 2.3.3, numpy 2.2.6, scipy 1.15.3,
statsmodels 0.15.0, pyfixest 0.60.0, matplotlib 3.10.9, inequality 1.1.1, pingouin 0.6.1;
R 4.4.0 with fixest 0.13.2, glmmTMB 1.1.14, data.table 1.17.8, ineq 0.2.13, did 2.1.2,
didimputation 0.3.0, ritest 0.1.0 (GitHub grantmcdermott/ritest), sandwich, lmtest, car, cobalt.
Pin `did` and `didimputation` to these versions; later versions move some event-study estimates.

## Rules applied everywhere

- **Mirror duplicates removed.** A GitHub commit with an internal-host twin (same person,
  author date, size bins and issue ids), matched one-to-one, is dropped.
- **Window 2026-06-29 to 2026-08-02**, five full Monday weeks; the six-day, right-truncated
  week of 3-8 August is dropped. The seven weeks before serve as the pre-period.
- **A unit (day, week, person) is dropped only if it has zero metered tokens and zero
  commits.** Raw commit counts, no daily cap.
- **Survey dropdown ceilings are capped, not dropped** ("more than 30 hours" counts as 30).

## Where each result comes from

| paper | script | output |
|---|---|---|
| Setting and data, the telemetry panel | `companion/build_dataset.py`, `paper_extras.py` | `out/companion/people.csv`, `person_week.csv`, `out/extras.json` |
| RQ1: coupling of spend and commits (within and between engineers, lines, bands, time units, lead-lag, instrument) | `companion/analyze_panel.py`, `paper_extras.py`, `telemetry/models.R`, `telemetry/stage_part8.R` | `out/companion/results_panel.json` (`rq3`), `out/extras.json`, `out/telemetry/results*.json` |
| RQ1: how much of weekly spend output and engineer characteristics explain | `companion/analyze_panel.py`, `telemetry/stage_part10.R` | `results_panel.json` (`rq4`), `results_part10.json` |
| RQ1: stability of spend across weeks | `telemetry/stage_part9.R`, `paper_extras.py` | `results_part9.json`, `extras.json` |
| RQ1: teams | `team_level.py` | `out/team_level.json` |
| RQ2: spend not tied to output, floor and curve estimates, by group | `companion/analyze_panel.py`, `paper_extras.py`, `telemetry/models.R`, `telemetry/stage_part6.R`, `telemetry/stage_part7.R` | `results_panel.json` (`rq1`, `rq2`), `extras.json`, `results*.json` |
| RQ2: survey, what engineers use the agents for | `survey/survey_numbers.py`, `survey/survey_facts.py`, `companion/survey_crossq.py`, `paper_extras.py` | `out/survey/*.json`, `out/companion/survey_crossq.csv` |
| RQ2: thematic coding of the free-text answers | `companion/survey_thematic.py` (codes per answer, codebooks) | `out/companion/survey_thematic.log`, `survey_thematic_codes.csv`, `survey_thematic_codebook.csv` |
| RQ2: model of reported hours saved | `companion/paper_figs_survey_model.py` | `out/companion/paper_figs_survey_model.log` |

| figure | script |
|---|---|
| `fig0-spend-distribution`, `fig1-two-slopes`, `fig3-dark-spend`, `fig4a-spread`, `fig4b-explained`, `fig5-stability`, `fig7-lag`, `fig8-dark-by-group` | `companion/paper_figs.py` |
| `fig9a-activities`, `fig9b-effects` | `companion/paper_figs_survey.py` |
| `fig10-themes`, `fig10b-themes-rest` | `companion/survey_thematic.py` |
| `fig11-survey-model` | `companion/paper_figs_survey_model.py` |

`fig1b-pre-metering`, `fig6-output-change` and `fig11b-colleagues` are also written but are
not in the paper.

The headers of the telemetry scripts number research questions as in the study plan that
preceded the paper (RQ1 to RQ4, Parts 6 to 10); the tables above map them to the paper.
Some scripts compute more than the paper reports.

## Layout

    run_all.sh            runs everything in order
    lib/                  shared loading and windows (spend_common.py), figure style (paperstyle.py)
    telemetry/            person-day and person-week panels (panels.py); R models (models.R and
                          the stage scripts, which share common.R)
    companion/            person-week tables (build_dataset.py), the panel estimators
                          (analyze_panel.py, glm.py), the figures, the survey coding and model
    survey/               survey loading and coding (common.py) and the survey numbers
    paper_extras.py       numbers the paper quotes that no other script writes out
    team_level.py         team-level coupling with team-cluster bootstrap intervals
