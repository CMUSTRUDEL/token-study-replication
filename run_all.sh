#!/usr/bin/env bash
# Reproduce the numbers and figures of the paper from the anonymized extract.
#
# Rules, hard-wired as defaults in every script:
#   * mirror duplicates removed: a GitHub commit with an internal-host twin (same person,
#     author date, size bins, issue ids), matched one-to-one, is dropped;
#   * analysis window 2026-06-29 .. 2026-08-02 (five full Monday weeks; the six-day,
#     right-truncated week of Aug 3-8 is dropped);
#   * a unit (person-day, person-week, block, person) is dropped only if it has zero metered
#     tokens and zero commits.
#
# Input: the extract in $TOKEN_SPEND_DATA (default ./data; see README.md for the layout).
# Outputs: out/ (numbers) and figs/ (figures).
#   bash run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW="$(cd "${TOKEN_SPEND_DATA:-$HERE/data}" && pwd)"; export TOKEN_SPEND_DATA="$RAW"
OUT="$HERE/out"; FIGS="$HERE/figs"
mkdir -p "$OUT/telemetry" "$OUT/survey" "$OUT/companion" "$FIGS"
export R_PROFILE_USER=/dev/null MPLBACKEND=Agg
PY="${PYTHON:-python3}"

echo "== 1. telemetry panels and models"
"$PY" "$HERE/telemetry/panels.py"
Rscript "$HERE/telemetry/models.R"
for st in "$HERE"/telemetry/stage_*.R; do Rscript "$st"; done

echo "== 2. companion pipeline (person-week panel, figures)"
W="$OUT/companion/work"; rm -rf "$W"; mkdir -p "$W/analysis/out"
ln -sfn "$RAW" "$W/data"
ln -sfn "$FIGS" "$W/figs"
cp "$HERE/companion/survey_crossq.py" "$W/analysis/"
( cd "$W"
  "$PY" "$HERE/companion/build_dataset.py"
  "$PY" "$HERE/companion/analyze_panel.py" > "$OUT/companion/analyze_panel.log"
  "$PY" "$HERE/companion/paper_figs.py"       > "$OUT/companion/paper_figs.log"
  "$PY" "$HERE/companion/survey_thematic.py"  > "$OUT/companion/survey_thematic.log"
  "$PY" "$HERE/companion/paper_figs_survey.py"
  "$PY" "$HERE/companion/survey_crossq.py"    > "$OUT/companion/survey_crossq.log"
  "$PY" "$HERE/companion/paper_figs_survey_model.py" > "$OUT/companion/paper_figs_survey_model.log" )
cp "$W"/analysis/out/*.json "$W"/analysis/out/*.csv "$OUT/companion/" 2>/dev/null || true

echo "== 3. survey"
"$PY" "$HERE/survey/survey_numbers.py"
"$PY" "$HERE/survey/survey_facts.py"

echo "== 4. remaining numbers"
"$PY" "$HERE/paper_extras.py"
"$PY" "$HERE/team_level.py"
