# Part 7 of the token-study plan: RQ2, the consumption that does not convert
# (token-study-loc.tex Sec. 4.3), the telemetry half.
#
#   The quantity is the residual of the primary RQ1 model (tokens, log(1 + x)
#   form, size + kind + work type, weekly, person and week fixed effects):
#   observed minus fitted tokens, i.e. what a person's week consumed beyond what
#   that person's output explains. It is scaled by the person's mean fitted
#   tokens, so a value of 1 means "one typical week's worth of unexplained
#   consumption", and a *high-residual week* is the top third of a person's own
#   weeks (people with at least three estimation weeks).
#
#   (a) What it predicts: rework signals the following week. The plan's
#       preferred signals (reverts, repeated edits of the same files) are not
#       in the extract; what the records support is commits that name a
#       Bug-type issue, and review friction on the person's GitHub PRs opened
#       the following week (reviewers, time open). Measured at t+1 because a
#       rework signal is itself an artifact and would lower the same week's
#       residual mechanically.
#   (b) What mode it took: how a person's high-residual weeks differ from their
#       other weeks in the shape of use. This telemetry is agent-only (agent
#       runs carry 99.9% of tokens), so the plan's agent / chat / completion
#       split collapses; the modes left are generation share (writing- vs
#       reading-heavy), tokens per request, requests per active day, remote
#       runs, vendor, session length and off-hours use.
#
#   Rscript telemetry/stage_part7.R     (writes results_part7.json)

HERE <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(HERE, "common.R"))
set.seed(20261007)
S7 <- list()

# ---------------------------------------------------------------- the residual
X <- window_panel(1L)[, tok := tokens / 1e6]
kn <- form_knots(X)
m <- fit_form(X, SETS$Bt, "log1p", kn, "win")
pp <- pred_pair(m, X, SETS$Bt, "log1p", kn)
U <- copy(pp$U)[, fitted := pp$p1][, resid := tok - fitted]
U[, rel := resid / mean(fitted), by = user_id]
U[, n_est := .N, by = user_id]
U[, high := NA]
U[n_est >= 3, high := rel >= quantile(rel, 2 / 3), by = user_id]
U[, lratio := fifelse(tok > 0, log(tok / fitted), NA_real_)]
S7$residual <- list(weeks = nrow(U), people = uniqueN(U$user_id), high_weeks = U[high == TRUE, .N],
                    rel_q = unname(quantile(U$rel, c(.1, .25, .5, .75, .9))), sd_rel = sd(U$rel),
                    share_abs_resid = U[, sum(abs(resid))] / U[, sum(tok)],
                    within_person_share = {   # of the variance of the relative residual, how much is within person
                      v_tot <- var(U$rel); v_between <- var(U[, mean(rel), by = user_id]$V1); 1 - v_between / v_tot })

# ---------------------------------------------------------------- (a) next-week rework
RVa <- RV[author == TRUE & created >= as.IDate("2026-06-29") & created <= as.IDate(LINK_END)]
RVa[, win := as.integer((created - as.IDate("2026-05-11")) %/% 7) - 7L]
prf <- RVa[, .(prs_n = .N, toiw_mean = mean(toiw), nrev_mean = mean(n_reviewers_in_window)), by = .(user_id, win)]
NX <- X[, .(user_id, win = win - 1L, bug_next = bug, commits_next = commits, linked_next = linked, present_next = present)]
U <- merge(U, NX, by = c("user_id", "win"), all.x = TRUE)
U <- merge(U, prf[, .(user_id, win = win - 1L, prs_next = prs_n, toiw_next = toiw_mean, nrev_next = nrev_mean)],
           by = c("user_id", "win"), all.x = TRUE)
UN <- U[!is.na(bug_next)]
sp <- function(x, y) { ok <- is.finite(x) & is.finite(y); ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman")); list(rho = unname(ct$estimate), p = ct$p.value, n = sum(ok)) }
S7$rework <- list(
  n = nrow(UN),
  spearman_bug = sp(UN$rel, UN$bug_next),
  spearman_bugshare = with(UN[commits_next > 0], sp(rel, bug_next / commits_next)),
  spearman_commits = sp(UN$rel, UN$commits_next),
  # Poisson, person and week effects, next week's total output held fixed: does the residual predict *more bug work*?
  bug_rel = one(fepois(bug_next ~ rel + log1p(commits_next) | user_id + win, UN, cluster = ~user_id), "rel"),
  bug_high = one(fepois(bug_next ~ high + log1p(commits_next) | user_id + win, UN[!is.na(high)], cluster = ~user_id), "highTRUE"),
  bug_rel_nocontrol = one(fepois(bug_next ~ rel | user_id + win, UN, cluster = ~user_id), "rel"),
  # the same residual against next week's output itself (the preparation reading)
  commits_rel = one(fepois(commits_next ~ rel | user_id + win, UN, cluster = ~user_id), "rel"),
  commits_high = one(fepois(commits_next ~ high | user_id + win, UN[!is.na(high)], cluster = ~user_id), "highTRUE"),
  # review friction on PRs opened next week (GitHub authors only)
  prs_next_rows = UN[!is.na(prs_next), .N], prs_next_people = UN[!is.na(prs_next), uniqueN(user_id)],
  toiw_rel = one(feols(log1p(toiw_next) ~ rel | user_id + win, UN[!is.na(prs_next)], cluster = ~user_id), "rel"),
  nrev_rel = one(feols(nrev_next ~ rel | user_id + win, UN[!is.na(prs_next)], cluster = ~user_id), "rel"),
  means = list(bug_high = UN[high == TRUE, mean(bug_next)], bug_other = UN[high == FALSE, mean(bug_next)],
               bugshare_high = UN[high == TRUE & commits_next > 0, mean(bug_next / commits_next)],
               bugshare_other = UN[high == FALSE & commits_next > 0, mean(bug_next / commits_next)]))

# ---------------------------------------------------------------- (b) mode of use
UM <- U[tokens > 0]
UM[, `:=`(gen_share = tokens_out / tokens, ltpe = log(tokens / events), lepd = log(events / ai_days),
          remote_share = events_remote / events, agent_b_share = tokens_agent_b / tokens, off_share = events_off / events,
          leps = log(events / pmax(sessions, 1)), long_share = events_long / events)]
MODES <- c(gen_share = "generation share of tokens", ltpe = "log tokens per request", lepd = "log requests per active day",
           remote_share = "share of requests run remotely", agent_b_share = "share of tokens on Agent B",
           off_share = "share of requests off-hours", leps = "log requests per session", long_share = "share of requests in sessions over 2 h")
S7$modes <- lapply(names(MODES), function(v) {
  f_rel <- as.formula(paste(v, "~ rel | user_id + win")); f_hi <- as.formula(paste(v, "~ high | user_id + win"))
  D <- UM[is.finite(get(v))]
  within_sd <- D[, sd(get(v) - mean(get(v))), by = user_id][, sqrt(mean(V1^2, na.rm = TRUE))]
  list(var = v, label = MODES[[v]], mean = D[, mean(get(v))], within_sd = within_sd,
       rel = one(feols(f_rel, D, cluster = ~user_id), "rel"),
       high = one(feols(f_hi, D[!is.na(high)], cluster = ~user_id), "highTRUE"),
       mean_high = D[high == TRUE, mean(get(v))], mean_other = D[high == FALSE, mean(get(v))])
})
S7$modes_bh <- as.list(setNames(p.adjust(sapply(S7$modes, function(z) z$high$p), "BH"), names(MODES)))
S7$overall <- list(gen_share = UM[, sum(tokens_out) / sum(tokens)], remote_share = UM[, sum(events_remote) / sum(events)],
                   off_share = UM[, sum(events_off) / sum(events)], long_share = UM[, sum(events_long) / sum(events)],
                   events_per_session = UM[, sum(events) / sum(sessions)], tokens_per_request = UM[, sum(tokens) / sum(events)])

write_results(list(part7 = S7), "results_part7.json")
