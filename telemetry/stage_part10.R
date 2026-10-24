# Part 10 of the token-study plan: RQ4, do conventional productivity metrics
# already explain consumption? (token-study-loc.tex Sec. 4.5)
#
#   The baseline with s(output) replaced by a conventional-metric profile,
#   The profile the extract supports for every engineer:
#     quality  - commits naming a Bug-type issue (the only quality signal
#                without reverts or issue dates)
#     activity - commits
#   GitHub PR terms (authored, time open, reviewed) were dropped 2026-10-23:
#   they exist for the minority of engineers on GitHub and are zero by
#   construction for the rest. Of DORA's four keys none has a counterpart.
#   The statistic is the *incremental* share of Poisson deviance explained by
#   adding the profile to a null that holds the terms that version keeps:
#     within  - person and week effects; folds hold out person-weeks
#     between - week effects only;     folds hold out whole people
#   reported out of sample (5 folds) and in sample. Also: the non-output
#   dimensions added to the RQ1 baseline (what the frameworks add beyond
#   counting artifacts), and the between version with pre-period work mix
#   and person characteristics (the plan's case-mix adjustment).
#
#   Rscript telemetry/stage_part10.R    (writes results_part10.json)

HERE <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(HERE, "common.R"))
set.seed(20261010)
S10 <- list()

X <- window_panel(1L)[, tok := tokens / 1e6]
RVa <- RV[author == TRUE & created >= as.IDate("2026-06-29") & created <= as.IDate(LINK_END)]
RVa[, win := as.integer((created - as.IDate("2026-05-11")) %/% 7) - 7L]
X <- merge(X, RVa[, .(toiw_mean = mean(toiw)), by = .(user_id, win)], by = c("user_id", "win"), all.x = TRUE)
X[, has_pr := as.integer(prs > 0)][is.na(toiw_mean), toiw_mean := 0]
X[, `:=`(l_prs = log1p(prs), l_toiw = log1p(toiw_mean), l_bug = log1p(bug), l_commits = log1p(commits), l_revs = log1p(revs),
         bugshare = fifelse(commits > 0, bug / commits, 0))]
X <- merge(X, P[, .(user_id, family = as.character(family), is_manager, org_l2, lpre, pre_direct_share, pre_linked_share, pre_large_share)], by = "user_id")
TOP3 <- P[, .N, by = org_l2][order(-N)][1:3, org_l2]      # the three largest divisions
X[, org := fifelse(org_l2 %in% TOP3, org_l2, "other")]
kn <- form_knots(X)
BASE <- add_form(X, SETS$Bt, "log1p", kn)$vars        # RQ1's output terms, log(1 + x)
PROFILE <- c("l_bug", "l_commits")
NONOUT <- c("bugshare")
CASEMIX <- c("lpre", "pre_direct_share", "pre_linked_share", "pre_large_share", "i(family)", "is_manager", "i(org)")
XF <- add_form(X, SETS$Bt, "log1p", kn)$X
dev <- function(y, mu) sum(poisson()$dev.resids(y, mu, rep(1, length(y))))
fml <- function(rhs, fe) as.formula(paste("tok ~", if (length(rhs)) paste(rhs, collapse = " + ") else "1", "|", fe))
fit <- function(D, rhs, fe) fepois(fml(rhs, fe), D)
# deviance share, in and out of sample; folds over rows (within) or over people (between)
dev_share <- function(rhs_null, rhs_model, fe, by_person, K = 5) {
  rhs_model <- c(rhs_null, rhs_model)                 # the model nests the null: the share is an increment
  m0 <- fit(XF, rhs_null, fe); m1 <- fit(XF, rhs_model, fe)
  ins <- 1 - deviance(m1) / deviance(m0)
  fold <- if (by_person) { ids <- unique(XF$user_id); f <- sample(rep(1:K, length.out = length(ids))); f[match(XF$user_id, ids)] } else sample(rep(1:K, length.out = nrow(XF)))
  d0 <- 0; d1 <- 0; nn <- 0
  for (k in 1:K) {
    tr <- XF[fold != k]; te <- XF[fold == k]
    a0 <- fit(tr, rhs_null, fe); a1 <- fit(tr, rhs_model, fe)
    p0 <- predict(a0, newdata = te); p1 <- predict(a1, newdata = te)
    ok <- is.finite(p0) & is.finite(p1)
    d0 <- d0 + dev(te$tok[ok], p0[ok]); d1 <- d1 + dev(te$tok[ok], p1[ok]); nn <- nn + sum(ok)
  }
  list(in_sample = ins, out_of_sample = 1 - d1 / d0, held_out_units = nn, k = K,
       terms = tidy_fx(fepois(fml(rhs_model, fe), XF, cluster = ~user_id))$coef)
}
# within person: person and week effects retained
S10$within <- list(
  profile = dev_share(character(0), PROFILE, "user_id + win", FALSE),
  output  = dev_share(character(0), BASE, "user_id + win", FALSE),
  output_plus_nonoutput = dev_share(BASE, NONOUT, "user_id + win", FALSE),
  output_plus_profile = dev_share(BASE, PROFILE, "user_id + win", FALSE))
# between people: week effects only; folds hold out whole people
S10$between <- list(
  profile = dev_share(character(0), PROFILE, "win", TRUE),
  output  = dev_share(character(0), BASE, "win", TRUE),
  profile_plus_casemix = dev_share(PROFILE, CASEMIX, "win", TRUE),
  casemix = dev_share(character(0), CASEMIX, "win", TRUE),
  person_means = { # the ceiling on any between-person profile: how much do person identities explain?
    m0 <- fit(XF, character(0), "win"); m1 <- fit(XF, character(0), "user_id + win"); 1 - deviance(m1) / deviance(m0) })
# the spread the profile leaves: 90/10 ratio of person-mean tokens, raw and after the between-person profile
m_b <- fit(XF, PROFILE, "win")
PMp <- data.table(user_id = XF$user_id[obs(m_b)], tok = XF$tok[obs(m_b)], fit = predict(m_b))[, .(tok = mean(tok), fit = mean(fit)), by = user_id]
q <- function(v) unname(quantile(v, .9) / quantile(v, .1))
S10$spread <- list(people = nrow(PMp), raw_90_10 = q(PMp$tok), ratio_to_fitted_90_10 = q(PMp$tok / PMp$fit),
                   sd_log_raw = sd(log(PMp$tok[PMp$tok > 0])), sd_log_ratio = sd(log((PMp$tok / PMp$fit)[PMp$tok > 0])))
S10$profile <- list(quality = "l_bug", activity = "l_commits",
                    not_available = c("deployment frequency", "change-failure rate", "time to restore", "revert rate", "issues created or resolved", "review comments by week", "GitHub PR terms (minority coverage)"))
write_results(list(part10 = S10), "results_part10.json")
