# Part 6 of the token-study plan: the rest of RQ1 (token-study-loc.tex Sec. 4.2).
#
#   (a) Preparation or unrecovered cost: do heavy-use units with no artifact
#       precede more output than light-use units with no artifact? The plan's
#       lagged design, with the low-spend control for mean reversion. Weekly
#       (leads of 1 and 2 weeks; six weeks allow no more) and daily (the next
#       1, 3 and 5 present weekdays), the daily one within person.
#   (b) Trajectory: the plan's tenure test is not identified here (most users
#       start on the day logging starts); what is reported is how the floor
#       and the ceiling move over the six calendar weeks, descriptively.
#   (c) Step 3: how much of the between-person variance of the output-
#       independent component observable person characteristics account for,
#       from the correlated random effect (glmmTMB) with and without them; the
#       component's level by group; and the share of it that is common to a team.
#
#   Rscript telemetry/stage_part6.R     (writes results_part6.json)

HERE <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(HERE, "common.R"))
set.seed(20261006)
NBOOT <- as.integer(Sys.getenv("BOOT_R", "200"))
setFixest_nthreads(1)
S6 <- list()
CTL <- glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4), parallel = 1L)   # one thread: reproducible fits

# ---------------------------------------------------------------- (a) preparation test
prep_fit <- function(X, out) {
  m <- fepois(as.formula(paste(out, "~ high + lpmean | win")), X, cluster = ~user_id)
  c(one(m, "highTRUE"), list(mean_high = X[high == TRUE, mean(get(out))], mean_low = X[high == FALSE, mean(get(out))],
                             n_high = X[high == TRUE, .N], n_low = X[high == FALSE, .N]))
}
# weekly: units with no commit (output is commits only); high = tokens above the median of nonzero weeks
X <- window_panel(1L)[, tok := tokens / 1e6]
X[, artifact := commits > 0]
cut_w <- median(X[tok > 0, tok])
# the person's usual output before metering (weeks 0-6), the plan's between-person
# control for mean reversion; it contains no link-window outcome
X <- merge(X, W[user == TRUE & week < 7L, .(lpmean = log1p(mean(commits))), by = user_id], by = "user_id", all.x = TRUE)
for (k in 1:2) {
  L <- X[, .(user_id, win = win - k, lead = commits)]
  setnames(L, "lead", paste0("c_lead", k))
  X <- merge(X, L, by = c("user_id", "win"), all.x = TRUE)
}
Z <- X[artifact == FALSE][, high := tok > cut_w]
S6$prep_week <- list(cut = cut_w,
  lead1 = prep_fit(Z[!is.na(c_lead1)], "c_lead1"), lead2 = prep_fit(Z[!is.na(c_lead2)], "c_lead2"),
  lead1_working = prep_fit(Z[!is.na(c_lead1) & present >= 4], "c_lead1"))
# daily: weekdays outside inactive spells with no commit; leads over each person's present weekdays
Dd <- merge(copy(DD), RV[created >= as.IDate("2026-06-29") & created <= as.IDate(LINK_END),
                        .(prs = sum(author), revs = sum(reviewer)), by = .(user_id, date = created)],
            by = c("user_id", "date"), all.x = TRUE)
Dd[is.na(prs), prs := 0][is.na(revs), revs := 0]
Dd[, tok := tokens / 1e6]
setorder(Dd, user_id, date)
Dd[, `:=`(n1 = shift(commits, -1), n3 = frollsum(shift(commits, -1), 3, align = "left"),
          n5 = frollsum(shift(commits, -1), 5, align = "left")), by = user_id]
cut_d <- median(Dd[tok > 0, tok])
ZD <- Dd[commits_raw == 0][, high := tok > cut_d]
pd <- function(out) {
  Y <- ZD[!is.na(get(out))]
  m <- fepois(as.formula(paste(out, "~ high | user_id + date")), Y, cluster = ~user_id)
  c(one(m, "highTRUE"), list(mean_high = Y[high == TRUE, mean(get(out))], mean_low = Y[high == FALSE, mean(get(out))],
                             n_high = Y[high == TRUE, .N], n_low = Y[high == FALSE, .N]))
}
S6$prep_day <- list(cut = cut_d, next1 = pd("n1"), next3 = pd("n3"), next5 = pd("n5"))

# ---------------------------------------------------------------- (b) the bounds over the window
X1 <- window_panel(1L)[, tok := tokens / 1e6]
kn <- form_knots(X1)
m <- fit_form(X1, SETS$Bt, "log1p", kn, "win")
pp <- pred_pair(m, X1, SETS$Bt, "log1p", kn)
byw <- data.table(win = pp$U$win, p0 = pp$p0, p1 = pp$p1)[, .(ceiling = sum(p0) / sum(p1)), by = win]
X1[, artifact := commits > 0 | prs > 0 | revs > 0]
cut1 <- median(X1[tok > 0, tok])
flw <- X1[, .(floor = sum(tok[tok > cut1 & !artifact]) / sum(tok), tokens = sum(tok)), by = win]
S6$by_week <- as.list(merge(byw, flw, by = "win")[order(win)])
FU <- P[user == TRUE, .(first_use)]
S6$onset <- list(users = nrow(FU), by_first_link_day = FU[first_use <= as.IDate("2026-06-29"), .N],
                 late = P[user == TRUE & late == TRUE, .N])

# ---------------------------------------------------------------- (c) the person term
# person characteristics: job family, manager, division (the three largest, other), pre-period output and mix
TOP3 <- P[, .N, by = org_l2][order(-N)][1:3, org_l2]      # the three largest divisions
PZ <- P[user == TRUE, .(user_id, family = relevel(factor(as.character(family)), ref = "Developer"),
                        manager = as.integer(is_manager),
                        org = relevel(factor(fifelse(org_l2 %in% TOP3, org_l2, "other")), ref = TOP3[1]),
                        team, lpre, pre_direct_share, pre_linked_share, pre_large_share)]
F <- add_form(X1, SETS$Bt, "log1p", kn)
XC <- merge(F$X, PZ, by = "user_id")
for (v in F$vars) XC[, (paste0("mb_", v)) := mean(get(v)), by = user_id]
ZV <- c("family", "manager", "org", "lpre", "pre_direct_share", "pre_linked_share", "pre_large_share")
base_rhs <- c(F$vars, paste0("mb_", F$vars), "factor(win)")
fit_c <- function(D, z, fam = poisson, re = "(1 | user_id)")
  quiet_int(glmmTMB(as.formula(paste("tok ~", paste(c(base_rhs, z), collapse = " + "), "+", re)), data = D, family = fam, control = CTL))
var_share <- function(D) {
  s0 <- VarCorr(fit_c(D, character(0)))$cond$user_id[1]
  s1 <- VarCorr(fit_c(D, ZV))$cond$user_id[1]
  1 - s1 / s0
}
m0 <- fit_c(XC, character(0)); m1 <- fit_c(XC, ZV)
s0 <- VarCorr(m0)$cond$user_id[1]; s1 <- VarCorr(m1)$cond$user_id[1]
cf <- summary(m1)$coefficients$cond
zrows <- grep("^(family|manager|org|lpre|pre_)", rownames(cf), value = TRUE)
bs <- boot_share(XC, NBOOT, var_share)
S6$decomp <- c(list(sigma2_without = s0, sigma2_with = s1, share = 1 - s1 / s0,
                    converged = c(m0$fit$convergence == 0, m1$fit$convergence == 0),
                    pdHess = c(isTRUE(m0$sdr$pdHess), isTRUE(m1$sdr$pdHess)),
                    coef = lapply(zrows, function(r) list(term = r, b = cf[r, 1], se = cf[r, 2], p = cf[r, 4])),
                    people = uniqueN(XC$user_id)), bs)
# the same with negative binomial, as the second dispersion specification
S6$decomp_nb <- list(share = 1 - VarCorr(fit_c(XC, ZV, nbinom2))$cond$user_id[1] / VarCorr(fit_c(XC, character(0), nbinom2))$cond$user_id[1])
# how much of the person term is common to a team: team as a second random effect
mt <- fit_c(XC, character(0), re = "(1 | team) + (1 | user_id)")
vt <- VarCorr(mt)$cond
S6$team <- list(sigma2_team = vt$team[1], sigma2_person = vt$user_id[1], team_share = vt$team[1] / (vt$team[1] + vt$user_id[1]),
                teams = uniqueN(XC$team), converged = mt$fit$convergence == 0, pdHess = isTRUE(mt$sdr$pdHess))
# the component's level by group (population summaries): per person, predicted tokens at zero output / predicted tokens
pc <- data.table(user_id = pp$U$user_id, p0 = pp$p0, p1 = pp$p1)[, .(p0 = sum(p0), p1 = sum(p1)), by = user_id]
pc <- merge(pc, PZ[, .(user_id, family, manager)], by = "user_id")
S6$by_group <- list(
  family = as.list(pc[, .(people = .N, median_share = median(p0 / p1), pooled_share = sum(p0) / sum(p1)), by = family][order(family)]),
  manager = as.list(pc[, .(people = .N, median_share = median(p0 / p1), pooled_share = sum(p0) / sum(p1)), by = manager][order(manager)]))

write_results(list(part6 = S6), "results_part6.json")
