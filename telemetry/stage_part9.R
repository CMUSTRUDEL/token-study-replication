# Part 9 of the token-study plan: RQ3 (b), stability of conversion
# (token-study-loc.tex Sec. 4.4, the stability component), on tokens.
#
#   Two person-level quantities from the returns model
#     E[tokens_it] = exp(a_i + d_i log(commits_it) + t_t),  commits > 0:
#   the level a_i (consumption at one commit, given the week) and the slope
#   d_i (the person's own elasticity). Split-half reliability: each fitted on
#   odd and on even link weeks, correlated across people present in both, with
#   the Spearman-Brown correction for a half-length measurement. The plan's
#   gate: the check on d_i needs enough positive-output weeks per person
#   (roughly 20 in the plan); the records supply at most 6, so the slope
#   result is reported against the gate. Also: the raw tokens-per-commit
#   ratio (the dashboard version), the between-person share of variance in
#   consumption given output (the plan's ICC), and the rank correlation of
#   RQ3's a_i with RQ1's c_i, which the plan asks for since the two person
#   intercepts are not the same quantity.
#
#   Rscript telemetry/stage_part9.R     (writes results_part9.json)

HERE <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(HERE, "common.R"))
set.seed(20261009)
S9 <- list()
sb <- function(r) 2 * r / (1 + r)                      # Spearman-Brown, two halves

X <- window_panel(1L)[, tok := tokens / 1e6]
Xp <- X[commits > 0][, lcm := log(commits)][, half := fifelse(win %% 2L == 0L, "even", "odd")]
# weeks per person available to the split
wk <- Xp[, .(weeks = .N, odd = sum(half == "odd"), even = sum(half == "even"),
             var_odd = uniqueN(commits[half == "odd"]) >= 2, var_even = uniqueN(commits[half == "even"]) >= 2), by = user_id]
S9$gate <- list(people = nrow(wk), weeks_median = median(wk$weeks), weeks_dist = as.list(wk[, .N, by = weeks][order(weeks)]),
                both_halves = wk[odd > 0 & even > 0, .N], both_halves_2plus = wk[odd >= 2 & even >= 2, .N],
                slope_estimable = wk[odd >= 2 & even >= 2 & var_odd & var_even, .N], plan_needs = 20)

# ---------------------------------------------------------------- the level a_i
lev <- lapply(c("odd", "even"), function(h) {
  m <- fepois(tok ~ lcm | user_id + win, Xp[half == h])
  fe <- fixef(m)$user_id
  data.table(user_id = names(fe), a = unname(fe))
})
L <- merge(lev[[1]], lev[[2]], by = "user_id", suffixes = c("_odd", "_even"))
cs <- cor.test(L$a_odd, L$a_even, method = "spearman", exact = FALSE); cp <- cor.test(L$a_odd, L$a_even)
S9$level <- list(people = nrow(L), spearman = unname(cs$estimate), spearman_p = cs$p.value, spearman_sb = sb(unname(cs$estimate)),
                 pearson = unname(cp$estimate), pearson_ci = unname(cp$conf.int), pearson_sb = sb(unname(cp$estimate)))
# the dashboard version: raw tokens per commit per half
R2 <- Xp[, .(ratio = sum(tok) / sum(commits)), by = .(user_id, half)]
R2 <- dcast(R2, user_id ~ half, value.var = "ratio")[!is.na(odd) & !is.na(even)]
cr <- cor.test(R2$odd, R2$even, method = "spearman", exact = FALSE)
S9$ratio <- list(people = nrow(R2), spearman = unname(cr$estimate), spearman_sb = sb(unname(cr$estimate)), p = cr$p.value)

# ---------------------------------------------------------------- the slope d_i
ok_ids <- wk[odd >= 2 & even >= 2 & var_odd & var_even, user_id]
slp <- lapply(c("odd", "even"), function(h) {
  D <- Xp[half == h & user_id %in% ok_ids]
  m <- fepois(tok ~ 1 | user_id[lcm] + win, D)
  s <- fixef(m)[["user_id[[lcm]]"]]
  data.table(user_id = names(s), d = unname(s))
})
Sd <- merge(slp[[1]], slp[[2]], by = "user_id", suffixes = c("_odd", "_even"))
Sd <- Sd[is.finite(d_odd) & is.finite(d_even)]
cd <- cor.test(Sd$d_odd, Sd$d_even, method = "spearman", exact = FALSE)
S9$slope <- list(people = nrow(Sd), spearman = unname(cd$estimate), spearman_p = cd$p.value, spearman_sb = sb(unname(cd$estimate)),
                 sd_odd = sd(Sd$d_odd), sd_even = sd(Sd$d_even), median_odd = median(Sd$d_odd), median_even = median(Sd$d_even),
                 iqr_odd = unname(quantile(Sd$d_odd, c(.25, .75))), iqr_even = unname(quantile(Sd$d_even, c(.25, .75))))

# ---------------------------------------------------------------- between vs within (the plan's ICC)
# log tokens on positive-token weeks, after output and week: how much of the remaining variance is the person?
Xl <- Xp[tokens > 0]
m_full <- feols(log(tok) ~ lcm | user_id + win, Xl)
m_nofe <- feols(log(tok) ~ lcm | win, Xl)
pe <- fixef(m_full)$user_id
S9$icc <- list(between = var(pe), within = var(resid(m_full)), share_between = var(pe) / (var(pe) + var(resid(m_full))),
               r2_within_person_added = 1 - var(resid(m_full)) / var(resid(m_nofe)), weeks = nrow(Xl))

# ---------------------------------------------------------------- RQ3's a_i against RQ1's c_i
kn <- form_knots(X)
m_rq1 <- fit_form(X, SETS$Bt, "log1p", kn, "win")
m_rq3 <- fepois(tok ~ lcm | user_id + win, Xp)
C1 <- data.table(user_id = names(fixef(m_rq1)$user_id), c = unname(fixef(m_rq1)$user_id))
A3 <- data.table(user_id = names(fixef(m_rq3)$user_id), a = unname(fixef(m_rq3)$user_id))
CA <- merge(C1, A3, by = "user_id")
cc <- cor.test(CA$c, CA$a, method = "spearman", exact = FALSE)
S9$c_vs_a <- list(people = nrow(CA), spearman = unname(cc$estimate), p = cc$p.value)

write_results(list(part9 = S9), "results_part9.json")
