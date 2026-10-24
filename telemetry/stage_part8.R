# Part 8 of the token-study plan: RQ3 (a), how consumption scales with output
# (token-study-loc.tex Sec. 4.4, the returns component), on tokens throughout.
#
#   E[tokens_it] = exp(a_i + d log(output_it) + t_t),   output_it > 0
#
#   d is the elasticity: percent change in tokens per percent change in
#   output; d = 1 is proportionality, the property cost-per-artifact ratios
#   assume. Fitted one output measure at a time: commits (best identified)
#   and GitHub PRs authored (the stand-in for merged PRs; scarce, so expected
#   to be attenuated more). Units of a day, 1, 2 and 4 weeks. Checks the plan
#   names: the restriction's cost (how many periods drop and whether they
#   differ in consumption), weeks dominated by a single agent (tooling), and
#   the ones carried over from claim 25 on credits: the layers of consumption,
#   drift over the window, the shape (commit bands), slopes by group and per
#   person. Zero-token periods stay in (Poisson handles them); the credit
#   results of claim 25 are the sensitivity.
#
#   Rscript telemetry/stage_part8.R     (writes results_part8.json)

HERE <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(HERE, "common.R"))
set.seed(20261008)
S8 <- list()
elas <- function(X, y, out, time) {
  m <- fepois(as.formula(paste(y, "~ log(", out, ") |", "user_id +", time)), X, cluster = ~user_id)
  c(one(m, paste0("log(", out, ")")), list(vs_one = test_equal(m, paste0("log(", out, ")"), 1)))
}

# ---------------------------------------------------------------- the elasticity, by unit, and what the restriction drops
for (g in c("day", "w1", "w2", "w4")) {
  X <- if (g == "day") copy(DD) else window_panel(as.integer(sub("w", "", g)))    # daily commits are the capped counts, as everywhere
  X[, tok := tokens / 1e6]
  time <- if (g == "day") "date" else "win"
  Xp <- X[commits > 0]
  S8$commits[[g]] <- c(elas(Xp, "tok", "commits", time),
    list(dropped = list(units = X[commits == 0, .N], of = nrow(X), token_share = X[commits == 0, sum(tok)] / X[, sum(tok)],
                        mean_tok_dropped = X[commits == 0, mean(tok)], mean_tok_kept = Xp[, mean(tok)])))
  if (g != "day") {
    Xq <- X[prs > 0]
    S8$prs[[g]] <- c(elas(Xq, "tok", "prs", time), list(units = nrow(Xq), people = uniqueN(Xq$user_id),
                                                        median_prs = median(Xq$prs), share_one = Xq[prs == 1, .N] / nrow(Xq)))
  }
}
# both measures together, weekly, on weeks with both (partial elasticities, for the collinearity note)
X1 <- window_panel(1L)[, tok := tokens / 1e6]
Xb <- X1[commits > 0 & prs > 0]
mb <- fepois(tok ~ log(commits) + log(prs) | user_id + win, Xb, cluster = ~user_id)
S8$joint <- list(fit = tidy_fx(mb), units = nrow(Xb), cor_logs = Xb[, cor(log(commits), log(prs))])

# ---------------------------------------------------------------- single-agent weeks (tooling)
X1[, agent_b_sh := fifelse(tokens > 0, tokens_agent_b / tokens, NA_real_)]
Xp1 <- X1[commits > 0]
S8$single_agent <- list(
  all = elas(Xp1, "tok", "commits", "win"),
  agent_a_only = c(elas(Xp1[agent_b_sh <= 0.1], "tok", "commits", "win"), list(units = Xp1[agent_b_sh <= 0.1, .N])),
  agent_b_only = c(elas(Xp1[agent_b_sh >= 0.9], "tok", "commits", "win"), list(units = Xp1[agent_b_sh >= 0.9, .N])),
  mixed = c(elas(Xp1[agent_b_sh > 0.1 & agent_b_sh < 0.9], "tok", "commits", "win"), list(units = Xp1[agent_b_sh > 0.1 & agent_b_sh < 0.9, .N])),
  share_single = Xp1[!is.na(agent_b_sh), mean(agent_b_sh <= 0.1 | agent_b_sh >= 0.9)])

# ---------------------------------------------------------------- layers, drift, shape, groups, persons (tokens)
Xl <- Xp1[tokens > 0]
S8$layers <- list(events = one(fepois(events ~ log(commits) | user_id + win, Xp1, cluster = ~user_id), "log(commits)"),
                  tokens_per_request = one(feols(log(tokens / events) ~ log(commits) | user_id + win, Xl, cluster = ~user_id), "log(commits)"),
                  gen_share = one(feols(log(tokens_out / tokens) ~ log(commits) | user_id + win, Xl[tokens_out > 0], cluster = ~user_id), "log(commits)"))
Xp1[, tc := win - mean(win)]
S8$drift <- tidy_fx(fepois(tok ~ log(commits) + log(commits):tc | user_id + win, Xp1, cluster = ~user_id), c("log(commits)", "log(commits):tc"))
Xp1[, lcm := log(commits)]
Xp1[, band := as.character(cut(commits, c(0, 1, 3, 7, 15, 31, Inf), labels = c("1", "2-3", "4-7", "8-15", "16-31", "32+")))]
mband <- fepois(tok ~ i(band, ref = "1") | user_id + win, Xp1, cluster = ~user_id)
mband_t <- fepois(tok ~ lcm + i(band, ref = "1") | user_id + win, Xp1, cluster = ~user_id)
S8$bands <- list(fit = tidy_fx(mband), counts = as.list(Xp1[, .N, by = band][order(band)]),
                 geo_mid = as.list(Xp1[, .(gm = exp(mean(lcm))), by = band][order(band)]), test = wald_zero(mband_t, "^band::"))
Xg <- merge(Xp1, P[, .(user_id, family, is_manager, tier, base_tercile)], by = "user_id")
Xg[, manager := fifelse(is_manager, "manager", "IC")][, tier := as.character(tier)]
grp <- list(); pv <- c()
for (gname in c("family", "manager", "tier", "base_tercile")) {
  ms <- fepois(as.formula(paste0("tok ~ lcm:i(", gname, ") | user_id + win")), Xg, cluster = ~user_id)
  mt <- fepois(as.formula(paste0("tok ~ lcm + lcm:i(", gname, ") | user_id + win")), Xg, cluster = ~user_id)
  w_ <- wald_zero(mt, paste0("^lcm:", gname, "::"))
  grp[[gname]] <- list(slopes = tidy_fx(ms)$coef, wald_p = w_$p, meta = meta_fx(ms))
  pv <- c(pv, setNames(w_$p, gname))
}
S8$groups <- grp; S8$groups_bh <- as.list(p.adjust(pv, "BH"))
Xs <- Xp1[, if (.N >= 4 && uniqueN(commits) >= 3) .SD, by = user_id]
mv <- fepois(tok ~ 1 | user_id[lcm] + win, Xs)
sl <- fixef(mv)[["user_id[[lcm]]"]]
S8$person_slopes <- list(people = length(sl), median = median(sl), q = unname(quantile(sl, c(.1, .25, .5, .75, .9))),
                         pooled_same_sample = unname(coef(fepois(tok ~ lcm | user_id + win, Xs))["lcm"]))

write_results(list(part8 = S8), "results_part8.json")
