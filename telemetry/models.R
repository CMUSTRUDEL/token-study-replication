# Models for the telemetry experiments. Reads the panels written by panels.py
# and writes every estimate to out/telemetry/results.json.
#
#   Rscript telemetry/models.R
#
# Conventions: Poisson with fixed effects (fixest::fepois) for counts, so zero
# weeks stay in; standard errors clustered by person throughout; coefficients
# are log points, reported in the report as exp(b) - 1.

# did, car, ritest, cobalt and ineq (and their dependencies) come from a
# library built from a dated CRAN snapshot matching this R (4.3.3); newer
# copies elsewhere on the path need R 4.4. See Appendix D of findings.md.
HERE <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(HERE, "common.R"))   # libraries, data, panels and all helpers

# ============================================================ A. who uses AI
m <- glm(user01 ~ family + is_manager + lpre + pre_direct_share + pre_github_share + div_top,
         data = P, family = binomial())
ct <- coeftest(m, vcov = vcovHC(m, type = "HC1"))
put("adopt_logit", list(coef = data.table(term = rownames(ct), b = ct[, 1], se = ct[, 2], p = ct[, 4]),
                        n = nobs(m), meta = list(formula = paste(deparse(formula(m)), collapse = " "),
                                                 method = "glm binomial (logit)", vcov = "HC1", n_input = nrow(P), n_used = nobs(m))))
m <- feols(log(spent) ~ family + is_manager + lpre + pre_direct_share + pre_github_share + div_top,
           data = P[user == TRUE], vcov = "hetero")
put("intensity_ols", tidy_fx(m))

# standardised mean differences (cobalt) and the Gini of link-window spend (ineq)
cov <- P[, .(lpre, pre_direct_share, pre_github_share, pre_linked_share,
             is_manager = as.numeric(is_manager), div_top = as.numeric(div_top))]
hv <- P$tier %in% c("heavy", "non-user")
put("smd", list(users = as.list(cobalt::col_w_smd(cov, treat = P$user01, std = TRUE, s.d.denom = "pooled")),
                heavy = as.list(cobalt::col_w_smd(cov[hv], treat = as.integer(P$tier[hv] == "heavy"), std = TRUE, s.d.denom = "pooled")),
                method = "cobalt::col_w_smd, pooled SD"))
put("gini_spend_ineq", ineq::Gini(P[user == TRUE, spent]))

# ================================================ B1. between-person ANCOVA
# link-window commits given pre-period commits: selection-on-observables only.
m1 <- fepois(link_commits ~ user01 + lpre + family + is_manager + div_top, data = P, vcov = "hetero")
m2 <- fepois(link_commits ~ tier + lpre + family + is_manager + div_top, data = P, vcov = "hetero")
m3 <- fepois(link_commit_days ~ tier + lpre + family + is_manager + div_top, data = P, vcov = "hetero")
m4 <- fepois(link_commits ~ tier + lpre + family + is_manager + div_top + log(link_present + 1),
             data = P, vcov = "hetero")
put("ancova_user", one(m1, "user01"))
put("ancova_tier", tidy_fx(m2, c("tierlight", "tiermedium", "tierheavy")))
put("ancova_tier_days", tidy_fx(m3, c("tierlight", "tiermedium", "tierheavy")))
put("ancova_tier_presence", tidy_fx(m4, c("tierlight", "tiermedium", "tierheavy")))
# the same comparison one step earlier: does link-window usage "predict" the
# pre-period, before any of it was logged? If heavy users were already
# different in May, the tier gap is not about July.
WE <- W[week <= 3, .(pre_a = sum(commits)), by = user_id]
WLt <- W[week %in% 4:6, .(pre_b = sum(commits)), by = user_id]
P <- merge(P, WE, by = "user_id"); P <- merge(P, WLt, by = "user_id")
m5 <- fepois(pre_b ~ tier + log1p(pre_a) + family + is_manager + div_top, data = P, vcov = "hetero")
put("ancova_tier_placebo_pre", tidy_fx(m5, c("tierlight", "tiermedium", "tierheavy")))

# ===================================== B2. within-person weekly dose-response
base <- fepois(commits ~ lspend | user_id + week, WL, cluster = ~user_id)
put("dr_base", one(base, "lspend"))

WLp <- WL[present > 0]
pres <- fepois(commits ~ lspend + log(present) | user_id + week, WLp, cluster = ~user_id)
put("dr_presence", one(pres, "lspend"))

# falsification: next week's spend should not predict this week's commits
ll <- fepois(commits ~ lspend + f(lspend, 1) + l(lspend, 1) + log(present) | user_id + week,
             WLp, panel.id = ~user_id + week, cluster = ~user_id)
put("dr_leadlag", tidy_fx(ll))
llb <- fepois(commits ~ lspend + f(lspend, 1) + l(lspend, 1) | user_id + week,
              WL, panel.id = ~user_id + week, cluster = ~user_id)
put("dr_leadlag_nopres", tidy_fx(llb))

# intensive margin only: weeks with some AI use; and the extensive margin
put("dr_intensive", one(fepois(commits ~ lspend + log(present) | user_id + week, WLp[spent > 0], cluster = ~user_id), "lspend"))
WLp[, any_ai := as.integer(events > 0)]
put("dr_extensive", one(fepois(commits ~ any_ai + log(present) | user_id + week, WLp, cluster = ~user_id), "any_ai"))
# accumulated use (the mandate paper's channel): spend through the previous week
setorder(WL, user_id, week); WL[, cum_prev := shift(cumsum(spent), 1, fill = 0), by = user_id]
WLp <- merge(WLp, WL[, .(user_id, week, cum_prev)], by = c("user_id", "week"))
put("dr_cumulative", tidy_fx(fepois(commits ~ lspend + log1p(cum_prev / 1e6) + log(present) | user_id + week,
                                    WLp[week > min(week)], cluster = ~user_id)))
# scale: within-person SD of weekly log spend, to express the slope per SD
put("dr_within_sd", sd(WLp[, lspend - mean(lspend), by = user_id]$V1))

# alternative exposure measures
put("dr_events", one(fepois(commits ~ levents + log(present) | user_id + week, WLp, cluster = ~user_id), "levents"))
put("dr_tokens", one(fepois(commits ~ ltokens + log(present) | user_id + week, WLp, cluster = ~user_id), "ltokens"))
put("dr_aidays", one(fepois(commits ~ ai_days + log(present) | user_id + week, WLp, cluster = ~user_id), "ai_days"))

# Heilman-style: a person's heaviest-usage week against their other weeks, and
# within-person usage quartiles
WLp[, top_week := as.integer(spent == max(spent) & spent > 0), by = user_id]
WLp[, q := {r <- frank(spent, ties.method = "average"); cut(r / .N, c(0, .25, .5, .75, 1), labels = FALSE, include.lowest = TRUE)}, by = user_id]
put("dr_topweek", one(fepois(commits ~ top_week + log(present) | user_id + week, WLp, cluster = ~user_id), "top_week"))
mq <- fepois(commits ~ i(q, ref = 1) + log(present) | user_id + week, WLp, cluster = ~user_id)
put("dr_quartile", tidy_fx(mq, c("q::2", "q::3", "q::4")))

# other outcomes
for (y in c("commit_days", "direct", "large", "linked")) {
  put(paste0("dr_y_", y), one(fepois(as.formula(paste(y, "~ lspend + log(present) | user_id + week")),
                                     WLp, cluster = ~user_id), "lspend"))
}
# robustness
put("dr_drop_last", one(fepois(commits ~ lspend + log(present) | user_id + week, WLp[week < 12], cluster = ~user_id), "lspend"))
put("dr_raw", one(fepois(commits_raw ~ lspend + log(present) | user_id + week, WLp, cluster = ~user_id), "lspend"))
put("dr_ols", one(feols(log1p(commits) ~ lspend + log(present) | user_id + week, WLp, cluster = ~user_id), "lspend"))
topspender <- P[user == TRUE][order(-spent)][1, user_id]
put("dr_no_top", one(fepois(commits ~ lspend + log(present) | user_id + week, WLp[user_id != topspender], cluster = ~user_id), "lspend"))
put("dr_full_presence", one(fepois(commits ~ lspend | user_id + week,
                                   WLp[, if (all(present == weekdays)) .SD, by = user_id], cluster = ~user_id), "lspend"))

# permutation inference: shuffle each person's weekly spend across their own
# weeks (keeps who-is-a-heavy-user fixed, breaks only the timing), via ritest.
# ritest computes permuted coefficients by OLS after demeaning, so it is valid
# for linear models only; it is run on the OLS specification (log(1 + commits)),
# not on the Poisson one. ritest re-reads the model's data by name and needs
# numeric columns, so the model gets its own frozen copy with an integer id.
WRI <- WLp[, .(uid = as.integer(factor(user_id)), week, commits, lspend, lpresent = log(present))]
WRI[, lc := log1p(commits)]
ols_ri <- feols(lc ~ lspend + lpresent | uid + week, WRI, cluster = ~uid)
stopifnot(isTRUE(all.equal(unname(coef(ols_ri)["lspend"]), R[["dr_ols"]]$b)))
rt <- ritest::ritest(ols_ri, "lspend", reps = 500, strata = "uid", seed = 20260926,
                     verbose = FALSE, parallel = FALSE, stack = FALSE)
put("dr_permutation", list(model = "OLS, log(1 + commits)", observed = unname(coef(ols_ri)["lspend"]),
                           observed_se = unname(se(ols_ri)["lspend"]), p = rt$pval, reps = 500,
                           null_sd = sd(rt$betas), null_q = unname(quantile(rt$betas, c(.025, .975))),
                           method = "ritest, lspend permuted within person"))

# ---- further checks on the claim-9 model (same set as the scaling model)
# (1) time grains: 2- and 4-week periods
dr_grain <- list()
for (k in c(2L, 4L)) {
  G <- W[user == TRUE & link == 1, .(commits = sum(commits), spent = sum(spent), present = sum(present)),
         by = .(user_id, win = (week - 7L) %/% k)][present > 0]   # log(present) control needs present > 0
  if (5L %/% k >= 2L) G <- G[win < 5L %/% k]                      # complete blocks only (see window_panel)
  G[, lspend := log1p(spent / 1e6)]
  dr_grain[[paste0("w", k)]] <- one(fepois(commits ~ lspend + log(present) | user_id + win, G, cluster = ~user_id), "lspend")
}
put("dr_grain", dr_grain)
# (2) shape: spend in bands of log(1 + spend / 10^6), no assumed shape, and a
#     test that the bands add nothing to the linear term
WLs <- copy(WLp)[, sband := cut(lspend, c(-Inf, 0, 1, 2, 3, 4, Inf), labels = c("none", "0-1", "1-2", "2-3", "3-4", "4+"))]
WLs[, sband := as.character(sband)]
mbs <- fepois(commits ~ i(sband, ref = "none") + log(present) | user_id + week, WLs, cluster = ~user_id)
mbs_t <- fepois(commits ~ lspend + i(sband, ref = "none") + log(present) | user_id + week, WLs, cluster = ~user_id)
put("dr_bands", list(fit = tidy_fx(mbs), counts = as.list(WLs[, .N, by = sband][order(sband)]),
                     mid = as.list(WLs[, .(m = mean(lspend)), by = sband][order(sband)]),
                     test = wald_zero(mbs_t, "^sband::")))
# (3) a slope per person: fixest varying slopes, people with 4+ present weeks
#     and at least 3 distinct weekly spend levels; descriptive
WLv2 <- WLp[, if (.N >= 4 && uniqueN(round(lspend, 3)) >= 3) .SD, by = user_id]
mvs <- fepois(commits ~ log(present) | user_id[lspend] + week, WLv2)
pss <- fixef(mvs)[["user_id[[lspend]]"]]
put("dr_person_slopes", list(people = length(pss), median = median(pss), q = unname(quantile(pss, c(.1, .25, .5, .75, .9))),
                             pooled_same_sample = unname(coef(fepois(commits ~ lspend + log(present) | user_id + week, WLv2))["lspend"]),
                             slopes = unname(pss)))

# ======================================= B3. daily distributed lags (weekdays)
DL <- D[user == TRUE & away == FALSE]
DL[, lspend := log1p(spent / 1e6)]
days <- sort(unique(D$date)); DL[, t := match(date, days)]
dl <- fepois(commits ~ l(lspend, -3:3) | user_id + date, DL, panel.id = ~user_id + t, cluster = ~user_id)
put("daily_lags", tidy_fx(dl))
dl2 <- fepois(commit_any ~ l(lspend, -3:3) | user_id + date,
              DL[, commit_any := as.integer(commits_raw > 0)], panel.id = ~user_id + t, cluster = ~user_id)
put("daily_lags_any", tidy_fx(dl2))

# ======================== B4. late adopters: staggered event study, weekly
# Treated: users whose first AI event is on or after 2026-07-06, so at least one
# telemetry week of observed non-use. Controls: the 178 who never use AI.
# People already using AI in the first telemetry week have no observed onset
# and are left out.
ES <- W[user == FALSE | late == TRUE]
ES[, g := ifelse(user == TRUE, as.integer(cohort_week), 0L)]
ES[, id := as.integer(factor(user_id))]
ES[, lc := log1p(commits)]
ES[, lcd := log1p(commit_days)]
put("es_counts", list(treated = uniqueN(ES[g > 0, user_id]), controls = uniqueN(ES[g == 0, user_id]),
                      cohorts = as.list(ES[g > 0, .(n = uniqueN(user_id)), by = g][order(g)])))
# people with at least two untreated rows: the sample on which imputation and
# OLS-first-stage implementations (which drop singleton groups) coincide
min2 <- function(dt) dt[, if (sum(g == 0 | week < g) >= 2) .SD, by = id]
bj_static <- function(dt, y) as.list(as.data.table(did_imputation(data = dt, yname = y, gname = "g", tname = "week", idname = "id")))
es_run <- function(dt, y, tag, pre = -6:-1) {
  put(paste0("es_sample_", tag), list(rows = nrow(dt), units = uniqueN(dt$id),
                                      treated_units = uniqueN(dt[g > 0, id]), control_units = uniqueN(dt[g == 0, id]),
                                      treated_post_rows = nrow(dt[g > 0 & week >= g]), weeks = sort(unique(dt$week))))
  st <- did_imputation(data = dt, yname = y, gname = "g", tname = "week", idname = "id")
  dy <- did_imputation(data = dt, yname = y, gname = "g", tname = "week", idname = "id",
                       horizon = 0:4, pretrends = pre)
  put(paste0("es_static_", tag), as.list(as.data.table(st)))
  put(paste0("es_dynamic_", tag), as.data.table(dy))
}
es_run(ES, "lc", "commits")
es_run(ES, "lcd", "commitdays")
# sustained adopters only: AI active in at least two link weeks after onset
sus <- W[user == TRUE & late == TRUE & link == 1 & events > 0, .N, by = user_id][N >= 2, user_id]
es_run(ES[g == 0 | user_id %in% sus], "lc", "commits_sustained")
# the pre-adoption dip: people coming out of a spell with no visible activity
# (no commits, no AI; the cause is not in the data) resume both together. Two corrections. (1) Anticipation: drop each adopter's three weeks
# before onset, so the counterfactual is imputed from their earlier weeks.
ES[, h := ifelse(g > 0, week - g, NA_integer_)]
es_run(ES[is.na(h) | !(h %in% -3:-1)], "lc", "commits_noanticip", -7:-4)
es_run(ES[is.na(h) | !(h %in% -3:-1)], "lcd", "commitdays_noanticip", -7:-4)
# (2) Inactivity, defined the same way for everyone from commits alone: drop
# person-weeks with three or more weekdays inside a 5+ weekday no-commit spell.
es_run(ES[active_out == 1], "lc", "commits_present")
es_run(ES[active_out == 1 & (is.na(h) | !(h %in% -3:-1))], "lc", "commits_present_noanticip", -7:-4)
# sensitivity: some adopters keep fewer than two untreated weeks once inactive
# weeks are dropped. Imputation identifies their person effect from that one week; OLS
# first stages that drop singleton groups (fixest, did2s) do not. Requiring two
# untreated weeks gives the sample on which every implementation agrees.
ES_p2 <- ES[active_out == 1][, if (sum(g == 0 | week < g) >= 2) .SD, by = id]
es_run(ES_p2, "lc", "commits_present_min2")
# Sun-Abraham as a second estimator
ES[, g_sa := ifelse(g == 0, 10000L, g)]
sa <- feols(lc ~ sunab(g_sa, week, ref.p = -1) | id + week, ES, cluster = ~id)
put("es_sunab", tidy_fx(sa))
put("es_sunab_att", as.list(summary(sa, agg = "ATT")$coeftable[1, ]))
# pre-trend joint test on Sun-Abraham leads

put("es_sunab_pretest", wald_zero(sa, "^week::-[2-6]:cohort"))

# ======================= B4b. switching to a new model released in the window
# The new frontier model (model_02) first appears on 2026-07-24. A person "switches" in the first
# week it carries at least half their spend. Sample: users, link weeks with
# some AI use (a switch can only happen in such a week, so the comparison
# weeks must have use too). Controls are users who have not switched yet or
# never do. Estimates the capability channel with use held on.
MS <- W[user == TRUE & link == 1 & spent > 0]
sw <- MS[spent_new_model / spent >= 0.5, .(g = min(week)), by = user_id]
MS <- merge(MS, sw, by = "user_id", all.x = TRUE); MS[is.na(g), g := 0L]
MS[, id := as.integer(factor(user_id))]
MS[, `:=`(lc = log1p(commits), ls = log(spent), ltk = log(tokens), lprice = log(spent / tokens))]
put("switch_counts", list(switchers = uniqueN(MS[g > 0, user_id]), others = uniqueN(MS[g == 0, user_id]),
                          cohorts = as.list(MS[g > 0, .(n = uniqueN(user_id)), by = g][order(g)])))
put("switch_sample", list(rows = nrow(MS), units = uniqueN(MS$id), weeks = sort(unique(MS$week))))
for (y in c("lc", "ls", "ltk", "lprice")) {
  st <- did_imputation(data = MS, yname = y, gname = "g", tname = "week", idname = "id")
  dy <- did_imputation(data = MS, yname = y, gname = "g", tname = "week", idname = "id",
                       horizon = 0:1, pretrends = -2:-1)
  put(paste0("switch_static_", y), as.list(as.data.table(st)))
  put(paste0("switch_static_", y, "_min2"), bj_static(min2(MS), y))
  put(paste0("switch_dynamic_", y), as.data.table(dy))
}

# how much of the switch gain travels through the extra use? A descriptive
# mediation check (spend is itself moved by the switch, so it is a bad control
# in the causal sense): the switch effect with and without weekly spend.
MS[, post := as.integer(g > 0 & week >= g)]
m_a <- fepois(commits ~ post + log(present) | user_id + week, MS[present > 0], cluster = ~user_id)
m_b <- fepois(commits ~ post + ls + log(present) | user_id + week, MS[present > 0], cluster = ~user_id)
put("switch_twfe", one(m_a, "post"))
put("switch_twfe_spend", tidy_fx(m_b, c("post", "ls")))

# ============================== B5. what changes inside a commit (commit level)
CU <- CL[user == TRUE & present > 0]
CU[, lspend := log1p(spent / 1e6)]
CU[, lsize := log1p(size_bin)]
for (y in c("large", "linked", "direct", "github", "lsize")) {
  put(paste0("commit_", y), one(feols(as.formula(paste(y, "~ lspend | user_id + week")), CU, cluster = ~user_id), "lspend"))
}
put("commit_base_rates", list(large = mean(CU$large), linked = mean(CU$linked), direct = mean(CU$direct),
                              github = mean(CU$github), n = nrow(CU)))

# ============================================= B6. heterogeneity of the slope
WLp[, main_vendor := fifelse(is.na(agent_a_share), "none", fifelse(agent_a_share >= 0.5, "Agent A", "Agent B"))]
WLp[, manager := fifelse(is_manager, "manager", "IC")]
het <- list()
pv <- c()
for (g in c("manager", "family", "base_tercile", "main_vendor", "tier")) {
  f <- as.formula(paste0("commits ~ lspend:i(", g, ") + log(present) | user_id + week"))
  m <- fepois(f, WLp, cluster = ~user_id)
  # equality of slopes: difference from the pooled slope via an interaction model
  f2 <- as.formula(paste0("commits ~ lspend + lspend:i(", g, ") + log(present) | user_id + week"))
  m2 <- fepois(f2, WLp, cluster = ~user_id)
  w2 <- wald_zero(m2, paste0("^lspend:", g, "::"))
  het[[g]] <- list(slopes = tidy_fx(m)$coef[grepl("lspend", term)], wald_p = w2$p, wald_stat = w2$stat, wald_df = w2$df,
                   meta = meta_fx(m), meta_test = meta_fx(m2))
  pv <- c(pv, setNames(w2$p, g))
}
put("heterogeneity", het)
put("heterogeneity_bh", as.list(p.adjust(pv, "BH")))

# ============================== C. review and issues (cross-sectional, thin)
per <- function(d0, d1) RV[created >= as.IDate(d0) & created <= as.IDate(d1),
                           .(authored = sum(author), reviewed = sum(reviewer)), by = user_id]
pre_r <- per("2026-05-11", "2026-06-28"); setnames(pre_r, c("user_id", "pre_auth", "pre_rev"))
lnk_r <- per("2026-06-29", LINK_END); setnames(lnk_r, c("user_id", "lnk_auth", "lnk_rev"))
PR <- merge(P[, .(user_id, tier, lpre, family, is_manager, div_top)], pre_r, by = "user_id", all.x = TRUE)
PR <- merge(PR, lnk_r, by = "user_id", all.x = TRUE)
for (v in c("pre_auth", "pre_rev", "lnk_auth", "lnk_rev")) PR[is.na(get(v)), (v) := 0]
PRg <- PR[user_id %in% RV$user_id]
put("review_people", nrow(PRg))
put("review_authored", tidy_fx(fepois(lnk_auth ~ tier + log1p(pre_auth) + lpre + family + is_manager, PRg, vcov = "hetero"),
                               c("tierlight", "tiermedium", "tierheavy")))
put("review_reviewed", tidy_fx(fepois(lnk_rev ~ tier + log1p(pre_rev) + lpre + family + is_manager, PRg, vcov = "hetero"),
                               c("tierlight", "tiermedium", "tierheavy")))
# authored PRs opened in the link window: days open in window by author tier,
# for PRs of the same size and opening week
AU <- RV[author == TRUE & created >= as.IDate("2026-06-29")]
bin <- function(x) as.numeric(sub("\\+$", "", x))
AU[, `:=`(ladd = log1p(bin(additions)), lfiles = log1p(bin(n_files)), wk = as.integer((created - as.IDate("2026-05-11")) %/% 7))]
AU <- merge(AU, P[, .(user_id, tier)], by = "user_id")
put("review_toiw", tidy_fx(feols(log1p(toiw) ~ tier + ladd + lfiles | wk, AU, cluster = ~user_id),
                           c("tierlight", "tiermedium", "tierheavy")))
put("review_toiw_n", nrow(AU))

YT <- fread(file.path(DATA, "issues.csv"))
resolved <- c("Fixed", "Done", "Verified", "Fixed in Branch", "Fixed In Branch", "Released", "Completed",
              "Merged", "Solved", "Closed", "Deployed", "Answered")
IS <- YT[, .(n_issues = .N, n_resolved = sum(state %in% resolved)), by = user_id]
put("issues_resolved_states", resolved)
PI <- merge(P[, .(user_id, tier, lpre, family, is_manager, div_top)], IS, by = "user_id", all.x = TRUE)
PI[is.na(n_issues), `:=`(n_issues = 0, n_resolved = 0)]
put("issues_n", tidy_fx(fepois(n_issues ~ tier + lpre + family + is_manager + div_top, PI, vcov = "hetero"),
                        c("tierlight", "tiermedium", "tierheavy")))
put("issues_resolved", tidy_fx(fepois(n_resolved ~ tier + log1p(n_issues) + lpre + family + is_manager + div_top,
                                      PI[n_issues > 0], vcov = "hetero"), c("tierlight", "tiermedium", "tierheavy")))

# ======================================================================
# M. The mandate paper's tests (He, Agarwal et al. 2026), run on our data.
# Their developer-month panel becomes our person-week panel; their "Adopted"
# switch is observable here only for the late adopters (first AI event on or
# after 2026-07-06), compared with the never-users. Following their code,
# log models use active periods only (commits > 0), which also keeps inactive
# weeks out; the Poisson check keeps zeros within each person's active span.
# ======================================================================
setorder(W, user_id, week)
W[, cum_prev := shift(cumsum(spent), 1, fill = 0), by = user_id]
M <- W[user == FALSE | late == TRUE]
M[, g := ifelse(user == TRUE, as.integer(cohort_week), 0L)]
M[, id := as.integer(factor(user_id))]
M[, adopted := as.integer(g > 0 & week >= g)]
M[, post := as.integer(week >= 7)]                      # telemetry on (their Post = mandate)
M[, lcum := log1p(cum_prev / 1e6)]
MA <- M[commits > 0]
MA[, ly := log(commits)]
put("m_sample", list(rows_all = nrow(M), rows_active = nrow(MA), adopters = uniqueN(M[g > 0, user_id]),
                     never = uniqueN(M[g == 0, user_id]), active_share = nrow(MA) / nrow(M)))

# Eq 1, their baseline and their preferred (calendar-week effects) forms
e1a <- feols(ly ~ adopted + post | user_id, MA, cluster = ~user_id)
e1b <- feols(ly ~ adopted | user_id + week, MA, cluster = ~user_id)
put("m_eq1_baseline", tidy_fx(e1a, c("adopted", "post")))
put("m_eq1_weekfe", one(e1b, "adopted"))
# zero-inclusive Poisson within each person's active span (first to last active week)
span <- M[commits > 0, .(first = min(week), last = max(week)), by = user_id]
MS2 <- merge(M, span, by = "user_id")[week >= first & week <= last]
put("m_eq1_poisson", one(fepois(commits ~ adopted | user_id + week, MS2, cluster = ~user_id), "adopted"))
# ever-adopters only (no never-users in the comparison)
put("m_eq1_adopters_only", one(feols(ly ~ adopted | user_id + week, MA[g > 0], cluster = ~user_id), "adopted"))

# the estimator battery, on their outcome (log commits, active weeks) and on
# ours from claim 12 (log(1 + commits), all weeks)
# Callaway & Sant'Anna (2021) through the did package, as in the mandate
# paper's code: not-yet-treated controls, universal base period, no
# anticipation, unbalanced panels allowed, analytic standard errors.
cs_run <- function(dt, y) {
  a <- did::att_gt(yname = y, tname = "week", idname = "id", gname = "g", data = as.data.frame(dt),
                   control_group = "notyettreated", base_period = "universal", anticipation = 0,
                   allow_unbalanced_panel = TRUE, bstrap = FALSE, est_method = "reg", print_details = FALSE)
  s <- did::aggte(a, type = "simple", na.rm = TRUE)
  list(att = s$overall.att, se = s$overall.se, n_gt = length(a$att),
       method = "did::att_gt (not-yet-treated, universal base) + aggte(simple)")
}
bj_run <- function(dt, y) as.list(as.data.table(did_imputation(data = dt, yname = y, gname = "g", tname = "week", idname = "id")))
M[, lc := log1p(commits)]
put("m_battery_active", list(
  twfe = one(e1a, "adopted"), twfe_week = one(e1b, "adopted"),
  adopters_only = R[["m_eq1_adopters_only"]],
  cs = tryCatch(cs_run(MA, "ly"), error = function(e) list(error = conditionMessage(e))),
  bjs = tryCatch(bj_run(MA, "ly"), error = function(e) list(error = conditionMessage(e)))))
put("m_bjs_min2", list(active = bj_static(min2(MA), "ly"), all = bj_static(min2(M), "lc")))
put("m_battery_all", list(
  twfe = one(feols(lc ~ adopted + post | user_id, M, cluster = ~user_id), "adopted"),
  twfe_week = one(feols(lc ~ adopted | user_id + week, M, cluster = ~user_id), "adopted"),
  adopters_only = one(feols(lc ~ adopted | user_id + week, M[g > 0], cluster = ~user_id), "adopted"),
  cs = tryCatch(cs_run(M, "lc"), error = function(e) list(error = conditionMessage(e))),
  bjs = tryCatch(bj_run(M, "lc"), error = function(e) list(error = conditionMessage(e)))))

# Eq 2: the release ladder with a placebo, for users and, as a falsification
# the paper could not run, for non-users (who use no model at all). Steps
# switch on in the first full week after a model's first appearance.
RL <- W[commits > 0]
RL[, ly := log(commits)]
# first full Monday-week after each model's first appearance (a vendor-B model family
# on 2026-07-09, the new frontier model on 2026-07-24), telemetry on, and a placebo week
LADDER <- list(post = 7L, step_model_b = 9L, step_new_model = 11L, placebo = 4L)
RL[, `:=`(post = as.integer(week >= LADDER$post), step_model_b = as.integer(week >= LADDER$step_model_b),
          step_new_model = as.integer(week >= LADDER$step_new_model), placebo = as.integer(week >= LADDER$placebo))]
put("m_ladder_weeks", LADDER)
put("m_ladder_users", tidy_fx(feols(ly ~ post + step_model_b + step_new_model + placebo | user_id, RL[user == TRUE], cluster = ~user_id),
                              c("post", "step_model_b", "step_new_model", "placebo")))
put("m_ladder_nonusers", tidy_fx(feols(ly ~ post + step_model_b + step_new_model + placebo | user_id, RL[user == FALSE], cluster = ~user_id),
                                 c("post", "step_model_b", "step_new_model", "placebo")))
RL[, u := as.integer(user)]
put("m_ladder_diff", tidy_fx(feols(ly ~ (post + step_model_b + step_new_model + placebo) * u | user_id, RL, cluster = ~user_id),
                             c("post:u", "step_model_b:u", "step_new_model:u", "placebo:u")))

# Eq 3: adoption jump plus accumulated use, evaluated at the adopters' mean
# accumulated use in the final week
e3a <- feols(ly ~ adopted + lcum + post | user_id, MA, cluster = ~user_id)
e3b <- feols(ly ~ adopted + lcum | user_id + week, MA, cluster = ~user_id)
lcum_end <- M[g > 0 & week == max(week), mean(lcum)]
put("m_eq3_baseline", tidy_fx(e3a, c("adopted", "lcum", "post")))
put("m_eq3_weekfe", tidy_fx(e3b, c("adopted", "lcum")))
put("m_eq3_lcum_end", lcum_end)

# Fig 5: event study by post-adoption usage tier (late adopters in thirds by
# their link-window spend), each against the never-users
tiers_late <- P[late == TRUE, .(user_id, spent)]
tiers_late[, ltier := as.character(cut(frank(spent, ties.method = "first") / .N, c(0, 1/3, 2/3, 1), labels = c("light", "average", "heavy")))]
MA2 <- merge(MA, tiers_late[, .(user_id, ltier)], by = "user_id", all.x = TRUE)
MA2[, rel := ifelse(g > 0, week - g, -1000L)]
MA2 <- MA2[g == 0 | rel >= -6]
es_tier <- list()
for (tr in c("light", "average", "heavy")) {
  m <- feols(ly ~ i(rel, ref = c(-1, -1000)) | user_id + week, MA2[g == 0 | ltier == tr], cluster = ~user_id)
  es_tier[[tr]] <- list(coef = tidy_fx(m)$coef, adopters = uniqueN(MA2[ltier == tr, user_id]), meta = meta_fx(m))
}
put("m_es_tiers", es_tier)
# Table III: the adoption jump by manager flag and by developer vs other roles
MA[, dev := as.integer(family == "Developer")]
put("m_het_manager", tidy_fx(feols(ly ~ adopted + adopted:is_manager | user_id + week, MA, cluster = ~user_id), c("adopted", "adopted:is_managerTRUE")))
put("m_het_dev", tidy_fx(feols(ly ~ adopted + adopted:dev | user_id + week, MA, cluster = ~user_id), c("adopted", "adopted:dev")))
put("m_het_counts", list(adopter_managers = uniqueN(M[g > 0 & is_manager == TRUE, user_id]),
                         adopter_devs = uniqueN(M[g > 0 & family == "Developer", user_id])))

# Eq 4, adapted: the same author's PRs opened in heavier- vs lighter-AI weeks
# (no AI label exists here; the author's AI spend that week stands in for it)
AU2 <- merge(AU, W[, .(user_id, wk = week, lspend_w = log1p(spent / 1e6))], by = c("user_id", "wk"))
AU2 <- AU2[user_id %in% P[user == TRUE, user_id]]
AU2[, n_rev := n_reviewers]
put("m_eq4_n", list(prs = nrow(AU2), authors = uniqueN(AU2$user_id)))
put("m_eq4_toiw", one(feols(log1p(toiw) ~ lspend_w + ladd + lfiles | user_id + wk, AU2, cluster = ~user_id), "lspend_w"))
put("m_eq4_reviewers", one(fepois(n_rev ~ lspend_w + ladd + lfiles | user_id + wk, AU2, cluster = ~user_id), "lspend_w"))
put("m_eq4_size", one(feols(ladd ~ lspend_w | user_id + wk, AU2, cluster = ~user_id), "lspend_w"))

# ======================================================================
# D. Dark spend and conversion: the token-study draft's RQ1 and RQ3.
# Token spend is the outcome and recorded output the explanation, in a
# Poisson model with person and period fixed effects (the draft's correlated
# random effect has the same within-person identification; lme4/glmmTMB do
# not load on this R). Output enters as raw counts, so the model's predicted
# spend with every output set to zero is spend not associated with any
# recorded output: an upper bound on "dark" spend. Units: users, link window,
# periods with at least one weekday present.
# ======================================================================
set.seed(20260926)
dark <- list()
for (k in c(1L, 2L, 4L)) {
  X <- window_panel(k)
  dark[[paste0("w", k)]] <- list(
    raw_zero_share = X[commits == 0, sum(spent)] / X[, sum(spent)],
    units = nrow(X), zero_units = X[commits == 0, .N],
    a = dark_fit(X, OUT_A, "win"), b = dark_fit(X, OUT_B, "win"), c = dark_fit(X, OUT_C, "win"))
}
dark[["day"]] <- list(raw_zero_share = DD[commits_raw == 0, sum(spent)] / DD[, sum(spent)],
                      units = nrow(DD), zero_units = DD[commits_raw == 0, .N],
                      a = dark_fit(DD, OUT_A, "date"), b = dark_fit(DD, OUT_B, "date"))
put("dark", dark)

# ======================================================================
# D0. The common specification of the token-study plan (August draft),
#     token-study-loc.tex Sec. 4.1 and Appendix A:
#       E[consumption_it] = exp(c_i + b's(output_it) + t_t)
#     - consumption: total tokens is primary, credits (`spent`) the
#       sensitivity; both in millions, which leaves the Poisson mean model
#       unchanged (fixest) and keeps the mixed models numerically stable
#     - c_i: a correlated random effect (Mundlak: person means of every
#       output measure in the linear predictor, glmmTMB), with person fixed
#       effects (fixest) as the cross-check; the within-person b is the same
#       under both
#     - dispersion: Poisson, and negative binomial (NB2) as the second
#       specification; gamma on positive weeks as the paper's sensitivity
#     - output: A commits; B + size and kind; Bt = B with issue-linked
#       commits split by the linked issue's type (our stand-in for merged PRs
#       by type); C = Bt + GitHub PRs authored and reviewed
#     The share is predicted consumption with every output measure at zero,
#     over predicted consumption. In the CRE models the Mundlak means stay at
#     their values: they are part of the person term, not output. The
#     population-averaged share uses fixed effects only; the lognormal
#     retransformation factor exp(sigma^2 / 2) multiplies numerator and
#     denominator alike and cancels, and is reported for completeness.
# ======================================================================
NBOOT <- as.integer(Sys.getenv("BOOT_R", "200"))   # bootstrap draws; BOOT_R=3 for a quick smoke test
set.seed(20261004)
setFixest_nthreads(1)   # the bootstraps fork; keep fixest single-threaded inside them
CS <- list()
# (1) fixed-effects Poisson, every grain, tokens and credits, every output set
DD[, `:=`(tok = tokens / 1e6, spm = spent / 1e6)]
for (g in c("w1", "w2", "w4", "day")) {
  X <- if (g == "day") DD else window_panel(as.integer(sub("w", "", g)))
  X[, `:=`(tok = tokens / 1e6, spm = spent / 1e6)]
  time <- if (g == "day") "date" else "win"
  sets <- if (g == "day") SETS[c("A", "B", "Bt")] else SETS
  CS$fe[[g]] <- lapply(c(tok = "tok", spm = "spm"), function(y) lapply(sets, function(ov) {
    m <- fepois(fe_fml(y, ov, time), X, cluster = ~user_id)
    bs <- boot_share(X, NBOOT, function(B) share_fe(fepois(fe_fml(y, ov, time), B), B, ov))
    c(list(share = share_fe(m, X, ov)), bs, list(fit = tidy_fx(m)))
  }))
}
# (2) weekly: correlated random effects, Poisson and NB2, and FE NB2, tokens and credits
X <- window_panel(1L)[, `:=`(tok = tokens / 1e6, spm = spent / 1e6)]
for (y in c("tok", "spm")) for (sn in names(SETS)) {
  ov <- SETS[[sn]]; XC <- cre_data(X, ov)
  mp <- fit_cre(XC, y, ov, "win", poisson)
  mn <- fit_cre(XC, y, ov, "win", nbinom2)
  fn <- fenegbin(fe_fml(y, ov, "win"), X, cluster = ~user_id)
  bs <- boot_share(X, NBOOT, function(B) { BC <- cre_data(B, ov); share_cre(fit_cre(BC, y, ov, "win", poisson), BC, ov) })
  CS$cre[[y]][[sn]] <- list(
    poisson = c(list(share = share_cre(mp, XC, ov), share_pop = share_cre(mp, XC, ov, re = NA),
                     retrans = exp(VarCorr(mp)$cond$user_id[1] / 2)), bs, list(fit = tidy_cre(mp, ov))),
    nbinom2 = list(share = share_cre(mn, XC, ov), share_pop = share_cre(mn, XC, ov, re = NA), fit = tidy_cre(mn, ov)),
    fe_nbinom2 = list(share = share_fe(fn, X, ov), fit = tidy_fx(fn)))
}
# (3) gamma sensitivity on positive-consumption weeks (gamma is undefined at zero)
XG <- X[tokens > 0]
CS$gamma <- lapply(SETS, function(ov) {
  m <- feglm(fe_fml("tok", ov, "win"), XG, family = Gamma(link = "log"), cluster = ~user_id, glm.iter = 500)
  list(share = share_fe(m, XG, ov), fit = tidy_fx(m), dropped_zero_weeks = X[tokens == 0, .N])
})
# (4) participation margin: share of present user-weeks with no AI use, by week
CS$participation <- list(overall = X[, mean(tokens == 0)],
                         by_week = as.list(X[, .(zero = mean(tokens == 0), n = .N), by = win][order(win)]))
# (5) correlations among the output measures (collinearity, and the type split)
CS$output_cor <- cor(as.matrix(X[, ..OUT_CT]))
# (6) tokens against credits: how differently the two readings classify people
PT <- X[, .(tok = sum(tokens), spm = sum(spent)), by = user_id]
PT[, `:=`(tq = cut(rank(tok), 3, labels = FALSE), sq = cut(rank(spm), 3, labels = FALSE))]
CS$tok_vs_credits <- list(spearman = cor(PT$tok, PT$spm, method = "spearman"),
                          tercile_disagree = PT[, mean(tq != sq)], people = nrow(PT))
put("common_spec", CS)

# ======================================================================
# D1. RQ1, the ceiling (plan Sec. 4.2 and the pilot, Sec. 5). The counts
#     form bends upward where consumption flattens, so the plan makes the
#     flexible forms primary and keeps counts as the sensitivity:
#       counts : output measures as they are
#       log1p  : log(1 + x) for every output measure
#       spline : natural spline in commits (splines::ns, one interior knot at
#                the median of positive-commit units, boundary knots at 0 and
#                the maximum, so the basis is exactly zero at zero commits),
#                log(1 + x) for the other measures
#     All three are defined at zero output, so the share is read the same
#     way. Diagnostics the plan names: the calibration check (observed over
#     fitted consumption in all zero-output units), agreement across
#     dispersion families, and the tool-mix-adjusted bound. Added: 5-fold
#     out-of-sample Poisson deviance, to compare the forms on held-out units.
#     Tokens throughout; fixed effects (claim 27 shows the CRE conditional
#     share is identical), with the CRE fitted as a check on the primary set.
# ======================================================================
set.seed(20261005)
CE <- list(knots = list())
for (g in c("w1", "w2", "w4", "day")) {
  X <- if (g == "day") copy(DD) else window_panel(as.integer(sub("w", "", g)))
  X[, tok := tokens / 1e6]
  time <- if (g == "day") "date" else "win"
  kn <- form_knots(X); CE$knots[[g]] <- kn
  sets <- if (g == "day") SETS[c("A", "B", "Bt")] else SETS
  for (fm in FORMS) for (sn in names(sets)) {
    ov <- sets[[sn]]
    m <- fit_form(X, ov, fm, kn, time)
    bs <- boot_share(X, NBOOT, function(B) ceil_share(fit_form(B, ov, fm, kn, time), B, ov, fm, kn))
    CE$fe[[g]][[fm]][[sn]] <- c(list(share = ceil_share(m, X, ov, fm, kn), calib = calib(m, X, ov, fm, kn)), bs, list(fit = tidy_fx(m)))
  }
}
# weekly: dispersion families, CRE check, tool mix, out-of-sample deviance
X <- window_panel(1L)[, tok := tokens / 1e6]
X[, `:=`(agent_b_sh = fifelse(tokens > 0, tokens_agent_b / tokens, 0), new_model_sh = fifelse(tokens > 0, tokens_new_model / tokens, 0))]
kn <- CE$knots$w1
XG <- X[tokens > 0]
for (fm in FORMS) for (sn in names(SETS)) {
  ov <- SETS[[sn]]
  mnb <- fit_form(X, ov, fm, kn, "win", "nb"); mg <- fit_form(XG, ov, fm, kn, "win", "gamma")
  mt <- fit_form(X, ov, fm, kn, "win", extra = c("agent_b_sh", "new_model_sh"))
  bt <- boot_share(X, NBOOT, function(B) ceil_share(fit_form(B, ov, fm, kn, "win", extra = c("agent_b_sh", "new_model_sh")), B, ov, fm, kn))
  CE$fam[[fm]][[sn]] <- list(nb = ceil_share(mnb, X, ov, fm, kn), gamma = ceil_share(mg, XG, ov, fm, kn),
                             nb_calib = calib(mnb, X, ov, fm, kn)$ratio)
  CE$tool[[fm]][[sn]] <- c(list(share = ceil_share(mt, X, ov, fm, kn), fit = tidy_fx(mt, c("agent_b_sh", "new_model_sh"))), bt)
}
# the CRE (Mundlak) conditional share on the primary forms: should match FE
for (fm in c("log1p", "spline")) for (sn in c("Bt", "C")) {
  ov <- SETS[[sn]]; F <- add_form(X, ov, fm, kn); XC <- F$X
  for (v in F$vars) XC[, (paste0("mb_", v)) := mean(get(v)), by = user_id]
  mc <- quiet_int(glmmTMB(as.formula(paste("tok ~", paste(c(F$vars, paste0("mb_", F$vars), "factor(win)"), collapse = " + "), "+ (1 | user_id)")),
                          data = XC, family = poisson, control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4), parallel = 1L)))
  Z <- add_form(zero_out(X, ov), ov, fm, kn)$X
  for (v in F$vars) Z[, (paste0("mb_", v)) := XC[[paste0("mb_", v)]]]
  CE$cre[[fm]][[sn]] <- list(share = sum(predict(mc, Z, type = "response")) / sum(predict(mc, XC, type = "response")),
                             converged = mc$fit$convergence == 0, pdHess = isTRUE(mc$sdr$pdHess))
  # the plain random effect, without the Mundlak means
  mp <- quiet_int(glmmTMB(as.formula(paste("tok ~", paste(c(F$vars, "factor(win)"), collapse = " + "), "+ (1 | user_id)")),
                          data = XC, family = poisson, control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4), parallel = 1L)))
  CE$re_plain[[fm]][[sn]] <- list(share = sum(predict(mp, Z, type = "response")) / sum(predict(mp, XC, type = "response")),
                                  converged = mp$fit$convergence == 0, pdHess = isTRUE(mp$sdr$pdHess))
}
# out-of-sample: 5 random folds of person-weeks; Poisson deviance (stats::poisson()$dev.resids) on held-out units
fold <- sample(rep(1:5, length.out = nrow(X)))
for (fm in FORMS) for (sn in names(SETS)) {
  ov <- SETS[[sn]]; dv <- 0; nn <- 0
  for (k in 1:5) {
    m <- fit_form(X[fold != k], ov, fm, kn, "win")
    te <- add_form(X[fold == k], ov, fm, kn)$X
    mu <- predict(m, newdata = te); ok <- is.finite(mu)
    dv <- dv + sum(poisson()$dev.resids(te$tok[ok], mu[ok], rep(1, sum(ok)))); nn <- nn + sum(ok)
  }
  CE$cv[[fm]][[sn]] <- list(deviance = dv, units = nn)
}
put("ceiling", CE)

# ======================================================================
# D2. RQ1, the floor and concentration (plan Sec. 4.2).
#   Floor: share of all tokens falling in units with substantial use and no
#   artifact at all. Substantial = tokens above the median of nonzero units
#   (the plan's primary, absolute cut), or above the person's own median of
#   their nonzero units (its within-person sensitivity). No artifact = no
#   commit (GitHub PRs and reviews dropped 2026-10-23: output is commits only);
#   issues have no dates and cannot enter. Units of 1, 2 and 4 weeks (the plan's series) and days.
#   Leakage: how long authored PRs stay open in the window (the plan's
#   open-to-merge proxy; our extract has open time, not merge time).
#   Concentration: Gini (ineq) of person totals over the link window for
#   tokens, commits and PRs authored, users only; and how the ceiling's
#   zero-output component (log form, weekly) is spread across people.
# ======================================================================
RVd <- RV[created >= as.IDate("2026-06-29") & created <= as.IDate(LINK_END),
          .(prs = sum(author), revs = sum(reviewer)), by = .(user_id, date = created)]
floor_of <- function(X) {
  X <- copy(X)
  X[, artifact := commits > 0]
  cut <- median(X[tok > 0, tok])
  X[, own := median(tok[tok > 0]), by = user_id]
  X[is.na(own), own := Inf]
  c(abs = X[tok > cut & !artifact, sum(tok)] / X[, sum(tok)],
    within = X[tok > own & !artifact, sum(tok)] / X[, sum(tok)],
    any = X[artifact == FALSE, sum(tok)] / X[, sum(tok)])
}
FL <- list()
for (g in c("day", "w1", "w2", "w4")) {
  if (g == "day") {
    X <- merge(copy(DD), RVd, by = c("user_id", "date"), all.x = TRUE)
    X[is.na(prs), prs := 0][is.na(revs), revs := 0]
  } else X <- window_panel(as.integer(sub("w", "", g)))
  X[, tok := tokens / 1e6]
  f <- floor_of(X)
  b <- lapply(c("abs", "within"), function(k) boot_share(X, NBOOT, function(B) floor_of(B)[[k]]))
  names(b) <- c("abs", "within")
  FL[[g]] <- list(abs = c(list(share = f[["abs"]]), b$abs), within = c(list(share = f[["within"]]), b$within),
                  any = f[["any"]], units = nrow(X), no_artifact_units = X[!(commits > 0), .N],
                  floor_units = X[tok > median(X[tok > 0, tok]) & !(commits > 0), .N],
                  cut = median(X[tok > 0, tok]))
}
FL$leakage <- with(RV[author == TRUE & created >= as.IDate("2026-06-29") & created <= as.IDate(LINK_END)],
                   list(prs = length(toiw), q = unname(quantile(toiw, c(.25, .5, .75, .9))), under_7 = mean(toiw < 7),
                        open_to_end = mean(toiw >= as.numeric(as.IDate(LINK_END) - created) - 1e-6)))
# concentration over the link window, users
PU <- WLall[, .(tok = sum(tokens), commits = sum(commits), prs = sum(prs)), by = user_id]
FL$gini <- list(tokens = ineq::Gini(PU$tok), commits = ineq::Gini(PU$commits), prs = ineq::Gini(PU$prs), people = nrow(PU),
                people_with_prs = PU[prs > 0, .N])
# the ceiling's zero-output component (log form, + work type, weekly) across people
X <- window_panel(1L)[, tok := tokens / 1e6]
m <- fit_form(X, SETS$Bt, "log1p", CE$knots$w1, "win")
pp <- pred_pair(m, X, SETS$Bt, "log1p", CE$knots$w1)
PC <- data.table(user_id = pp$U$user_id, p0 = pp$p0, p1 = pp$p1)[, .(p0 = sum(p0), p1 = sum(p1)), by = user_id]
PC[, ratio := p0 / p1]
setorder(PC, -p0)
FL$component <- list(people = nrow(PC), ratio_q = unname(quantile(PC$ratio, c(.1, .25, .5, .75, .9))),
                     ratio_weighted = PC[, sum(p0) / sum(p1)], gini_component = ineq::Gini(PC$p0),
                     top10_share = PC[1:ceiling(.N / 10), sum(p0)] / PC[, sum(p0)],
                     top10_share_tokens = PC[order(-p1)][1:ceiling(.N / 10), sum(p1)] / PC[, sum(p1)])
put("floor", FL)

# RQ3 (a): how spend scales with commits, on periods with at least one commit.
# An elasticity of 1 means spend rises in proportion; below 1, spend per
# commit falls as volume rises, so cost-per-commit ranks people by volume.
test_equal <- function(m, name, value) {
  b <- coef(m); L <- matrix(0, 1, length(b), dimnames = list(NULL, names(b))); L[1, name] <- 1
  lh <- car::linearHypothesis(m, L, rhs = value, test = "Chisq", vcov. = vcov(m), coef. = b)
  list(stat = lh$Chisq[2], p = lh[["Pr(>Chisq)"]][2])
}
scal <- list()
for (k in c(1L, 2L, 4L)) {
  X <- window_panel(k)[commits > 0]
  m <- fepois(spent ~ log(commits) | user_id + win, X, cluster = ~user_id)
  scal[[paste0("w", k)]] <- c(one(m, "log(commits)"), list(vs_one = test_equal(m, "log(commits)", 1)))
}
DDp <- DD[commits_raw > 0]
m <- fepois(spent ~ log(commits) | user_id + date, DDp, cluster = ~user_id)
scal[["day"]] <- c(one(m, "log(commits)"), list(vs_one = test_equal(m, "log(commits)", 1)))
put("scaling", scal)
# the same scaling for the layers spend is made of: requests, tokens, and
# price per token (spend = tokens x price, so the tokens and price elasticities
# add up to the spend elasticity; price is a ratio, so it is fitted by OLS on
# log price)
scal_layers <- list()
for (k in c(1L, 2L, 4L)) {
  X <- window_panel(k)[commits > 0 & tokens > 0]
  scal_layers[[paste0("w", k)]] <- list(
    tokens = c(one(fepois(tokens ~ log(commits) | user_id + win, X, cluster = ~user_id), "log(commits)"),
               list(vs_one = test_equal(fepois(tokens ~ log(commits) | user_id + win, X, cluster = ~user_id), "log(commits)", 1))),
    events = one(fepois(events ~ log(commits) | user_id + win, X, cluster = ~user_id), "log(commits)"),
    price = one(feols(log(spent / tokens) ~ log(commits) | user_id + win, X, cluster = ~user_id), "log(commits)"))
}
XD <- DD[commits_raw > 0 & tokens > 0]
scal_layers[["day"]] <- list(
  tokens = c(one(fepois(tokens ~ log(commits) | user_id + date, XD, cluster = ~user_id), "log(commits)"),
             list(vs_one = test_equal(fepois(tokens ~ log(commits) | user_id + date, XD, cluster = ~user_id), "log(commits)", 1))),
  events = one(fepois(events ~ log(commits) | user_id + date, XD, cluster = ~user_id), "log(commits)"),
  price = one(feols(log(spent / tokens) ~ log(commits) | user_id + date, XD, cluster = ~user_id), "log(commits)"))
put("scaling_layers", scal_layers)
# does the elasticity drift over the six weeks (engineers learning the tools)?
X1 <- window_panel(1L)[commits > 0][, tc := win - mean(win)]
put("scaling_trend", tidy_fx(fepois(spent ~ log(commits) + log(commits):tc | user_id + win, X1, cluster = ~user_id),
                             c("log(commits)", "log(commits):tc")))

# Assumption checks for the scaling model (weekly grain, periods with commits).
# (1) Is the elasticity the same for everyone?
#   (a) a separate slope for every person: fixest varying-slope fixed effects,
#       for people with at least 4 weeks with commits and some variation in
#       commits; descriptive, since slopes from 4-6 weeks are noisy.
X1a <- copy(X1)[, lcm := log(commits)]
X1a <- X1a[, if (.N >= 4 && uniqueN(commits) >= 3) .SD, by = user_id]
mv <- fepois(spent ~ 1 | user_id[lcm] + win, X1a)
ps_slopes <- fixef(mv)[["user_id[[lcm]]"]]
put("scaling_person_slopes", list(people = length(ps_slopes), median = median(ps_slopes),
                                  q = unname(quantile(ps_slopes, c(.1, .25, .5, .75, .9))),
                                  share_neg = mean(ps_slopes < 0), share_above_1 = mean(ps_slopes > 1),
                                  pooled_same_sample = unname(coef(fepois(spent ~ lcm | user_id + win, X1a))["lcm"]),
                                  slopes = unname(ps_slopes)))
#   (b) systematic differences: a slope per group, and a car Wald test that
#       the group slopes are equal (interaction model), BH across the splits
X1b <- merge(X1[, lcm := log(commits)], P[, .(user_id, family, is_manager, tier, base_tercile)], by = "user_id")
X1b[, manager := fifelse(is_manager, "manager", "IC")]
X1b[, tier := as.character(tier)]
het_s <- list(); pv_s <- c()
for (gname in c("family", "manager", "tier", "base_tercile")) {
  m_slopes <- fepois(as.formula(paste0("spent ~ lcm:i(", gname, ") | user_id + win")), X1b, cluster = ~user_id)
  m_test <- fepois(as.formula(paste0("spent ~ lcm + lcm:i(", gname, ") | user_id + win")), X1b, cluster = ~user_id)
  w_ <- wald_zero(m_test, paste0("^lcm:", gname, "::"))
  het_s[[gname]] <- list(slopes = tidy_fx(m_slopes)$coef, wald_p = w_$p, wald_stat = w_$stat, wald_df = w_$df, meta = meta_fx(m_slopes))
  pv_s <- c(pv_s, setNames(w_$p, gname))
}
put("scaling_groups", het_s)
put("scaling_groups_bh", as.list(p.adjust(pv_s, "BH")))
# (2) Is the elasticity constant across output levels? Commit bands with no
#     assumed shape, and a test that the bands add nothing to log commits.
X1c <- copy(X1)[, lcm := log(commits)]
X1c[, band := cut(commits, c(0, 1, 3, 7, 15, 31, Inf), labels = c("1", "2-3", "4-7", "8-15", "16-31", "32+"))]
X1c[, band := as.character(band)]
mb <- fepois(spent ~ i(band, ref = "1") | user_id + win, X1c, cluster = ~user_id)
mb_test <- fepois(spent ~ lcm + i(band, ref = "1") | user_id + win, X1c, cluster = ~user_id)
put("scaling_bands", list(fit = tidy_fx(mb), counts = as.list(X1c[, .N, by = band][order(band)]),
                          geo_mid = as.list(X1c[, .(gm = exp(mean(log(commits)))), by = band][order(band)]),
                          test = wald_zero(mb_test, "^band::")))

# RQ3 (b): is a person's conversion level stable? Person effects from the
# scaling model fitted separately on odd and on even link weeks, correlated
# across the people present in both halves; the raw spend-per-commit ratio
# is shown alongside, as the dashboard version of the same idea.
half_fe <- function(par) {
  m <- fepois(spent ~ log(commits) | user_id + win, X1[win %% 2 == par])
  data.table(user_id = names(fixef(m)$user_id), fe = unname(fixef(m)$user_id))
}
HH <- merge(half_fe(0), half_fe(1), by = "user_id", suffixes = c("_even", "_odd"))
ct_s <- cor.test(HH$fe_even, HH$fe_odd, method = "spearman", exact = FALSE)
ct_p <- cor.test(HH$fe_even, HH$fe_odd, method = "pearson")
RR <- X1[, .(spc = sum(spent) / sum(commits)), by = .(user_id, par = win %% 2)]
RR <- dcast(RR, user_id ~ par, value.var = "spc")[!is.na(`0`) & !is.na(`1`)]
ct_r <- cor.test(RR$`0`, RR$`1`, method = "spearman", exact = FALSE)
put("reliability_points", HH)
put("reliability", list(people = nrow(HH), spearman = unname(ct_s$estimate), spearman_p = ct_s$p.value,
                        pearson = unname(ct_p$estimate), pearson_ci = unname(ct_p$conf.int),
                        ratio_people = nrow(RR), ratio_spearman = unname(ct_r$estimate), ratio_p = ct_r$p.value))

write_results(R, "results.json")
