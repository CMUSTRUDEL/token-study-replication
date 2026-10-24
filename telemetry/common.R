# Shared setup for the telemetry models: libraries, data, panels and every
# helper. Sourced by models.R and by the stage scripts (stage_*.R), so a new
# part can be run on its own without re-running everything before it.
# Objects defined here do no random draws; seeds are set by the callers.

suppressPackageStartupMessages({
  library(fixest); library(data.table); library(jsonlite)
  library(sandwich); library(lmtest); library(didimputation)
  library(did); library(car); library(ritest); library(cobalt); library(ineq)
  library(glmmTMB)
})
setFixest_notes(FALSE)
set.seed(20260926)

HERE0 <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
DIR <- file.path(dirname(HERE0), "out", "telemetry")
DATA <- Sys.getenv("TOKEN_SPEND_DATA", file.path(dirname(HERE0), "data"))
LINK_END <- Sys.getenv("LINK_END", "2026-08-02")
PANEL_RULE <- Sys.getenv("PANEL_RULE", "any")   # "any": a unit is in if it has anything recorded   # last day of the link window (panels.py uses the same env var)
P  <- fread(file.path(DIR, "persons.csv"))
W  <- fread(file.path(DIR, "person_week.csv"))
D  <- fread(file.path(DIR, "person_day.csv"))
CL <- fread(file.path(DIR, "commits_link.csv"))

R <- list()          # every result goes in here
put <- function(key, value) R[[key]] <<- value

# What every fitted model records about itself, so the report can state the
# exact specification and every observation the estimator dropped.
meta_fx <- function(m) {
  list(formula = paste(deparse(formula(m, type = "full")), collapse = " "),
       method = m$method,
       vcov = trimws(sub("Standard-errors:", "", grep("^Standard-errors", capture.output(print(summary(m))), value = TRUE)[1])),
       n_input = m$nobs_origin, n_used = nobs(m),
       n_dropped = m$nobs_origin - nobs(m),
       fe_groups_dropped = if (is.null(m$fixef_removed)) list() else lapply(m$fixef_removed, length),
       fe_sizes = if (is.null(m$fixef_sizes)) list() else as.list(m$fixef_sizes))
}
tidy_fx <- function(m, terms = NULL) {
  ct <- coeftable(m)
  ci <- confint(m)
  out <- data.table(term = rownames(ct), b = ct[, 1], se = ct[, 2], p = ct[, 4],
                    lo = ci[, 1], hi = ci[, 2])
  if (!is.null(terms)) out <- out[term %in% terms]
  list(coef = out, n = nobs(m),
       clusters = tryCatch(as.integer(m$fixef_sizes[["user_id"]]), error = function(e) NA_integer_),
       meta = meta_fx(m))
}
# joint Wald test that the coefficients matching `pattern` are all zero:
# car::linearHypothesis with a selection matrix, chi-square form, using the
# model's own (clustered) covariance matrix
# (For Sun-Abraham models fixest reports aggregated coefficients but the
# cohort-specific covariance matrix, so the test runs on the cohort-specific
# coefficients, which is the standard Sun-Abraham pre-trend test.)
wald_zero <- function(m, pattern) {
  V <- vcov(m)
  b <- coef(m)
  if (length(b) != ncol(V)) b <- coef(m, agg = FALSE)
  k <- grep(pattern, names(b))
  L <- matrix(0, length(k), length(b), dimnames = list(NULL, names(b)))
  L[cbind(seq_along(k), k)] <- 1
  lh <- car::linearHypothesis(m, L, test = "Chisq", vcov. = V, coef. = b)
  list(stat = lh$Chisq[2], df = lh$Df[2], p = lh[["Pr(>Chisq)"]][2], terms = names(b)[k],
       method = "car::linearHypothesis")
}
one <- function(m, term) {
  t <- tidy_fx(m, term)
  c(as.list(t$coef[1]), list(n = t$n, clusters = t$clusters, meta = t$meta))
}

# ---------------------------------------------------------------- shared prep
P[, lpre := log1p(pre_commits)]
for (v in c("pre_direct_share", "pre_github_share", "pre_linked_share", "pre_large_share"))
  P[is.na(get(v)), (v) := median(P[[v]], na.rm = TRUE)]
P[, family := relevel(factor(family), ref = "Developer")]
P[, tier := factor(tier, levels = c("non-user", "light", "medium", "heavy"))]
P[, user01 := as.integer(user)]

W[, lspend := log1p(spent / 1e6)]      # spent in millions; log1p keeps zero weeks
W[, levents := log1p(events)]
W[, ltokens := log1p(tokens / 1e6)]
WL <- W[user == TRUE & link == 1]      # users, the six link-window weeks
WL[, week := as.integer(week)]


# ---- review data (sections C and D)
RV <- fread(file.path(DATA, "reviews.csv"))
RV[, created := as.IDate(created)]
RV[, author := grepl("author", roles)]; RV[, reviewer := grepl("reviewer", roles)]

# ---- section D prep and the claim-24 helpers
RVw <- RV[created >= as.IDate("2026-06-29") & created <= as.IDate(LINK_END)]
RVw[, week := as.integer((created - as.IDate("2026-05-11")) %/% 7)]
prw <- RVw[, .(prs = sum(author), revs = sum(reviewer)), by = .(user_id, week)]
WLall <- merge(W[user == TRUE & link == 1], prw, by = c("user_id", "week"), all.x = TRUE)
WLall[is.na(prs), prs := 0][is.na(revs), revs := 0]
WLall[, `:=`(adds100 = adds / 100, dels100 = dels / 100)]     # lines (bin upper bounds), in hundreds

window_panel <- function(k) {
  X <- WLall[, .(spent = sum(spent), tokens = sum(tokens), events = sum(events),
                 commits = sum(commits), large = sum(large), linked = sum(linked),
                 bug = sum(bug), task = sum(task), feature = sum(feature), otype = sum(otype),
                 tokens_agent_b = sum(tokens_agent_b), tokens_new_model = sum(tokens_new_model),
                 tokens_out = sum(tokens_out), events_remote = sum(events_remote), events_off = sum(events_off),
                 events_long = sum(events_long), sessions = sum(sessions), ai_days = sum(ai_days),
                 direct = sum(direct), adds100 = sum(adds100), dels100 = sum(dels100), prs = sum(prs),
                 revs = sum(revs), present = sum(present), any_rec = sum(any_rec)), by = .(user_id, win = (week - 7L) %/% k)]
  # complete blocks only, as in the companion pipeline: with five link weeks, k = 2
  # keeps weeks 0-1 and 2-3 and drops week 4. For k = 4 every person has one complete
  # block, so no within-person fit exists; that panel keeps the {weeks 0-3 | week 4}
  # stub for the telemetry report only, and the paper does not quote it.
  if (5L %/% k >= 2L) X <- X[win < 5L %/% k]
  if (PANEL_RULE == "any") X[any_rec > 0] else X[present > 0]
}
OUT_A <- c("commits")
OUT_B <- c("commits", "large", "linked", "direct", "adds100", "dels100")
OUT_C <- c(OUT_B, "prs", "revs")
dark_share <- function(m, X, outvars) {
  U <- X[obs(m)]
  U0 <- copy(U)[, (outvars) := 0]
  sum(predict(m, newdata = U0)) / sum(predict(m, newdata = U))
}
fml_of <- function(outvars, time) as.formula(paste("spent ~", paste(outvars, collapse = " + "), "|", "user_id +", time))
dark_fit <- function(X, outvars, time, R = 200) {
  m <- fepois(fml_of(outvars, time), X, cluster = ~user_id)
  ids <- unique(X$user_id)
  stat <- function(idv, i) {                         # resample people, keep each person's periods together
    draw <- data.table(user_id = idv[i])[, bid := .I]
    B <- merge(draw, X, by = "user_id", allow.cartesian = TRUE)
    B[, user_id := paste0(user_id, "_", bid)]
    mb <- fepois(fml_of(outvars, time), B)
    dark_share(mb, B, outvars)
  }
  b <- boot::boot(ids, stat, R = R)
  list(share = dark_share(m, X, outvars), ci = unname(quantile(b$t, c(.025, .975), na.rm = TRUE)),
       boot_se = sd(b$t, na.rm = TRUE), reps = R, fit = tidy_fx(m))
}

# ---- daily panel for section D
DD <- D[user == TRUE & away == FALSE]
DD[, `:=`(adds100 = adds / 100, dels100 = dels / 100)]

# ---- D0 helpers: the common specification
OUT_BT <- c("commits", "large", "direct", "adds100", "dels100", "bug", "task", "feature", "otype")
OUT_CT <- c(OUT_BT, "prs", "revs")
SETS <- list(A = OUT_A, B = OUT_B, Bt = OUT_BT, C = OUT_CT)
quiet_int <- function(expr) withCallingHandlers(expr, warning = function(w)
  if (grepl("non-integer", conditionMessage(w))) invokeRestart("muffleWarning"))
fe_fml <- function(y, outvars, time) as.formula(paste(y, "~", paste(outvars, collapse = " + "), "|", "user_id +", time))
cre_data <- function(X, outvars) {
  X <- copy(X)
  for (v in outvars) X[, (paste0("mb_", v)) := mean(get(v)), by = user_id]
  X
}
cre_fml <- function(y, outvars, time) as.formula(paste(y, "~", paste(c(outvars, paste0("mb_", outvars),
  paste0("factor(", time, ")")), collapse = " + "), "+ (1 | user_id)"))
share_fe <- function(m, X, outvars) {
  U <- X[obs(m)]; U0 <- copy(U)[, (outvars) := 0]
  sum(predict(m, newdata = U0)) / sum(predict(m, newdata = U))
}
share_cre <- function(m, X, outvars, re = NULL) {
  X0 <- copy(X)[, (outvars) := 0]
  sum(predict(m, newdata = X0, type = "response", re.form = re, allow.new.levels = TRUE)) /
    sum(predict(m, newdata = X, type = "response", re.form = re, allow.new.levels = TRUE))
}
# nlminb's default iteration cap stops some of these fits short; raised, it
# reaches the same optimum as BFGS
fit_cre <- function(X, y, outvars, time, fam) quiet_int(glmmTMB(cre_fml(y, outvars, time), data = X, family = fam,
  control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4), parallel = 1L)))
tidy_cre <- function(m, outvars) {
  s <- summary(m)$coefficients$cond
  list(coef = lapply(outvars, function(v) list(term = v, b = s[v, 1], se = s[v, 2], p = s[v, 4])),
       sigma_person = sqrt(VarCorr(m)$cond$user_id[1]), n = nobs(m), converged = m$fit$convergence == 0,
       pdHess = isTRUE(m$sdr$pdHess))
}
# person bootstrap of a share: resample people, keep their periods together,
# relabel repeated people so each copy gets its own person term
boot_share <- function(X, R, f) {
  ids <- unique(X$user_id)
  b <- boot::boot(ids, function(idv, i) {
    draw <- data.table(user_id = idv[i])[, bid := .I]
    B <- merge(draw, X, by = "user_id", allow.cartesian = TRUE)
    B[, user_id := paste0(user_id, "_", bid)]
    tryCatch(f(B), error = function(e) NA_real_)
  }, R = R, parallel = "multicore", ncpus = 8)
  list(ci = unname(quantile(b$t, c(.025, .975), na.rm = TRUE)), se = sd(b$t, na.rm = TRUE),
       reps = R, failed = sum(is.na(b$t)))
}

# ---- D1 helpers: functional forms for the ceiling
FORMS <- c("counts", "log1p", "spline")
form_knots <- function(X) list(k = median(X[commits > 0, commits]), b = c(0, max(X$commits)))
add_form <- function(X, ov, form, kn) {
  X <- copy(X)
  if (form == "counts") return(list(X = X, vars = ov))
  lv <- setdiff(ov, if (form == "spline") "commits" else character(0))
  for (v in lv) X[, (paste0("l_", v)) := log1p(get(v))]
  vars <- if (length(lv)) paste0("l_", lv) else character(0)    # paste0 on an empty vector would return "l_"
  if (form == "spline") {
    B <- splines::ns(X$commits, knots = kn$k, Boundary.knots = kn$b)
    for (j in seq_len(ncol(B))) X[, (paste0("ns", j)) := B[, j]]
    vars <- c(paste0("ns", seq_len(ncol(B))), vars)
  }
  list(X = X, vars = vars)
}
zero_out <- function(X, ov) copy(X)[, (ov) := 0]
pred_pair <- function(m, X, ov, form, kn) {       # predictions at actual and at zero output, estimation rows only
  U <- X[obs(m)]
  list(p1 = predict(m, newdata = add_form(U, ov, form, kn)$X),
       p0 = predict(m, newdata = add_form(zero_out(U, ov), ov, form, kn)$X), U = U)
}
fit_form <- function(X, ov, form, kn, time, est = "pois", y = "tok", extra = character(0)) {
  F <- add_form(X, ov, form, kn)
  f <- as.formula(paste(y, "~", paste(c(F$vars, extra), collapse = " + "), "| user_id +", time))
  switch(est, pois = fepois(f, F$X, cluster = ~user_id), nb = fenegbin(f, F$X, cluster = ~user_id),
         gamma = feglm(f, F$X, family = Gamma(link = "log"), cluster = ~user_id, glm.iter = 500))
}
ceil_share <- function(m, X, ov, form, kn) { pp <- pred_pair(m, X, ov, form, kn); sum(pp$p0) / sum(pp$p1) }
calib <- function(m, X, ov, form, kn) {           # observed / fitted over every unit with all output measures at zero
  pp <- pred_pair(m, X, ov, form, kn); z <- pp$U[, rowSums(.SD) == 0, .SDcols = ov]
  list(ratio = sum(pp$U$tok[z]) / sum(pp$p1[z]), units = sum(z))
}

# Wald test of one coefficient against a value (car), with the model's clustered covariance
test_equal <- function(m, name, value) {
  b <- coef(m); L <- matrix(0, 1, length(b), dimnames = list(NULL, names(b))); L[1, name] <- 1
  lh <- car::linearHypothesis(m, L, rhs = value, test = "Chisq", vcov. = vcov(m), coef. = b)
  list(stat = lh$Chisq[2], p = lh[["Pr(>Chisq)"]][2])
}

# write a results list to a JSON file in the cache directory
write_results <- function(x, name) {
  write_json(x, file.path(DIR, name), auto_unbox = TRUE, digits = NA, pretty = TRUE, na = "null")
  cat("wrote", length(x), "results to", file.path(DIR, name), "\n")
}
