#!/usr/bin/env Rscript

# Minimal Bayesian session-level model using brms.
# Fits a binomial logit model: successes ~ talk_ratio_tokens + student_tokens_sum + (1|domain) + (1|source_dir)
#
# Usage (copy/paste):
#   Rscript scripts/bayes/brms_session_model.R runs/_aggregated/session_view.csv.gz runs/_aggregated/brms_session_model.rds
#

suppressPackageStartupMessages({
  if (!requireNamespace("readr", quietly = TRUE)) install.packages("readr", repos="https://cloud.r-project.org")
  if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr", repos="https://cloud.r-project.org")
  if (!requireNamespace("brms", quietly = TRUE)) install.packages("brms", repos="https://cloud.r-project.org")
})

args <- commandArgs(trailingOnly = TRUE)
in_csv <- if (length(args) >= 1) args[[1]] else "runs/_aggregated/session_view.csv.gz"
out_rds <- if (length(args) >= 2) args[[2]] else "runs/_aggregated/brms_session_model.rds"

library(readr)
library(dplyr)
library(brms)

dir.create(dirname(out_rds), showWarnings = FALSE, recursive = TRUE)

df <- readr::read_csv(in_csv, show_col_types = FALSE)

# Filter to MCQ closed-book runs only (if present)
df <- df %>% filter(is.na(task) | task == "mcq")

# Success counts; guard against NAs
df <- df %>% mutate(
  successes = ifelse(is.na(correct_sum), round(acc_final * steps_n), correct_sum),
  trials = steps_n,
  talk_ratio_tokens = ifelse(is.na(talk_ratio_tokens), 0.5, talk_ratio_tokens),
  student_tokens_sum = ifelse(is.na(student_tokens_sum), 0, student_tokens_sum)
)

# Standardize continuous predictors
df <- df %>% mutate(
  z_talk = scale(talk_ratio_tokens)[,1],
  z_tokens = scale(student_tokens_sum)[,1]
)

formula <- bf(successes | trials(trials) ~ 1 + z_talk + s(z_tokens) + (1|domain) + (1|source_dir), family = binomial())

priors <- c(
  set_prior("normal(0, 1.5)", class = "Intercept"),
  set_prior("normal(0, 1)", class = "b"),
  set_prior("exponential(1)", class = "sd")
)

fit <- brm(
  formula = formula,
  data = df,
  prior = priors,
  iter = 2000,
  warmup = 1000,
  chains = 4,
  cores = max(1, parallel::detectCores() - 1),
  seed = 123,
  backend = "cmdstanr",
  silent = 2,
  refresh = 0
)

saveRDS(fit, out_rds)
cat("Saved:", out_rds, "\n")

summ <- summary(fit)
print(summ)

