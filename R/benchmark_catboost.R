# make it easy to add new learners to the benchmark!
library(checkpoint)
checkpoint("2020-02-20")

options(java.parameters = "-Xmx16000m") # should avoid java gc overhead

library(OpenML)
# devtools::install_github("mlr-org/mlr")
library(mlr)
library(catboost)
library(lightgbm)


require(devtools)
install_version("liquidSVM", version = "1.2.1")
library(liquidSVM)

load("./data/datasets.RData")
reg = rbind(reg, reg_syn)
source("./R/learners/catboost.R")
source("./R/learners/lightGBM.R")
#source("./R/learners/liquidSVM.R")

lrns = list(
  makeLearner("regr.ranger", num.trees = 500, num.threads = 5),
  #makeLearner("regr.tuneRanger", num.threads = 5, time.budget = 3600),
  makeLearner("regr.catboost", thread_count = 5),
  #makeLearner("regr.lightgbm", nrounds=500, learning_rate=0.1, num_threads=5), # Defaults from their experiment page: https://lightgbm.readthedocs.io/en/latest/Experiments.html
  #makeLearner("regr.xgboost", id="xgboost_def", nrounds=500),
  makeLearner("regr.liquidSVM", threads=5)
  #
)

rdesc <- makeResampleDesc("RepCV", reps=10, folds=5)

bmr = list()

for(i in c(1:nrow(reg))[-c(2,28)]) {
  print(i)
  task <- convertOMLTaskToMlr(getOMLTask(task.id = reg$task.id[i]))$mlr.task
  set.seed(i + 321)
  bmr[[i]] <- benchmark(lrns, task, rdesc, keep.pred = FALSE, models = FALSE, measures = list(mse, mae, medse, medae, rsq, spearmanrho, kendalltau, timetrain))
  save(bmr, file = "./data/bmr_catboost.RData")
}


lrns = list(
  makeLearner("regr.liquidSVM", threads=5)
  #
)

bmr = list()
for(i in c(1:nrow(reg))[-c(2,28)]) {
  print(i)
  task <- convertOMLTaskToMlr(getOMLTask(task.id = reg$task.id[i]))$mlr.task
  set.seed(i + 321)
  bmr_liq[[i]] <- benchmark(lrns, task, rdesc, keep.pred = FALSE, models = FALSE, measures = list(mse, mae, medse, medae, rsq, spearmanrho, kendalltau, timetrain))
  save(bmr, file = "./data/bmr_liquidSVM.RData")
}

lrns = list(
  makeLearner("regr.lightgbm", nrounds=500, learning_rate=0.1, num_threads=5) # Defaults from their experiment page: https://lightgbm.readthedocs.io/en/latest/Experiments.html
)

for(i in c(1:nrow(reg))[-c(2,28)]) {
  print(i)
  task <- convertOMLTaskToMlr(getOMLTask(task.id = reg$task.id[i]))$mlr.task
  set.seed(i + 321)
  bmr[[i]] <- benchmark(lrns, task, rdesc, keep.pred = FALSE, models = FALSE, measures = list(mse, mae, medse, medae, rsq, spearmanrho, kendalltau, timetrain))
  save(bmr, file = "./data/bmr_lightgbm.RData")
}



