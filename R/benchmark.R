options(java.parameters = "-XX:+UseG1GC") # Should avoid java gc overhead
options(java.parameters = "-Xmx16000m")

library(checkpoint)
library(mlr)
library(OpenML)

checkpoint("2019-03-01")

tasks = listOMLTasks(tag = "OpenML-Reg19")
ds = listOMLDataSets(tag = "OpenML-Reg19")

lrns = list(makeLearner("regr.glmnet"), makeLearner("regr.rpart"), makeLearner("regr.kknn")) #, makeLearner("regr.svm"), makeLearner("regr.ranger"), makeLearner("regr.xgboost"))
measures = measures = list(mse, mae, medse, medae, rsq, spearmanrho, kendalltau, timetrain)

bmr = list()
for(i in c(1:nrow(tasks))[-28]) {
  print(i)
  set.seed(123 + i)
  task = getOMLTask(tasks$task.id[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  #rdesc = makeResampleDesc("CV", iters = 2)
  rdesc = makeResampleDesc("RepCV", reps = 10, folds = 5)
  rin = makeResampleInstance(rdesc, task)
  bmr[[i]] = benchmark(lrns, task, rin, measures = measures, keep.pred = TRUE, models = FALSE)
  save(bmr, file = "data/bmr.RData")
}

