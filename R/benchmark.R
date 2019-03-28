library(checkpoint)
library(mlr)
library(OpenML)

checkpoint("2019-03-01")

ds = listOMLDataSets()
tasks = listOMLTasks(tag = "OpenML-Reg19")

tasks = listOMLDataSets(tag = "OpenML-Reg19")

set.seed(1234)

lrns <- list(makeLearner("regr.glmnet"), makeLearner("regr.IBk"), makeLearner("regr.kknn"), makeLearner("regr.rpart"), makeLearner("regr.ranger"))


for(i in 1:nrow(tasks)) {
  set.seed(123)
  task = getOMLTask(tasks$task.id[i])
  task = convertOMLTaskToMlr(task)$mlr.task
  rdesc = makeResampleDesc("CV", iters = 2)
  rdesc = makeResampleDesc("RepCV", reps = 10, folds = 5)
  bmr[[i]] = benchmark(lrns, task, rdesc)
}
