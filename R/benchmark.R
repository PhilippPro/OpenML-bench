options(java.parameters = "-XX:+UseG1GC") # Should avoid java gc overhead
options(java.parameters = "-Xmx16000m")

library(checkpoint)
library(mlr)
library(OpenML)

checkpoint("2019-03-01")

tasks = listOMLTasks(tag = "OpenML-Reg19")
ds = listOMLDataSets(tag = "OpenML-Reg19")

lrns = list(makeLearner("regr.glmnet"), makeLearner("regr.rpart"), makeLearner("regr.kknn")) #, makeLearner("regr.svm"), makeLearner("regr.ranger"), makeLearner("regr.xgboost"))
measures = measures = list(mse, mae, medse, medae, rsq, spearmanrho, kendalltau, timetrain, timepredict)

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

bmr

# Analysis

results = data.frame("task.id" = character(), "learner.id" = character(), "rsq.test.mean" = double(), "kendalltau.test.mean" = double(), "timetrain.test.mean" = double())

for (i in c(1:length(bmr))[-28]){
  resi = getBMRAggrPerformances(bmr[[i]], as.df = TRUE)
  results = rbind(results, resi[, colnames(results)])
}

### Set Rsquared to 0 if smaller than 0 
# results[results[, 3]<=0, 3] <- 0 

### Set Zero if Kendalls Tau is NA
# results[is.na(results[,4]) == TRUE,4] <- 0

names(results)[2] <- "Learner"

library(ggplot2)
ggplot(data = results , aes(y = task.id, x = rsq.test.mean, group = Learner, color= Learner)) +
  geom_point() +
  lims(x=c(0,1))+
  ylab( "Dataset") +
  xlab("R-squared") #+
  #theme(plot.margin = unit(c(5,0,5,0),"cm"))#

library(ggplot2)
ggplot(data = results , aes(y = task.id, x = kendalltau.test.mean, group = Learner, color= Learner)) +
  geom_point() +
  lims(x=c(0,1))+
  ylab( "Dataset") +
  xlab("Kendall's Tau") #+
#theme(plot.margin = unit(c(5,0,5,0),"cm"))

library(ggplot2)
ggplot(data = results , aes(y = task.id, x = timetrain.test.mean, group = Learner, color= Learner)) +
  geom_point() +
  lims(x=c(0,1))+
  ylab( "Dataset") +
  xlab("Training time") #+
#theme(plot.margin = unit(c(5,0,5,0),"cm"))

# Grafik wie beim Vergleich der Tuningmethoden zur besseren Uebersicht

