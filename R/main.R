library(checkpoint)
checkpoint("2020-03-01")
# lightgbm package has to be put manually into checkpoint folder
source("./R/loadPackages.R")
source("./R/database.R")
source("./R/helper.R")

# mit GPU experimentieren bei catboost
# tuneRanger

####################################################################################################
###################################### Regression ##################################################
####################################################################################################

lrns = list(
  makeLearner("regr.ranger", num.trees = 500, num.threads = 5),
  makeLearner("regr.catboost", thread_count = 5),
  makeLearner("regr.liquidSVM", threads=5),
  makeLearner("regr.lightgbm", nrounds=500, learning_rate=0.1, num_threads=5), # Defaults from their experiment page: https://lightgbm.readthedocs.io/en/latest/Experiments.html
  makeDummyFeaturesWrapper(makeLearner("regr.xgboost", id="xgboost_def", nrounds=500, nthread=5)),
  makeDummyFeaturesWrapper(makeLearner("regr.xgboost", id="xgboost_best", nrounds=500, eta=0.0518715, subsample=0.8734055, booster="gbtree", 
    max_depth=11, min_child_weight=1.750185, colsample_bytree=0.7126651, colsample_bylevel=0.6375492, nthread=5))
)
lrns[[5]]$id = "xgboost_def"
lrns[[6]]$id = "xgboost_best"

library(OpenML)
tasks = listOMLTasks(tag = "OpenML-Reg19")
tasks = tasks[!(tasks$name %in% c("aloi", "BNG(satellite_image)", "black_friday")),] # black_friday später hinzunehmen

rdesc = makeResampleDesc("CV", iters=5)
createDatabase(rdesc, filename="regression_CV5.RData")
addDataset(new_tasks=tasks, filename="regression_CV5.RData")
addLearner(new_lrns=lrns, filename="regression_CV5.RData")

rdesc = makeResampleDesc("RepCV", folds=5, reps=10)
createDatabase(rdesc, filename="regression_RepCV5_10.RData")
addDataset(new_tasks=tasks, filename="regression_RepCV5_10.RData")
addLearner(new_lrns=lrns, filename="regression_RepCV5_10.RData")

# Get the results and evaluate them
res = extractResults(filename = "regression_RepCV5_10.RData")
evaluateResults(res)

res = excludeResults(res, 3)

plotResults(res, 5, ylab = "R-Squared", legend.pos = "bottomright")
plotResults(res, 6, ylab = "Spearman-Rho", ylim = c(0,1), legend.pos = "bottomright")
plotResults(res, 7, ylim = c(0,1), legend.pos = "bottomright")
plotResults(res, 8, ylab = "Training time in seconds", log = TRUE, legend.pos = "bottomright")
plotResults(res, 1, log = TRUE)


pdf("./figure/rsq_results.pdf", height = 5)
par(mar = c(4, 4, 1, 2) + 0.1)
plotResults(res, 5, ylab = "R-Squared", legend.pos = "bottomright")
dev.off()

pdf("./figure/spearman_results.pdf", height = 5)
par(mar = c(4, 4, 1, 2) + 0.1)
plotResults(res, 6, ylab = "Spearman-Rho", ylim = c(0,1), legend.pos = "bottomright")
dev.off()

pdf("./figure/time_results.pdf", height = 4)
par(mar = c(4, 4, 1, 2) + 0.1)
plotResults(res, 8, ylab = "Training time in seconds", log = TRUE, legend.pos = "bottomright")
dev.off()

# For the blog
png("./figure/rsq_results.png", height = 400)
par(mar = c(4, 4, 1, 2) + 0.1)
plotResults(res, 5, ylab = "R-Squared", legend.pos = "bottomright")
dev.off()

png("./figure/spearman_results.png", height = 400)
par(mar = c(4, 4, 1, 2) + 0.1)
plotResults(res, 6, ylab = "Spearman-Rho", ylim = c(0,1), legend.pos = "bottomright")
dev.off()

png("./figure/time_results.png", height = 400)
par(mar = c(4, 4, 1, 2) + 0.1)
plotResults(res, 8, ylab = "Training time in seconds", log = TRUE, legend.pos = "bottomright")
dev.off()



####################################################################################################
###################################### Classification ##############################################
####################################################################################################

lrns = list(
  makeLearner("classif.ranger", num.trees=500, num.threads=5, predict.type="prob"),
  #makeLearner("classif.catboost", thread_count = 5, predict.type="prob"),
  #makeLearner("classif.liquidSVM", threads=5, predict.type="prob"),
  makeLearner("classif.lightgbm", nrounds=500, learning_rate=0.1, num_threads=5, predict.type="prob"), # Defaults from their experiment page: https://lightgbm.readthedocs.io/en/latest/Experiments.html
  makeDummyFeaturesWrapper(makeLearner("classif.xgboost", id="xgboost_def", nrounds=500, nthread=5, predict.type="prob")),
  makeDummyFeaturesWrapper(makeLearner("classif.xgboost", id="xgboost_best", nrounds=500, eta=0.0518715, subsample=0.8734055, booster="gbtree", 
    max_depth=11, min_child_weight=1.750185, colsample_bytree=0.7126651, colsample_bylevel=0.6375492, nthread=5, predict.type="prob"))
)
lrns[[3]]$id = "xgboost_def"
lrns[[4]]$id = "xgboost_best"

tasks = listOMLTasks(tag = "OpenML-CC18")
tasks = tasks[tasks$number.of.missing.values == 0,]
#tasks = tasks[!(tasks$name %in% c("breast-w", "credit-approval", "eucalyptus", "sick")),] # blmissing valuesack_friday später hinzunehmen

rdesc = makeResampleDesc("CV", iters=5)
createDatabase(rdesc, filename="classification_CV5.RData", overwrite = TRUE)
addDataset(new_tasks=tasks, filename="classification_CV5.RData")
addLearner(new_lrns=lrns, filename="classification_CV5.RData")

rdesc = makeResampleDesc("RepCV", folds=5, iters = 10)
createDatabase(rdesc, filename="classification_RepCV5_10.RData")
addDataset(new_tasks=tasks, filename="classification_RepCV5_10.RData")
addLearner(new_lrns=lrns, filename="classification_RepCV5_10.RData")

# Get the results and evaluate them
res = extractResults(filename = "classification_CV5.RData")
evaluateResults(res)

plotResults(res, 1, log = FALSE, legend.pos = "bottomright")
plotResults(res, 2, log = FALSE, legend.pos = "bottomright", ylab = "Multiclass AUC", ylim = c(0.5, 1))
plotResults(res, 3, log = FALSE, legend.pos = "bottomright", ylab = "Multiclass Brier Score")
plotResults(res, 4, log = FALSE, legend.pos = "topleft", ylab = "Logarithmic Loss")
plotResults(res, 5, log = FALSE, legend.pos = "bottomright", ylab = "Training time in seconds")
plotResults(res, 5, log = TRUE, legend.pos = "bottomright", ylab = "Training time in seconds")

#
# gradient boosting algos better calibrated for probability (brier score)
# especially xgboost_best for logloss
# catboost is missing!










# Anhang



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


