source("./R/learners/catboost.R")

# make it easy to add new learners to the benchmark!
# include lightgbm, ...

options(java.parameters = "-Xmx16000m") # should avoid java gc overhead

library(OpenML)
# devtools::install_github("mlr-org/mlr")
library(mlr)
library(catboost)

load("./data/datasets.RData")
reg = rbind(reg, reg_syn)
source("./R/learners/catboost.R")

bmr_tune <- list()
lrns = list(
  makeLearner("regr.catboost")
  #
)

rdesc <- makeResampleDesc("CV", iters= 5)

for(i in c(1:nrow(reg))[-c(2,28)]) {
  print(i)
  task <- convertOMLTaskToMlr(getOMLTask(task.id = reg$task.id[i]))$mlr.task
  set.seed(i + 321)
  bmr[[i]] <- benchmark(lrns, task, rdesc, keep.pred = FALSE, models = FALSE, measures = list(mse, mae, medse, medae, rsq, spearmanrho, kendalltau, timetrain))
  save(bmr, file = "./data/bmr_catboost.RData")
}


# Analysis
load("./data/bmr_catboost.RData")

nr.learners = length(bmr[[1]]$learners)
nr.measures = length(bmr[[1]]$measures)
bmr[[2]] = NULL
bmr[[27]] = NULL

nr.learners = length(bmr[[1]]$learners)
resi = list()
resi[[1]] = data.frame(getBMRAggrPerformances(bmr[[1]]))

for(i in 1:length(bmr)) {
  if(!is.null(bmr[[i]])) {
    resi[[i]] = data.frame(getBMRAggrPerformances(bmr[[i]]))
    for(j in 1:nr.learners) {
      print(paste(i,j))
      for(k in 1:8) {
        if(is.na(resi[[i]][k,j])) {
          if(k %in% c(1:4, 8)) {
            resi[[i]][k,j] = max(resi[[i]][k,], na.rm = T)
          } else {
            resi[[i]][k,j] = min(resi[[i]][k,], na.rm = T)
          }
        }
      }
    }
    if(i == 1) {
      res_aggr = resi[[1]]
      res_aggr_rank = apply(resi[[1]], 1, rank)
    } else {
      res_aggr = res_aggr + resi[[i]]
      res_aggr_rank = res_aggr_rank + apply(resi[[i]], 1, rank)
    }
  }
}
res_aggr = res_aggr/(length(bmr))
res_aggr_rank = res_aggr_rank/(length(bmr))

#for(i in 3:length(bmr)) {
#  resi[[i]] = round(cbind(resi[[i]], data.frame(getBMRAggrPerformances(bmr_autoxgboost[[i]]))), 4)
#  resi[[i]] = round(cbind(resi[[i]], data.frame(getBMRAggrPerformances(bmr_liquidSVM[[i]]))), 4)
#}

lrn.names = sub('.*\\.', '', colnames(res_aggr))
meas.names = sub("\\..*", "", rownames(res_aggr))

tab1 = round(res_aggr[5:6,], 3)
tab2 = t(round(res_aggr_rank[,5:6], 3))

colnames(tab1) = colnames(tab2) = lrn.names
rownames(tab1) = rownames(tab2) = c("R-squared", "Spearman Rho")

library(xtable)
xtable(tab1)
xtable(tab2)

# R-squared

plot_results = function(j, log = FALSE, ylab = NULL, legend.pos = NULL) {
  ranger_res = matrix(NA, length(resi), ncol(resi[[1]]))
  ranger_res[1, ] = as.numeric(resi[[1]][j, ])
  for(i in 1:length(resi))
    ranger_res[i, ] = as.numeric(resi[[i]][j, ])
  
  ranger_res = ranger_res[order(ranger_res[,1]),]
  if(is.null(ylab))
    ylab = sub("\\..*", "", rownames(resi[[1]])[j])
  if(is.null(legend.pos))
    legend.pos = "topleft"
  if(log) {
    plot(ranger_res[,1], type = "l", xlab = paste("Datasets ordered by", ylab, "of ranger"), ylab = ylab, log = "y", lwd = 2, lty = 2, ylim = range(ranger_res))
  } else {
    plot(ranger_res[,1], type = "l", xlab = paste("Datasets ordered by", ylab, "of ranger"), ylab = ylab, ylim = c(-0.05,1), lwd = 2, lty = 2)
  }
  lines(ranger_res[,2], col = "red", lwd = 2)
  legend(legend.pos, legend = lrn.names, col = c("black", "red"), lwd = c(2,2), lty = c(2,1))
}

plot_results(5, ylab = "R-Squared")
plot_results(6, ylab = "Spearman-Rho")
plot_results(7)
plot_results(8, log = TRUE)
plot_results(1, log = TRUE)



pdf("./figure/rsq_results.pdf", height = 4)
par(mar = c(4, 4, 1, 2) + 0.1)
plot_results(5, ylab = "R-Squared")
dev.off()

pdf("./figure/spearman_results.pdf", height = 4)
par(mar = c(4, 4, 1, 2) + 0.1)
plot_results(6, ylab = "Spearmans-Rho", legend.pos = "bottomright")
dev.off()

pdf("./figure/time_results.pdf", height = 4)
par(mar = c(4, 4, 1, 2) + 0.1)
plot_results(8, ylab = "Training time in seconds", legend.pos = "bottomright", log = TRUE)
dev.off()

# mit seed nochmal neu laufen lassen
# Ergebnisse mit Janek besprechen; insbesondere Laufzeitbeschränkung...
