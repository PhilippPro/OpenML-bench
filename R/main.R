
source("./R/helper.R")

load("./data/bmr_xgboost_500.RData")
res_1 = extractResults(bmr)
load("./data/bmr_catboost.RData")
res_2 = extractResults(bmr)
load("./data/bmr_lightgbm.RData")
res_3 = extractResults(bmr)

res_2 = excludeResults(res_2, 3)
res = mergeResults(res_1, res_2)
res = mergeResults(res, res_3)

evaluateResults(res)

plotResults(res, 5, ylab = "R-Squared")
plotResults(res, 6, ylab = "Spearman-Rho")
plotResults(res, 7, legend.pos = "bottomright")
plotResults(res, 8, log = TRUE, legend.pos = "bottomright")
plotResults(res, 1, log = TRUE)



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


