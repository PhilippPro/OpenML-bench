options(java.parameters = "-XX:+UseG1GC") # Should avoid java gc overhead
options(java.parameters = "-Xmx16000m")

library(OpenML)
library(mlr)
library(catboost)
library(lightgbm)
library(devtools)
library(liquidSVM)
library(dummies)

load("./data/datasets.RData")
reg = rbind(reg, reg_syn)
source("./R/learners/catboost.R")
source("./R/learners/lightGBM.R")
#source("./R/learners/liquidSVM.R")