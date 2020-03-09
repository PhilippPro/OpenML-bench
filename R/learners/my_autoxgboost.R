options(java.parameters = "-Xmx16000m") # should avoid java gc overhead

library(OpenML)
# devtools::install_github("mlr-org/mlr")
library(mlr)
# devtools::install_github("ja-thomas/autoxgboost", dependencies = FALSE)
library(autoxgboost)

# Das hier ist leider auch schlecht (nicht immer besser als autoxgbparset). 
# Liegt wohl an der Implementation von autoxgboost und nicht unbedingt an den Hyperparameter Settings. 
my_xgb = makeParamSet(
   # makeIntegerParam("nrounds", lower = 800, upper = 5000), 
    makeNumericParam("eta", lower = -9, upper = -1, trafo = function(x) 2^x),
    makeNumericParam("subsample",lower = 0.45, upper = 0.96),
    makeDiscreteParam("booster", values = c("gbtree", "gblinear")),
    makeIntegerParam("max_depth", lower = 2, upper = 15, requires = quote(booster == "gbtree")),
    makeNumericParam("min_child_weight", lower = 1, upper = 7, requires = quote(booster == "gbtree")),
    makeNumericParam("colsample_bytree", lower = 0.4, upper = 0.9, requires = quote(booster == "gbtree")),
    makeNumericParam("colsample_bylevel", lower = 0.3, upper = 0.9, requires = quote(booster == "gbtree")),
    makeNumericParam("lambda", lower = -8, upper = 5, trafo = function(x) 2^x),
    makeNumericParam("alpha", lower = -9, upper = 3, trafo = function(x) 2^x)
)

#autoxgbparset

lrns = list(
  makeLearner("regr.autoxgboost", id = "myxgb", nthread = 5, par.set = my_xgb)#, max.nrounds = 5000)
)


# if less than 20 percent NA, impute by the mean of the other iterations
# for(i in seq_along(bmr_tune)[-c(2, 28)]) {
#   for(j in 1:nr.learners) {
#     print(paste(i,j))
#     for(k in 2:nr.measures) {
#       na.percentage = mean(is.na(bmr_tune[[i]]$results[[1]][[j]]$measures.test[k]))
#       if(na.percentage > 0 & na.percentage <= 0.2) {
#         resis = unlist(bmr_tune[[i]]$results[[1]][[j]]$measures.test[k])
#         bmr_tune[[i]]$results[[1]][[j]]$aggr[k-1] = mean(resis, na.rm = T)
#       }
#     }
#   }
# }
