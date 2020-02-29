# https://github.com/ja-thomas/autoxgboost/blob/boosting_backends/R/RLearner_classif_lightgbm.R
# Installation

# https://github.com/microsoft/LightGBM/tree/master/R-package
# 
# install Rtools, CMake, Visual Studio, ...
# cd R
# cd learners
# git clone --recursive https://github.com/microsoft/LightGBM.git
# #git clone -b fix-cat-with-most-freq-bin git@github.com:microsoft/LightGBM.git
# cd LightGBM
# Rscript build_r.R
#
# library(lightgbm)
#
# Default parameters for lightgbm:

library(mlr)
library(BBmisc)

## Define learner
makeRLearner.classif.lightgbm = function() {
  makeRLearnerClassif(
    cl = "classif.lightgbm",
    package = "lightgbm",
    par.set = makeParamSet(
      makeUntypedLearnerParam("early.stopping.data"),
      makeIntegerLearnerParam("nrounds", lower = 1, default = 10),
      makeDiscreteLearnerParam("metric", values = c("map", "auc", "binary_logloss", "binary_error", "multi_logloss", "multi_error")),
      makeIntegerLearnerParam("verbose", lower = -1, upper = 1, tunable = FALSE),
      makeLogicalLearnerParam("record", default = TRUE, tunable = FALSE),
      makeIntegerLearnerParam("eval_freq", lower = 1, tunable = FALSE, requires = quote(verbose > 0)),
      makeUntypedLearnerParam("init_model"),
      makeIntegerLearnerParam("early_stopping_rounds", lower = 1),
      makeDiscreteLearnerParam("boosting", values = c("gbdt", "dart", "rf"), requires = quote(boosting != "rf" || bagging_freq > 0 && bagging_fraction < 1 && feature_fraction < 1)),
      makeNumericLearnerParam("learning_rate", lower = 0, upper = 1, default = 0.1),
      makeIntegerLearnerParam("num_leaves", lower = 1),
      makeDiscreteLearnerParam("tree_learner", values = c("serial", "feature", "data", "voting")),
      makeIntegerLearnerParam("num_threads", lower = 1),
      makeDiscreteLearnerParam("device", values = c("cpu", "gpu")),
      makeIntegerLearnerParam("max_depth", lower = -1),
      makeNumericLearnerParam("min_sum_hessian_in_leaf", lower = 0),
      makeNumericLearnerParam("feature_fraction", lower = 0, upper = 1),
      makeNumericLearnerParam("bagging_fraction", lower = 0, upper = 1),
      makeIntegerLearnerParam("bagging_freq", lower = 0),
      makeNumericLearnerParam("lambda_l1", lower = 0),
      makeNumericLearnerParam("lambda_l2", lower = 0),
      makeNumericLearnerParam("min_split_gain", lower = 0),
      makeNumericLearnerParam("drop_rate", lower = 0, upper = 1, requires = quote(boosting == "dart")),
      makeNumericLearnerParam("skip_drop", lower = 0, upper = 1, requires = quote(boosting == "dart")),
      makeIntegerLearnerParam("max_drop", lower = 1, requires = quote(boosting == "dart")),
      makeLogicalLearnerParam("xgboost_dart_mode"),
      makeIntegerLearnerParam("max_cat_threshold", lower = 0),
      makeNumericLearnerParam("cat_l2", lower = 0),
      makeIntegerLearnerParam("max_cat_to_onehot", lower = 0)
    ),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob", "missings", "factors"),
    name = "Light Gradient Boosting Machine",
    short.name = "lightgbm",
    note = ""
  )
}

trainLearner.classif.lightgbm = function(.learner, .task, .subset, .weights = NULL, early.stopping.data = NULL, metric, ...) {
  
  pv = list(...)
  nc = length(getTaskDesc(.task)$class.levels)
  train = getTaskData(.task, .subset, target.extra = TRUE)
  feat.cols = colnames(train$data)[vlapply(train$data, is.factor)]
  prep = lightgbm::lgb.prepare_rules(train$data)
  pv$data = lightgbm::lgb.Dataset.construct(lightgbm::lgb.Dataset(data.matrix(prep$data), label = as.numeric(train$target) - 1, categorical_feature = feat.cols))
  if (!is.null(early.stopping.data))
    pv$valids = list(test = lightgbm::lgb.Dataset.create.valid(pv$data, data.matrix(early.stopping.data$data), label = as.numeric(early.stopping.data$target) - 1))
  pv$metric = coalesce(metric, "")
  
  if(nc == 2) {
    pv$objective = "binary"
    pv$num_class = 1
  } else {
    pv$objective = "multiclass"
    pv$num_class = nc
  }
  
  mod = do.call(lightgbm::lgb.train, pv)
  return(list(mod = mod, rules = prep$rules))
}

predictLearner.classif.lightgbm = function(.learner, .model, .newdata, ...) {
  td = .model$task.desc
  m = .model$learner.model
  cls = td$class.levels
  nc = length(cls)
  
  .newdata = data.matrix(lightgbm::lgb.prepare_rules(.newdata, rules = m$rules)$data)
  p = predict(m$mod, .newdata)
  
  if (nc == 2) {
    y = matrix(0, ncol = 2, nrow = nrow(.newdata))
    colnames(y) = cls
    y[, 1L] = 1 - p
    y[, 2L] = p
    if (.learner$predict.type == "prob") {
      return(y)
    } else {
      p = colnames(y)[max.col(y)]
      names(p) = NULL
      p = factor(p, levels = colnames(y))
      return(p)
    }
  } else {
    p = matrix(p, nrow = length(p) / nc, ncol = nc, byrow = TRUE)
    colnames(p) = cls
    if (.learner$predict.type == "prob") {
      return(p)
    } else {
      ind = max.col(p)
      cns = colnames(p)
      return(factor(cns[ind], levels = cns))
    }
  }
}

## Register learner
registerS3method("makeRLearner", "classif.lightgbm", makeRLearner.classif.lightgbm)
registerS3method("trainLearner", "classif.lightgbm", trainLearner.classif.lightgbm)
registerS3method("predictLearner", "classif.lightgbm", predictLearner.classif.lightgbm)


####################### Regression #########################

# Define learner
makeRLearner.regr.lightgbm = function() {
  makeRLearnerRegr(
    cl = "regr.lightgbm",
    package = "lightgbm",
    par.set = makeParamSet(
      makeUntypedLearnerParam("early.stopping.data"),
      makeIntegerLearnerParam("nrounds", lower = 1, default = 10),
      makeDiscreteLearnerParam("metric", values = c("l1", "l2", "l2_root", "quantile", "mape", "huber", "fair")),
      makeDiscreteLearnerParam("obj", values = c("regression_l2", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie", default = "regression_l2")),
      makeIntegerLearnerParam("verbose", lower = -1, upper = 1, tunable = FALSE),
      makeLogicalLearnerParam("record", default = TRUE, tunable = FALSE),
      makeIntegerLearnerParam("eval_freq", lower = 1, tunable = FALSE, requires = quote(verbose > 0)),
      makeUntypedLearnerParam("init_model"),
      makeIntegerLearnerParam("early_stopping_rounds", lower = 1),
      makeDiscreteLearnerParam("boosting", values = c("gbdt", "dart", "rf"), requires = quote(boosting != "rf" || bagging_freq > 0 && bagging_fraction < 1 && feature_fraction < 1)),
      makeNumericLearnerParam("learning_rate", lower = 0, upper = 1, default = 0.1),
      makeIntegerLearnerParam("num_leaves", lower = 1),
      makeDiscreteLearnerParam("tree_learner", values = c("serial", "feature", "data", "voting"), default = "serial"),
      makeIntegerLearnerParam("num_threads", lower = 1),
      makeDiscreteLearnerParam("device", values = c("cpu", "gpu"), default = "cpu"),
      makeIntegerLearnerParam("max_depth", lower = -1),
      makeIntegerLearnerParam("min_data_in_leaf", lower = 1),
      makeNumericLearnerParam("min_sum_hessian_in_leaf", lower = 0),
      makeNumericLearnerParam("feature_fraction", lower = 0, upper = 1),
      makeNumericLearnerParam("bagging_fraction", lower = 0, upper = 1),
      makeIntegerLearnerParam("bagging_freq", lower = 0),
      makeNumericLearnerParam("lambda_l1", lower = 0),
      makeNumericLearnerParam("lambda_l2", lower = 0),
      makeNumericLearnerParam("min_split_gain", lower = 0),
      makeNumericLearnerParam("drop_rate", lower = 0, upper = 1, requires = quote(boosting == "dart")),
      makeNumericLearnerParam("skip_drop", lower = 0, upper = 1, requires = quote(boosting == "dart")),
      makeIntegerLearnerParam("max_drop", lower = 0, requires = quote(boosting == "dart")),
      makeLogicalLearnerParam("xgboost_dart_mode", default = FALSE),
      makeIntegerLearnerParam("min_data_per_group", lower = 1),
      makeIntegerLearnerParam("max_cat_threshold", lower = 0),
      makeNumericLearnerParam("cat_l2", lower = 0),
      makeIntegerLearnerParam("max_cat_to_onehot", lower = 0)
    ),
    properties = c("numerics", "weights", "featimp", "missings", "factors"),
    name = "Light Gradient Boosting Machine",
    short.name = "lightgbm",
    par.vals = list(obj = "regression_l2"),
    note = ""
  )
}

trainLearner.regr.lightgbm = function(.learner, .task, .subset, .weights = NULL, early.stopping.data = NULL, metric, ...) {
  
  pv = list(...)
  train = getTaskData(.task, .subset, target.extra = TRUE)
  feat.cols = colnames(train$data)[vlapply(train$data, is.factor)]
  prep = lightgbm::lgb.prepare_rules(train$data)
  pv$data = lightgbm::lgb.Dataset(data.matrix(prep$data), label = as.numeric(train$target), categorical_feature = feat.cols)
  if (!is.null(early.stopping.data))
    pv$valids = list(test = lightgbm::lgb.Dataset.create.valid(pv$data, data.matrix(early.stopping.data$data), label = as.numeric(early.stopping.data$target)))
  pv$metric = coalesce(metric, "")
  
  mod = do.call(lightgbm::lgb.train, pv)
  return(list(mod = mod, rules = prep$rules))
}

predictLearner.regr.lightgbm = function(.learner, .model, .newdata, ...) {
  m = .model$learner.model
  .newdata = data.matrix(lightgbm::lgb.prepare_rules(.newdata, rules = m$rules)$data)
  predict(m$mod, .newdata)
}
## Register learner
registerS3method("makeRLearner", "regr.lightgbm", makeRLearner.regr.lightgbm)
registerS3method("trainLearner", "regr.lightgbm", trainLearner.regr.lightgbm)
registerS3method("predictLearner", "regr.lightgbm", predictLearner.regr.lightgbm)
