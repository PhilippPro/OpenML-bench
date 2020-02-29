library(mlr)

#install.packages('devtools')
#devtools::install_url('https://github.com/catboost/catboost/releases/download/v0.21/catboost-R-Windows-0.21.tgz', INSTALL_opts = c("--no-multiarch"))

## Define learner
makeRLearner.classif.catboost <- function() {
  makeRLearnerClassif(
    cl = "classif.catboost",
    package = "catboost",
    par.set = makeParamSet(
      makeNumericLearnerParam(id = "depth", lower = 1, upper = 16, default = 6),
      makeNumericLearnerParam(id = "learning_rate", lower = 0.001, upper = 1, default = 0.03),
      makeNumericLearnerParam(id = "iterations", lower = 1, upper = Inf, default = 1000),
      makeNumericLearnerParam(id = "l2_leaf_reg", lower = 1, upper = Inf, default = 3),
      makeNumericLearnerParam(id = "rsm", lower = 0.001, upper = 1, default = 1),
      makeNumericLearnerParam(id = "border_count", lower = 1, upper = 255, default = 254),
      makeIntegerLearnerParam(id = "thread_count", lower = 1, upper = Inf, default = 1)
    ),
    properties = c("twoclass", "multiclass", "numerics", "factors", "prob"),
    name = "CatBoost",
    short.name = "catboost"
  )
}

## Create training function for the learner
trainLearner.classif.catboost <- function(.learner, .task, .subset, .weights = NULL, ...) {
  task_data <- getTaskData(.task, target.extra = TRUE)
  task_pool <- catboost.load_pool(data = task_data$data, label = as.numeric(levels(task_data$target))[task_data$target])
  par_list <- list()
  par_list$params <- list(...)
  par_list$learn_pool <- task_pool
  par_list$test_pool <- NULL
  do.call(catboost::catboost.train, par_list)
}

## Create prediction method
predictLearner.classif.catboost <- function(.learner, .model, .newdata, ...) {
  test_pool <- catboost.load_pool(data = .newdata)
  if (.learner$predict.type == "response") {
    p_type <- "Class"
  } else if (.learner$predict.type == "prob") {
    p_type <- "Probability"
  } else {
    p_type <- .learner$predict.type
  }
  pred <- catboost::catboost.predict(
    model = .model$learner.model,
    pool = test_pool,
    prediction_type = p_type
  )
  matrix(c(1 - pred, pred), ncol = 2L, dimnames = list(NULL, c(0, 1)))
}

## Register learner
registerS3method("makeRLearner", "classif.catboost", makeRLearner.classif.catboost)
registerS3method("trainLearner", "classif.catboost", trainLearner.classif.catboost)
registerS3method("predictLearner", "classif.catboost", predictLearner.classif.catboost)


####################### Regression #########################

## Define learner
makeRLearner.regr.catboost <- function() {
  makeRLearnerRegr(
    cl = "regr.catboost",
    package = "catboost",
    par.set = makeParamSet(
      makeNumericLearnerParam(id = "depth", lower = 1, upper = 16, default = 6),
      makeNumericLearnerParam(id = "learning_rate", lower = 0.001, upper = 1, default = 0.03),
      makeNumericLearnerParam(id = "iterations", lower = 1, upper = Inf, default = 1000),
      makeIntegerLearnerParam(id = "thread_count", lower = 1, upper = Inf, default = 1)
    ),
    properties = c('numerics','factors','ordered'),
    name = "CatBoost",
    short.name = "catboost"
  )
}

## Create training function for the learner
trainLearner.regr.catboost <- function(.learner, .task, .subset, .weights = NULL, ...) {
  task_data <- getTaskData(.task, target.extra = TRUE)
  task_pool <- catboost.load_pool(data = task_data$data, label = task_data$target)
  par_list <- list()
  par_list$params <- list(...)
  par_list$learn_pool <- task_pool
  par_list$test_pool <- NULL
  do.call(catboost::catboost.train, par_list)
}

## Create prediction method
predictLearner.regr.catboost <- function(.learner, .model, .newdata, ...) {
  test_pool <- catboost.load_pool(data = .newdata)
  pred <- catboost::catboost.predict(
    model = .model$learner.model,
    pool = test_pool
  )
  pred
}

## Register learner
registerS3method("makeRLearner", "regr.catboost", makeRLearner.regr.catboost)
registerS3method("trainLearner", "regr.catboost", trainLearner.regr.catboost)
registerS3method("predictLearner", "regr.catboost", predictLearner.regr.catboost)
