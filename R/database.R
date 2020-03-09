# Schöne Printstatements einfügen

createDatabase = function(rdesc, filename, overwrite=FALSE) {
  bmr = list()
  lrns = list()
  tasks = data.frame()
  if(file.exists(paste0("./data/benchmark/", filename)) & !overwrite) {
    stop("File is already existing. Specify overwrite=TRUE to overwrite it")
  } else {
    save(bmr, lrns, tasks, rdesc, file = paste0("./data/benchmark/", filename))
  }
}

addDataset = function(new_tasks, filename) {
  load(paste0("./data/benchmark/", filename))
  nr.tasks = nrow(tasks)
  tasks = rbind(tasks, new_tasks)
  if(length(lrns)!=0){
    for(i in c(1:nrow(new_tasks))) {
      print(i)
      new_bmr = executeTasks(lrns, new_tasks[i,], rdesc)
      bmr[[nr.tasks + i]] = new_bmr
      names(bmr)[nr.tasks + i] = new_tasks[i,]$name
      save(bmr, lrns, tasks, rdesc, file = paste0("./data/benchmark/", filename))
    }
  }
  save(bmr, lrns, tasks, rdesc, file = paste0("./data/benchmark/", filename))
}

addLearner = function(new_lrns, rdesc, filename) {
  load(paste0("./data/benchmark/", filename))
  lrns = c(lrns, new_lrns)
  
  if(nrow(tasks)!=0) {
    for(i in c(1:nrow(tasks))) {
      print(i)
      new_bmr = executeTasks(new_lrns, tasks[i,], rdesc)
      if(length(bmr) < i) {
        bmr[[i]] = new_bmr
        names(bmr)[i] = tasks[i,]$name
      } else {
        bmr[[i]] = cbind(bmr[[i]], new_bmr)
      }
      save(bmr, lrns, tasks, rdesc, file = paste0("./data/benchmark/", filename))
    }
  }
  save(bmr, lrns, tasks, rdesc, file = paste0("./data/benchmark/", filename))
}


executeTasks = function(lrns, task_i, rdesc) {
    task <- convertOMLTaskToMlr(getOMLTask(task.id = task_i$task.id))$mlr.task
    if(lrns[[1]]$name == "xgboost") {
      target = task$task.desc$target
      cols = which(colnames(task$env$data) != target)
      task$env$data = data.frame(sapply(dummy.data.frame(task$env$data[,cols], sep = "_._"), as.numeric), 
        task$env$data[,target,drop = F])
      colnames(task$env$data) = make.names(colnames(task$env$data))
      task$task.desc$n.feat[2] = 0
    }
    set.seed(321)
    if(task$type == "regr")
      bmr <- benchmark(lrns, task, rdesc, keep.pred = FALSE, models = FALSE, measures = list(mse, mae, medse, medae, rsq, spearmanrho, kendalltau, timetrain))
    if(task$type == "classif")
      bmr <- benchmark(lrns, task, rdesc, keep.pred = FALSE, models = FALSE, measures = list(acc, multiclass.au1p, multiclass.brier, logloss, timetrain))
    return(data.frame(getBMRAggrPerformances(bmr)))
}


