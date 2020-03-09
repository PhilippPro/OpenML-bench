extractResults = function(filename) {
  load(paste0("./data/benchmark/", filename))
  return(bmr)
}

mergeResults = function(res_1, res_2) {
  if(length(res_1) != length(res_2))
    stop('Results do not have the same length')

  res = list()
  for(i in 1:length(res_1))
    res[[i]] = cbind(res_1[[i]], res_2[[i]])
  
  return(res)
}


evaluateResults = function(res, columns=c(5,6)) {
  nr.learners = ncol(res[[1]])
  for(i in 1:length(res)) {
    if(i == 1) {
      res_aggr = res[[1]]
      res_aggr_rank = apply(res[[1]], 1, rank)
    } else {
      res_aggr = res_aggr + res[[i]]
      res_aggr_rank = res_aggr_rank + apply(res[[i]], 1, rank)
    }
  }
  res_aggr = res_aggr/(length(res))
  res_aggr_rank = res_aggr_rank/(length(res))
  
  
  lrn.names = sub('.*\\.', '', colnames(res_aggr))
  meas.names = sub("\\..*", "", rownames(res_aggr))
  
  tab1 = round(res_aggr[5:6,], 3)
  tab2 = t(round(res_aggr_rank[,5:6], 3))
  colnames(tab1) = colnames(tab2) = lrn.names
  rownames(tab1) = rownames(tab2) = c("R-squared", "Spearman Rho")
  print(tab1)
  print(tab2)
}


excludeResults = function(res, excl_column) {
  for(i in 1:length(res))
    res[[i]] = res[[i]][,-excl_column, drop = FALSE]
  return(res)
}


plotResults = function(res, j, log = FALSE, ylab = NULL, legend.pos = NULL, ylim = NULL) {
  nr.learners = ncol(res[[1]])
  nr.datasets = length(res)
  lrn.names = sub('.*\\.', '', colnames(res[[1]]))
  res_ordered = matrix(NA, nr.datasets, nr.learners)
  res_ordered[1, ] = as.numeric(res[[1]][j, ,drop = FALSE])
  for(i in 1:nr.datasets)
    res_ordered[i, ] = as.numeric(res[[i]][j, ])
  
  res_ordered = res_ordered[order(res_ordered[,1]),, drop = FALSE]
  if(is.null(ylab))
    ylab = sub("\\..*", "", rownames(res[[1]])[j])
  if(is.null(legend.pos))
    legend.pos = "topleft"
  
  if(is.null(ylim))
    ylim = c(min(min(res_ordered),0),max(max(res_ordered),1))
  if(log) {
    plot(res_ordered[,1], type = "l", xlab = paste("Datasets ordered by", ylab, "of ranger"), ylab = ylab, log = "y", lwd = 2, lty = 2, ylim = range(res_ordered))
  } else {
    plot(res_ordered[,1], type = "l", xlab = paste("Datasets ordered by", ylab, "of ranger"), ylab = ylab, ylim = ylim, lwd = 2, lty = 2)
  }
  for(i in 2:nr.learners)
    lines(res_ordered[,i], col = i, lwd = 2)
  #lines(res_ordered[,3], col = "blue", lwd = 3)
  legend(legend.pos, legend = lrn.names, col = 1:nr.learners, lwd = rep(2, nr.learners), lty = c(2,rep(1, nr.learners-1)))
}



