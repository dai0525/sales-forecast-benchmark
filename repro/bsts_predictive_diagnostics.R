suppressPackageStartupMessages(library(posterior))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE);folder<-args[1]
for(path in list.files(folder,pattern='^[0-9]+[.]rds$',full.names=TRUE)){
  x<-readRDS(path);m<-nrow(x$predictive_draws[[1]]);n<-length(x$predictive_draws)
  arr<-array(NA_real_,dim=c(m,n,14),dimnames=list(NULL,NULL,c(paste0('h',1:13),'mean13')))
  for(i in seq_len(n))arr[,i,]<-cbind(x$predictive_draws[[i]],rowMeans(x$predictive_draws[[i]]))
  d<-as.data.frame(summarise_draws(as_draws_array(arr),mean,sd,rhat,ess_bulk,ess_tail,mcse_mean))
  write_json(d,sub('[.]rds$','_predictive.json',path),dataframe='rows',auto_unbox=TRUE,pretty=TRUE,digits=16,na='null')
}
