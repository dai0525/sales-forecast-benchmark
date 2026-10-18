# Original CRAN bsts package; parameter and state posterior sampling.
suppressPackageStartupMessages(library(bsts))
suppressPackageStartupMessages(library(posterior))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
root<-if(length(args)>0) args[1] else '.'
kind<-if(length(args)>1) args[2] else 'trend_harmonic'
niter<-if(length(args)>2) as.integer(args[3]) else 1000L
workers<-if(length(args)>3) as.integer(args[4]) else 2L
limit<-if(length(args)>4) as.integer(args[5]) else 0L
prior_mode<-if(length(args)>5) args[6] else 'default'
folder<-file.path(root,'audit/results/bsts_mcmc')
yall<-as.matrix(read.csv(file.path(folder,'train.csv'),header=FALSE))
ids<-fromJSON(file.path(folder,'ids.json'))
use_harmonic<-grepl('harmonic',kind)
indices<-if(kind=='trend_seasonal52' || grepl('spaced12',kind)) unique(as.integer(round(seq(1,nrow(yall),length.out=12)))) else seq_len(nrow(yall))
if(limit>0) indices<-head(indices,limit)
outdir<-file.path(folder,paste0(kind,'_',prior_mode,'_',niter));dir.create(outdir,recursive=TRUE,showWarnings=FALSE)
protocol<-list(package=as.character(packageVersion('bsts')),R=R.version.string,posterior=as.character(packageVersion('posterior')),kind=kind,indices=indices,ids=ids[indices],niter=niter,burn=niter%/%2,chains=2,horizon=13,seed_rule='20261018 + series_index*100 + chain',priors=prior_mode,prior_note='default: original package defaults; regularized: innovation SdPrior(.01*sd(y),sample.size=32, upper.limit=sd(y)), observation prior remains default; all scales from train only.',point_forecast='Mean of posterior predictive distribution, then clip at zero in Python',diagnostics='rank-normalized split Rhat, bulk/tail ESS for sampled SDs and final level/slope; target Rhat<=1.01 and ESS>=400',covariates='None in M4. No fabricated regression predictors; spike-and-slab regression inapplicable.',model=if(use_harmonic) 'AddLocalLinearTrend + AddTrig(period=52, frequencies=1:4, method=harmonic)' else 'AddLocalLinearTrend + AddSeasonal(nseasons=52)',workers=workers)
# Fail closed on stale checkpoints. Only explicitly audited legacy runs can be adopted.
signature<-list(train_md5=unname(tools::md5sum(file.path(folder,'train.csv'))),ids_md5=unname(tools::md5sum(file.path(folder,'ids.json'))),source_md5=unname(tools::md5sum(file.path(root,'repro/bench_bsts_mcmc.R'))),kind=kind,niter=as.character(niter),prior=prior_mode,seed='20261018+i*100+chain',bsts=as.character(packageVersion('bsts')),R=R.version.string)
old_protocol_path<-file.path(outdir,'protocol.json')
if(file.exists(old_protocol_path) && length(list.files(outdir,pattern='^[0-9]+[.]json$'))>0){
 old_protocol<-read_json(old_protocol_path,simplifyVector=TRUE)
 if(is.null(old_protocol$cache_signature)){
  if(Sys.getenv('BSTS_ADOPT_LEGACY')!='1')stop('Legacy BSTS cache requires explicit audited adoption; use a fresh output directory otherwise')
  if(!identical(as.character(old_protocol$kind),kind) || old_protocol$niter!=niter || !identical(as.character(old_protocol$priors),prior_mode) || !identical(as.character(old_protocol$ids),as.character(ids[indices])))stop('Legacy cache configuration/ID mismatch')
  protocol$legacy_adoption_note<-'Adopted after ID, training-input and inference-configuration audit; source hash below is the current guarded implementation, not a retroactive assertion of identical original diagnostic code.'
 } else if(!identical(old_protocol$cache_signature,signature))stop('BSTS cache fingerprint mismatch: training inputs, configuration, source or software changed')
}
protocol$cache_signature<-signature
write_json(protocol,file.path(outdir,'protocol.json'),pretty=TRUE,auto_unbox=TRUE)
plain<-function(x){ if(is.list(x)) return(lapply(unclass(x),plain)); if(is.object(x)) return(unclass(x)); x }
run_one<-function(i){
 dest<-file.path(outdir,sprintf('%03d.json',i))
 if(file.exists(dest)){
  cached<-read_json(dest,simplifyVector=TRUE)
  if(!isTRUE(cached$ok) || !identical(as.character(cached$id),as.character(ids[i])) || !file.exists(file.path(outdir,sprintf('%03d.rds',i))))stop('Invalid, failed or incomplete BSTS checkpoint')
  return(list(index=i,cached=TRUE))
 }
 started<-proc.time()[['elapsed']];y<-as.numeric(yall[i,]);chains<-list();draws<-list();warnings_seen<-character()
 ans<-tryCatch(withCallingHandlers({
  for(chain in 1:2){
   seed<-20261018L+i*100L+chain;set.seed(seed)
   innovation_prior<-if(prior_mode=='regularized') SdPrior(.01*sd(y),sample.size=32,upper.limit=sd(y)) else NULL
   ss<-AddLocalLinearTrend(list(),y,level.sigma.prior=innovation_prior,slope.sigma.prior=innovation_prior)
   ss<-if(use_harmonic) AddTrig(ss,y,period=52,frequencies=1:4,method='harmonic',sigma.prior=innovation_prior) else AddSeasonal(ss,y,nseasons=52,sigma.prior=innovation_prior)
   model<-bsts(y,state.specification=ss,niter=niter,ping=0,seed=seed,model.options=BstsOptions(save.state.contributions=FALSE,save.prediction.errors=FALSE))
   pred<-predict(model,horizon=13,burn=niter%/%2,seed=seed+1000000L)
   keep<-seq.int(niter%/%2+1,niter)
   trace<-cbind(sigma.obs=model$sigma.obs[keep],sigma.level=model$sigma.trend.level[keep],sigma.slope=model$sigma.trend.slope[keep],final.level=model$final.state[keep,1],final.slope=model$final.state[keep,2])
   if(use_harmonic){
    tr<-as.matrix(model$trig.coefficient.sd.52)[keep,,drop=FALSE];colnames(tr)<-paste0('sigma.trig.',seq_len(ncol(tr)));trace<-cbind(trace,tr)
   }else trace<-cbind(trace,sigma.seasonal=model$sigma.seasonal.52[keep])
   trace<-cbind(trace,predictive.mean13=rowMeans(pred$distribution),predictive.h1=pred$distribution[,1],predictive.h13=pred$distribution[,13])
   chains[[chain]]<-trace;draws[[chain]]<-pred$distribution
  }
  arr<-array(NA_real_,dim=c(nrow(chains[[1]]),2,ncol(chains[[1]])),dimnames=list(NULL,NULL,colnames(chains[[1]])))
  for(c in 1:2)arr[,c,]<-chains[[c]]
  diagnostics<-as.data.frame(summarise_draws(as_draws_array(arr),mean,sd,rhat,ess_bulk,ess_tail,mcse_mean))
  dist<-do.call(rbind,draws)
  chainmean<-do.call(rbind,lapply(draws,colMeans))
  res<-list(index=i,id=ids[i],ok=TRUE,mean=colMeans(dist),median=apply(dist,2,median),lower=apply(dist,2,quantile,probs=.025),upper=apply(dist,2,quantile,probs=.975),chain_means=chainmean,diagnostics=diagnostics,runtime_s=proc.time()[['elapsed']]-started,warnings=unique(warnings_seen),priors=list(observation=model$prior,state=model$state.specification))
  saveRDS(list(parameter_traces=arr,predictive_draws=draws),file.path(outdir,sprintf('%03d.rds',i)),compress='gzip')
  res
 },warning=function(w){warnings_seen<<-c(warnings_seen,conditionMessage(w));invokeRestart('muffleWarning')}),error=function(e)list(index=i,id=ids[i],ok=FALSE,error=conditionMessage(e),runtime_s=proc.time()[['elapsed']]-started))
 write_json(plain(ans),dest,pretty=TRUE,auto_unbox=TRUE,digits=16,na='null')
 cat(sprintf('%s %03d/%d %s %.2fs\n',kind,i,nrow(yall),if(ans$ok)'OK' else ans$error,ans$runtime_s));flush.console()
 ans[c('index','ok','runtime_s')]
}
started<-proc.time()[['elapsed']]
res<-parallel::mclapply(indices,run_one,mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
write_json(list(wall_s=proc.time()[['elapsed']]-started,series=res,session=capture.output(sessionInfo())),file.path(outdir,'run.json'),pretty=TRUE,auto_unbox=TRUE)
