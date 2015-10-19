## wd etc ####
library(readr)
library(xgboost)
require(Matrix)
require(caret)
require(stringr)
require(glmnet)
require(gbm)

xseed <- 10
vname <- "v9_r7"
todate <- str_replace_all(str_sub(Sys.time(), 0, 10), "-", "")

set.seed(xseed)

## extra functions ####
# print a formatted message
msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

auc<-function (actual, predicted) {
  
  r <- as.numeric(rank(predicted))
  
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
  
}

buildEnsemble <- function(parVec, xset, yvec)
{
  set.seed(20130912)
  # ensemble settings
  initSize <- parVec[1]; howMany <- parVec[2];
  blendIt <- parVec[3]; blendProp <- parVec[4]
  
  # storage matrix for blending coefficients
  arMat <- array(0, c(blendIt, ncol(xset)))
  colnames(arMat) <- colnames(xset)
  
  # loop over blending iterations
  dataPart <- createDataPartition(1:ncol(arMat), times = blendIt, p  = blendProp)
  for (bb in 1:blendIt)
  {
    idx <- dataPart[[bb]];    xx <- xset[,idx]
    
    # track individual scores
    trackScore <- apply(xx, 2, function(x) auc(yvec,x))
    
    # select the individual best performer - store the performance
    # and create the first column -> this way we have a non-empty ensemble
    bestOne <- which.max(trackScore)
    mastaz <- (rank(-trackScore) <= initSize)
    best.track <- trackScore[mastaz];    hillNames <- names(best.track)
    hill.df <- xx[,mastaz, drop = FALSE]
    
    # loop over adding consecutive predictors to the ensemble
    for(ee in 1 : howMany)
    {
      # add a second component
      trackScoreHill <- apply(xx, 2,
                              function(x) auc(yvec,rowMeans(cbind(x , hill.df))))
      
      best <- which.max(trackScoreHill)
      best.track <- c(best.track, max(trackScoreHill))
      hillNames <- c(hillNames,names(best))
      hill.df <- data.frame(hill.df, xx[,best])
    }
    
    ww <- summary(factor(hillNames))
    arMat[bb, names(ww)] <- ww
  }
  
  wgt <- colSums(arMat)/sum(arMat)
  
  return(wgt)
}

## load and process data ####
xtrain <- read_csv(file = paste("./input/xtrain_",vname,".csv", sep = ""))
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL
xtest <- read_csv(file = paste("./input/xtest_",vname,".csv", sep = ""))
id_test <- xtest$ID; xtest$ID <- NULL

# validation subset: id and target values
xfold <- read_csv(file = "./input/xfolds.csv")
subrange <- which(xfold$valid == 1)
xvalid <- xtrain[subrange,]; y_valid <- y[subrange]; id_valid <- id_train[subrange]
xtrain <- xtrain[-subrange,]; y <- y[-subrange]
rm(xfold, xtrain, y, id_train, vname, xtest, xvalid)

# the validation forecasts
file_list <- dir("./submissions/", pattern = "predValid", full.names = T)
xvalid <- array(0, c(length(id_valid), length(file_list)))
for (ii in 1:ncol(xvalid))
{
  xvalid[,ii] <- read_csv(file_list[[ii]])$target
}

# complete forecasts
file_list2 <- str_replace(file_list, "predValid", "predFull")
xfull <- array(0, c(length(id_test), length(file_list2)))
for (ii in 1:ncol(xvalid))
{
  xfull[,ii] <- read_csv(file_list2[[ii]])$target
}

## build ensemble ####
set.seed(10)
nTimes <- 40
idFix <- createDataPartition(y_valid, times = nTimes, p = 0.25)
xrange <-  seq(from = 0, to = 1, by = 0.05)
storageMat <- array(0, c(nTimes, length(xrange) + 2) )
for (ii in 1:nTimes)
{
  idx <- idFix[[ii]]
  xvalid0 <- xvalid[-idx,]; yvalid0 <- y_valid[-idx]
  xvalid1 <- xvalid[idx,]; yvalid1 <- y_valid[idx]
  
  # transform to rank
  xvalid0 <- apply(xvalid0, 2, rank); xvalid1 <- apply(xvalid1,2, rank)
  xsd <- apply(xvalid0,1,sd)

  xloc <- array(0, c(nrow(xvalid1), length(xrange)))
  
  for (jj in seq(xrange))
  {
    mod0 <- glmnet(x = xvalid0, y = yvalid0, alpha = xrange[jj], weights = sqrt(xsd))
    prx <- predict(mod0, xvalid1); prx1 <- prx[,ncol(prx)]
    storageMat[ii,jj] <- auc(yvalid1, prx1)
    xloc[,jj] <- prx1
  }
  storageMat[ii,length(xrange) + 1] <- auc(yvalid1, rowMeans(xloc))
  storageMat[ii,length(xrange) + 2] <- auc(yvalid1, rowMeans(apply(xloc,2,rank)))
  
  msg(ii)
   
}

# fit complete model
xvalid <- apply(xvalid, 2, rank); xfull <- apply(xfull,2, rank)
xsd <- apply(xvalid,1,sd)
mod0 <- glmnet(x = xvalid, y = y_valid, alpha = 0, weights = sqrt(xsd))
pred <- predict(mod0, xfull); pred2 <- pred[,ncol(pred)]

pred <- pred2
xfor <- data.frame(ID = id_test, target = pred)
xfor$target <- rank(xfor$target)/nrow(xfor)
write_csv(xfor, path = paste("./submissions/ensemble_",todate,".csv", sep = ""))