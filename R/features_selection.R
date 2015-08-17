## wd etc ####
require(Metrics)
require(caret)
require(readr)
require(doParallel)

## extra functions ####
# print a formatted message
msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

## configure the cluster ####
library(doParallel)
cl <- makeCluster(3);registerDoParallel(cl)

## read data ####
xtrain <- read_csv(file = "../input/train.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL

xtest <- read_csv(file = "./data/test.csv")
id_test <- xtest$ID; xtest$ID <- NULL

## preliminary preparation ####
# drop constant columns
is_missing <- colSums(is.na(xtrain))
constant_columns <- which(is_missing == nrow(xtrain))
xtrain <- xtrain[,-constant_columns]
xtest <- xtest[,-constant_columns]
rm(is_missing, constant_columns)

# check column types
is_missing <- colSums(is.na(xtrain))
is_missing <- which(is_missing > 0)
xtrain[is.na(xtrain)] <- -9999
xtest[is.na(xtest)] <- -9999
col_types <- unlist(lapply(xtrain, class))
fact_cols <- which(col_types == "character")

xtrain_fc <- xtrain[,fact_cols]
xtrain <- xtrain[, -fact_cols]

# SFSG # 

## feature selection ####
# select a good feature to start with
set.seed(20150817)
idFix <- createFolds(y, k = 10, list = T)
relevMat <- array(0, c(length(idFix), ncol(xtrain)))
for (ii in seq(idFix))
{
  idx <- idFix[[ii]]
  relevMat[ii,] <- apply(xtrain[idx,],2,function(s) auc(actual = y[idx], s))
  print(ii)
}
goodFeatures <- which.max(colMeans(relevMat))
goodScore <- mean(relevMat[,goodFeatures])
# goodFeatures: 78
# goodScore: 0.6422847

# iterate over features
# TODO: parallelize! 

relevMat <- array(-10, c(length(idFix), ncol(xtrain)))
for (ii in seq(idFix))
{
  idx <- idFix[[ii]]
  y0 <- y[-idx]; y1 <- y[idx]
  for (jj in 1:ncol(xtrain))
  {
    if (!(jj %in% goodFeatures))
    {
      x0 <- xtrain[-idx, c(jj, goodFeatures)]
      x1 <- xtrain[idx,c(jj, goodFeatures)]
      mod0 <- glmnet(y = y0, x = as.matrix(x0), alpha = 1, family = "binomial")
      pred <- predict(mod0, as.matrix(x1))
      relevMat[ii,jj ] <- Metrics::auc(y1,pred[,ncol(pred)])
      print(jj)
    }
    
  }
}