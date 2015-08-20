## wd etc ####
require(Metrics)
require(caret)
require(readr)
require(doParallel)
require(glmnet)

## extra functions ####
# print a formatted message
msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

## configure the cluster ####
library(doParallel)
cl <- makeCluster(5);registerDoParallel(cl)

## read data ####
xtrain <- read_csv(file = "./input/train.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL

xtest <- read_csv(file = "./input/test.csv")
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
xtrain[is.na(xtrain)] <- -1
xtest[is.na(xtest)] <- -1
col_types <- unlist(lapply(xtrain, class))
fact_cols <- which(col_types == "character")

xtrain_fc <- xtrain[,fact_cols]
xtrain <- xtrain[, -fact_cols]

# SFSG # 

## feature selection ####
# select a good feature to start with
set.seed(20150817)
idFix <- createFolds(y, k = 5, list = T)
relevMat <- array(0, c(length(idFix), ncol(xtrain)))
for (ii in seq(idFix))
{
  idx <- idFix[[ii]]
  relevMat[ii,] <- apply(xtrain[idx,],2,function(s) Metrics::auc(actual = y[idx], s))
  print(ii)
}
goodFeatures <- which.max(colMeans(relevMat))
goodScore <- mean(relevMat[,goodFeatures])
# goodFeatures: 78
# goodScore: 0.6422847

# iterate over features

while (length(goodFeatures) < 35)
{
  relevMat <- rep(-10, ncol(xtrain))
  for (jj in 1:ncol(xtrain))
  {
    if (!(jj %in% goodFeatures))
    {
      # surprisingly, overhead eats the whole advantage from parallelization
      system.time(a <- foreach(ii = 1:length(idFix), .combine = c, .packages = c("Metrics", "glmnet")) %do%
      {
        idx <- idFix[[ii]]
        y0 <- y[-idx]; y1 <- y[idx]
        x0 <- xtrain[-idx, c(jj, goodFeatures)]
        x1 <- xtrain[idx,c(jj, goodFeatures)]
        mod0 <- glmnet(y = y0, x = as.matrix(x0), alpha = 0, family = "gaussian")
        pred <- predict(mod0, as.matrix(x1))
        Metrics::auc(y1,pred[,ncol(pred)])
      }
      
	  )
	  relevMat[jj] <- mean(a)
      
    }
	if ((jj %% 300) == 0)
	{
		msg(jj)
	}
  }
  goodFeatures <- c(goodFeatures, which.max(relevMat))
  goodScore <- c(goodScore, max(relevMat))
}

rm(y0,y1,a,cl, col_types, fact_cols, idx, ii, is_missing,jj, mod0, pred)
rm(xtrain_fc,x0,x1)

xtrain <- xtrain[,goodFeatures]
xtrain$ID <- id_train
xtest$ID <- id_test
xtrain$target <- y
write_csv(xtrain, path = "./input/xtrain_r1.csv" )
write_csv(xtest, path = "./input/xtest_r1.csv" )


rm(id_test, id_train, idFix, msg, relevMat, xtest, xtrain,y)
save.image("greedy_feature_selection.RData")