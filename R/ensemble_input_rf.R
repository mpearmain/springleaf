## wd etc ####
library(readr); library(data.table)
library(Matrix); library(caret); require(stringr)
require(glmnet); require(ranger)

## additional functions #### 
# print a formatted message
msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

## cv setup #### 
xfold <- read_csv(file = "./input/xfolds.csv")
isTrain <- which(xfold$valid == 0)
isValid <- which(xfold$valid == 1)
rm(xfold)

# check dimensions
x1 <- read_csv(file = "./input/xtrain_v1_r1.csv")
nr1 <- nrow(x1); id_train <- x1$Id; y <- x1$Hazard
x1 <- read_csv(file = "./input/xtest_v1_r1.csv")
nr2 <- nrow(x1); id_test <- x1$Id
rm(x1)

# setup matrices
xv <- array(0, c(length(isValid),10))
xf <- array(0, c(nr2,10))

## fitting models ####
data_version <- "_v1_r2"
  # read data
  xtrain <- read_csv(file = paste("./input/xtrain",data_version,".csv",sep = ""))
  y <- xtrain$target; xtrain$target <- NULL
  y_train <- factor(y)[isTrain]; y_valid <- y[isValid]
  id_train <- xtrain$ID[isTrain]; id_valid <- xtrain$ID[isValid]
  xtrain$ID <- NULL
  xvalid <- xtrain[isValid,]; xtrain <- xtrain[isTrain,]
  xtest <- read_csv(file = paste("./input/xtest",data_version,".csv",sep = ""))
  id_test <- xtest$ID; xtest$ID <- NULL
  # RF
  rf1 <- ranger(y_train ~ ., data = xtrain, num.trees = 250, 
                probability = T, verbose = T,write.forest = T, seed = 12)
  xv[,1] <- predict(rf1,xvalid)$predictions[,2]
  xf[,1] <- predict(rf1,xtest)$predictions[,2]

data_version <- "_v1_r3"
  # read data
  xtrain <- read_csv(file = paste("./input/xtrain",data_version,".csv",sep = ""))
  y <- xtrain$target; xtrain$target <- NULL
  y_train <- factor(y)[isTrain]; y_valid <- y[isValid]
  id_train <- xtrain$ID[isTrain]; id_valid <- xtrain$ID[isValid]
  xtrain$ID <- NULL
  xvalid <- xtrain[isValid,]; xtrain <- xtrain[isTrain,]
  xtest <- read_csv(file = paste("./input/xtest",data_version,".csv",sep = ""))
  id_test <- xtest$ID; xtest$ID <- NULL
  # RF
  rf1 <- ranger(y_train ~ ., data = xtrain, num.trees = 250, 
                probability = T, verbose = T,write.forest = T, seed = 12)
  xv[,2] <- predict(rf1,xvalid)$predictions[,2]
  xf[,2] <- predict(rf1,xtest)$predictions[,2]

data_version <- "_v1_r1"
  # read data
  xtrain <- read_csv(file = paste("./input/xtrain",data_version,".csv",sep = ""))
  y <- xtrain$target; xtrain$target <- NULL
  y_train <- factor(y)[isTrain]; y_valid <- y[isValid]
  id_train <- xtrain$ID[isTrain]; id_valid <- xtrain$ID[isValid]
  xtrain$ID <- NULL
  xvalid <- xtrain[isValid,]; xtrain <- xtrain[isTrain,]
  xtest <- read_csv(file = paste("./input/xtest",data_version,".csv",sep = ""))
  id_test <- xtest$ID; xtest$ID <- NULL
  # RF
  rf1 <- ranger(y_train ~ ., data = xtrain, num.trees = 400, 
                probability = T, verbose = T,write.forest = T, seed = 12)
  xv[,3] <- predict(rf1,xvalid)$predictions[,2]
  xf[,3] <- predict(rf1,xtest)$predictions[,2]
  
  
  xgtrain <- xgb.DMatrix(data = as.matrix(xtrain), label= (y_train == 1)+ 0)
  xgval <-  xgb.DMatrix(data = as.matrix(xvalid), label= y_valid)
  watchlist <- list(val=xgval, train=xgtrain)
  
  # configure parameters
  param <- list(objective   = "binary:logistic",
                eval_metric = "auc",
                "eta" = 0.025,
                "min_child_weight" = 8,
                "subsample" = .5, "colsample_bytree" = .5,
                "max_depth" = 15,  "gamma" = 0.075, "silent" = 0)
  # fit the xgb
  clf <- xgb.train(params = param, data = xgtrain, 
                   nround=800, print.every.n = 25, watchlist=watchlist, 
                   early.stop.round = 50, maximize = TRUE)
  xv[,4] <- predict(clf, xgval)
data_version <- "_v1"
  # read data
  xtrain <- read_csv(file = paste("./input/xtrain",data_version,".csv",sep = ""))
  y <- xtrain$target; xtrain$target <- NULL
  y_train <- factor(y)[isTrain]; y_valid <- y[isValid]
  id_train <- xtrain$ID[isTrain]; id_valid <- xtrain$ID[isValid]
  xtrain$ID <- NULL
  xvalid <- xtrain[isValid,]; xtrain <- xtrain[isTrain,]
  xtest <- read_csv(file = paste("./input/xtest",data_version,".csv",sep = ""))
  id_test <- xtest$ID; xtest$ID <- NULL
  # RF
  rf1 <- ranger(y_train ~ ., data = xtrain, num.trees = 250, 
                probability = T, verbose = T,write.forest = T, seed = 12)
  xv[,4] <- predict(rf1,xvalid)$predictions[,2]
  xf[,4] <- predict(rf1,xtest)$predictions[,2]