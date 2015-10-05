## wd etc ####
library(readr)
library(xgboost)
require(Matrix)
require(caret)

xseed <- 123
vname <- "v9_r7"

set.seed(xseed)

## load and process data ####
xtrain <- read_csv(file = paste("./input/xtrain_",vname,".csv", sep = ""))
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL
xtest <- read_csv(file = paste("./input/xtest_",vname,".csv", sep = ""))
id_test <- xtest$ID; xtest$ID <- NULL

## xgboost  ####
#Set xgboost test and training and validation datasets
xfold <- read_csv(file = "./input/xfolds.csv")
subrange <- which(xfold$valid == 1)
xvalid <- xtrain[subrange,]; y_valid <- y[subrange]; id_valid <- id_train[subrange]
xtrain <- xtrain[-subrange,]; y <- y[-subrange]
xgtrain <- xgb.DMatrix(data = as.matrix(xtrain), label= y)
xgval <-  xgb.DMatrix(data = as.matrix(xvalid), label= y_valid)
watchlist <- list(val=xgval)
rm(xtrain, xvalid)

# configure parameters
param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              "eta" = 0.01,
              "min_child_weight" = 8,
              "subsample" = .5, "colsample_bytree" = .1,
              "max_depth" = 22,  "gamma" = 0.1, "silent" = 0)
# fit the xgb
clf <- xgb.train(params = param, data = xgtrain, 
                 nround=5000, print.every.n = 25, watchlist=watchlist, 
                 early.stop.round = 50, maximize = TRUE)

## generate submission on validation and test sets ####
# prediction on test set
cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=id_test); submission$target <- NA 
for (rows in split(1:nrow(xtest), ceiling((1:nrow(xtest))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(xtest[rows,]))
}
cat("saving the submission file\n")
fname <- paste("./submissions/predFull_data",stringr::str_replace(vname, "_",""),"_seed",xseed,"_20151004.csv", sep = "")
write_csv(submission, fname)

# prediction on validation set
submission <- data.frame(ID=id_valid); submission$target <- NA 
pred <- predict(clf, xgval)
submission$target <- pred
fname <- paste("./submissions/predValid_data",stringr::str_replace(vname, "_",""),"_seed",xseed,"_20151004.csv", sep = "")
write_csv(submission, fname)