## wd etc ####
library(readr)
library(xgboost)
require(Matrix)
require(caret)

xseed <- 1
set.seed(xseed)

## load and process data ####
xtrain <- read_csv(file = "./input/xtrain_v7.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL
xtest <- read_csv(file = "./input/xtest_v7.csv")
id_test <- xtest$ID; xtest$ID <- NULL

# reduction
train.unique.count=lapply(xtrain, function(x) length(unique(x)))
train.unique.count_1=unlist(train.unique.count[unlist(train.unique.count)==1])
train.unique.count_2=unlist(train.unique.count[unlist(train.unique.count)==2])
train.unique.count_2=train.unique.count_2[-which(names(train.unique.count_2)=='target')]

delete_const=names(train.unique.count_1)
delete_NA56=names(which(unlist(lapply(xtrain[,(names(xtrain) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145175))
delete_NA89=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145142))
delete_NA918=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==144313))



## xgboost  ####
#Set xgboost test and training and validation datasets
xfold <- read_csv(file = "./input/xfolds.csv")
subrange <- which(xfold$valid == 1)
xvalid <- xtrain[subrange,]; y_valid <- y[subrange]
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

## generate submission on complete test set ####
cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=id_test)
submission$target <- NA 
for (rows in split(1:nrow(xtest), ceiling((1:nrow(xtest))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(xtest[rows,]))
}

cat("saving the submission file\n")
fname <- paste("./submissions/xgboost_dataV6_20150905.csv", sep = "")
write_csv(submission, fname)