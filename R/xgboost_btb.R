## wd etc ####
library(readr)
library(xgboost)
require(Matrix)
require(caret)

xseed <- 1
set.seed(xseed)

## load and process data ####
xtrain <- read_csv(file = "./input/xtrain_v5.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL
xvalid <- read_csv(file = "./input/xvalid_v5.csv")
id_valid <- xvalid$ID; xvalid$ID <- NULL
y_valid <- xvalid$target; xvalid$target <- NULL
xtest <- read_csv(file = "./input/xtest_v5.csv")
id_test <- xtest$ID; xtest$ID <- NULL

## processing of factors
# drop the job titles for now
xtrain$facVAR0404 <- xtrain$facVAR0493 <- NULL
xvalid$facVAR0404 <- xvalid$facVAR0493 <- NULL
xtest$facVAR0404 <- xtest$facVAR0493 <- NULL
# sparsify
fact_cols <- grep("fac",colnames(xtrain))
xtr <- xtrain[,fact_cols] ; xv <- xvalid[,fact_cols]; xte <- xtest[,fact_cols]
xtrain <- xtrain[, -fact_cols]; xvalid <- xvalid[,-fact_cols]; xtest <- xtest[,-fact_cols]
xd <- rbind(xtr, xv, xte)
a <- sparse.model.matrix(~ . -1, data = xd); a <- as.matrix(a)
flc <- findLinearCombos(a); a <- a[,-flc$remove]
xtr_fc <- a[1:nrow(xtrain),]
xv_fc <- a[(nrow(xtrain)+1):(nrow(xtrain) + nrow(xvalid)),]
xte_fc <- a[(nrow(xtrain)+nrow(xvalid) + 1):nrow(a),]
xtrain <- data.frame(xtrain, xtr_fc)
xvalid <- data.frame(xvalid, xv_fc)
xtest <- data.frame(xtest, xte_fc)
rm(xte, xte_fc, xtr, xtr_fc, xv, xv_fc)

## xgboost  ####
#Set xgboost test and training and validation datasets
xgtrain <- xgb.DMatrix(data = as.matrix(xtrain), label= y)
xgval <-  xgb.DMatrix(data = as.matrix(xvalid), label= y_valid)
watchlist <- list(val=xgval)
rm(xtrain, xvalid)

# configure parameters
param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              "eta" = 0.004,
              "min_child_weight" = 8,
              "subsample" = .7, "colsample_bytree" = .7,
              "max_depth" = 20,  "gamma" = 0.1, "silent" = 0)
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
fname <- paste("./submissions/xgboost_dataV5_20150901.csv", sep = "")
write_csv(submission, fname)