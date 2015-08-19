## wd etc ####
library(readr)
library(xgboost)

set.seed(1)

## load and process data ####
cat("reading the train and test data\n")
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

# map all categoricals into numeric => probably BS
cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in colnames(xtrain)) {
  if (class(xtrain[[f]])=="character") {
    levels <- unique(c(xtrain[[f]], xtest[[f]]))
    xtrain[[f]] <- as.integer(factor(xtrain[[f]], levels=levels))
    xtest[[f]]  <- as.integer(factor(xtest[[f]],  levels=levels))
  }
}

# handle NAs
cat("replacing missing values with -1\n")
xtrain[is.na(xtrain)] <- -1
xtest[is.na(xtest)]   <- -1

## fit the model with early stopping ####
set.seed(20150817)
val_size <- 10000
subrange <- sample(nrow(xtrain), size = val_size)

# Set xgboost test and training and validation datasets
xgtest <- xgb.DMatrix(data = as.matrix(xtest))
xgtrain <- xgb.DMatrix(data = as.matrix(xtrain)[-subrange,], label= y[-subrange])
xgval <-  xgb.DMatrix(data = as.matrix(xtrain)[subrange,], label= y[subrange])
watchlist <- list(val=xgval, train=xgtrain)

# configure parameters
param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              "eta" = 0.02,
              "min_child_weight" = 8,
              "subsample" = .9, "colsample_bytree" = .7,
              "max_depth" = 11,  "gamma" = 0.025, "silent" = 0)
# fit the xgb
clf <- xgb.train(params = param, data = xgtrain, 
                 nround=900, print.every.n = 25, watchlist=watchlist, 
                 early.stop.round = 50, maximize = TRUE)



## generate submission on complete training set ####
cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=id_test)
submission$target <- NA 
for (rows in split(1:nrow(xtest), ceiling((1:nrow(xtest))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(xtest[rows,]))
}

cat("saving the submission file\n")
write_csv(submission, "./submissions/xgboost_submission_20150819.csv")