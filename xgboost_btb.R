## wd etc ####
library(readr)
library(xgboost)

set.seed(1)

## load and process data ####
cat("reading the train and test data\n")
xtrain <- read_csv(file = "./input/xtrain.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL

## fit the model with early stopping ####
set.seed(20150817)
val_size <- 14000
subrange <- sample(nrow(xtrain), size = val_size)

# Set xgboost test and training and validation datasets
xgtest <- xgb.DMatrix(data = as.matrix(xtest))
xgtrain <- xgb.DMatrix(data = as.matrix(xtrain)[-subrange,], label= y[-subrange])
xgval <-  xgb.DMatrix(data = as.matrix(xtrain)[subrange,], label= y[subrange])
watchlist <- list(val=xgval, train=xgtrain)

# configure parameters
param <- list(objective   = "binary:logistic",
              eval_metric = "auc",
              "eta" = 0.05,
              "min_child_weight" = 8,
              "subsample" = .9, "colsample_bytree" = .7,
              "max_depth" = 9,  "gamma" = 0.025, "silent" = 0)
# fit the xgb
clf <- xgb.train(params = param, data = xgtrain, 
                 nround=350, print.every.n = 25, watchlist=watchlist, 
                 early.stop.round = 50, maximize = TRUE)


## generate submission on complete training set ####
xtest <- read_csv(file = "./input/test.csv")
id_test <- xtest$ID; xtest$ID <- NULL

cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=id_test)
submission$target <- NA 
for (rows in split(1:nrow(xtest), ceiling((1:nrow(xtest))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(xtest[rows,]))
}

cat("saving the submission file\n")
write_csv(submission, "./submissions/xgboost_submission2_20150819.csv")