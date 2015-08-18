library(readr)
library(xgboost)

set.seed(1)

cat("reading the train and test data\n")
train <- read_csv("./input/train.csv")
test  <- read_csv("./input/test.csv")

feature.names <- names(train)[2:ncol(train)-1]

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)]   <- -1

cat("sampling train to get around 8GB memory limitations\n")
train <- train[sample(nrow(train), 40000),]
gc()

cat("training a XGBoost classifier\n")
clf <- xgboost(data        = data.matrix(train[,feature.names]),
               label       = train$target,
               nrounds     = 200,
               objective   = "binary:logistic",
               eval_metric = "auc",
               eta         = 0.05,
               verbose = 1)

cat("making predictions in batches due to 8GB memory limitation\n")
submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
  submission[rows, "target"] <- predict(clf, data.matrix(test[rows,feature.names]))
}

cat("saving the submission file\n")
write_csv(submission, "./submissions/xgboost_submission2_20150818.csv")