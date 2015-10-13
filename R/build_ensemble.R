## wd etc ####
library(readr)
library(xgboost)
require(Matrix)
require(caret)
require(stringr)


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

xrange <- c(1,3,5,6,7,8)

# complete forecasts
file_list2 <- str_replace(file_list, "predValid", "predFull")
xfull <- array(0, c(length(id_test), length(file_list2)))
for (ii in 1:ncol(xvalid))
{
  xfull[,ii] <- read_csv(file_list2[[ii]])$target
}

## build ensemble ####
nTimes <- 40
valRange <- seq(from = 0, to = 1, by = 0.05)
idFix <- createDataPartition(y_valid, times = nTimes, p = 0.25)
storageMat <- array(0, c(nTimes, 2 * length(valRange)))
for (ii in 1:nTimes)
{
  idx <- idFix[[ii]]
  xvalid0 <- xvalid[-idx,]; yvalid0 <- y_valid[-idx]
  xvalid1 <- xvalid[idx,]; yvalid1 <- y_valid[idx]
  # transform to rank
  xvalid0 <- apply(xvalid0, 2, rank); xvalid1 <- apply(xvalid1,2, rank)
  for (jj in 1:length(valRange))
  {
    
    mod0 <- glmnet(x = xvalid0, y = yvalid0, alpha = valRange[jj])
    prx <- predict(mod0, xvalid1); prx <- prx[,ncol(prx)]
    storageMat[ii,jj] <- auc(yvalid1, prx)
    
    mod0 <- glmnet(x = xvalid0, y = yvalid0, alpha = valRange[jj], standardize = F)
    prx <- predict(mod0, xvalid1); prx <- prx[,ncol(prx)]
    storageMat[ii,jj + length(valRange)] <- auc(yvalid1, prx)
    
    
 }

  msg(ii)
   
}

# fit complete model
mod0 <- glmnet(x = xvalid, y = y_valid, alpha = valRange[which.max(colMeans(storageMat))])
pred <- predict(mod0, xfull)
pred <- pred[,ncol(pred)]
xfor <- data.frame(ID = id_test, target = pred)