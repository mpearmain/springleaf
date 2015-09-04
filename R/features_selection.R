## wd etc ####
require(Metrics)
require(caret)
require(readr)
require(doParallel)
require(gbm)

## extra functions ####
# print a formatted message
msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

# SFSG # 

## feature selection - greedy ####
# select a good feature to start with
# set.seed(20150817)
# idFix <- createFolds(y, k = 5, list = T)
# relevMat <- array(0, c(length(idFix), ncol(xtrain)))
# for (ii in seq(idFix))
# {
#   idx <- idFix[[ii]]
#   relevMat[ii,] <- apply(xtrain[idx,],2,function(s) Metrics::auc(actual = y[idx], s))
#   print(ii)
# }
# goodFeatures <- which.max(colMeans(relevMat))
# goodScore <- mean(relevMat[,goodFeatures])
# goodFeatures: 78
# goodScore: 0.6422847

# iterate over features

# while (length(goodFeatures) < 35)
# {
#   relevMat <- rep(-10, ncol(xtrain))
#   for (jj in 1:ncol(xtrain))
#   {
#     if (!(jj %in% goodFeatures))
#     {
#       # surprisingly, overhead eats the whole advantage from parallelization
#       system.time(a <- foreach(ii = 1:length(idFix), .combine = c, .packages = c("Metrics", "glmnet")) %do%
#       {
#         idx <- idFix[[ii]]
#         y0 <- y[-idx]; y1 <- y[idx]
#         x0 <- xtrain[-idx, c(jj, goodFeatures)]
#         x1 <- xtrain[idx,c(jj, goodFeatures)]
#         mod0 <- glmnet(y = y0, x = as.matrix(x0), alpha = 0, family = "gaussian")
#         pred <- predict(mod0, as.matrix(x1))
#         Metrics::auc(y1,pred[,ncol(pred)])
#       }
#       
# 	  )
# 	  relevMat[jj] <- mean(a)
#       
#     }
# 	if ((jj %% 300) == 0)
# 	{
# 		msg(jj)
# 	}
#   }
#   goodFeatures <- c(goodFeatures, which.max(relevMat))
#   goodScore <- c(goodScore, max(relevMat))
# }
# 
# rm(y0,y1,a,cl, col_types, fact_cols, idx, ii, is_missing,jj, mod0, pred)
# rm(xtrain_fc,x0,x1)
# 
# xtrain <- xtrain[,goodFeatures]
# xtrain$ID <- id_train
# xtest$ID <- id_test
# xtrain$target <- y
# write_csv(xtrain, path = "./input/xtrain_r1.csv" )
# write_csv(xtest, path = "./input/xtest_r1.csv" )
# 
# 
# rm(id_test, id_train, idFix, msg, relevMat, xtest, xtrain,y)
# save.image("greedy_feature_selection.RData")

## feature selection - gbm-based ####
# load the results from greedy feature selection
for (which_version in c("v5"))
{
  xtrain <- read_csv(file = paste("./input/xtrain_",which_version,".csv",sep = "") )
  id_train <- xtrain$ID; xtrain$ID <- NULL
  y <- xtrain$target; xtrain$target <- NULL
  
  xtest <- read_csv(file = paste("./input/xtest_",which_version,".csv",sep = "") )
  id_test <- xtest$ID; xtest$ID <- NULL
  
  set.seed(20150817)
  idFix <-createDataPartition(y, times = 25, p = 0.1, list = T)
  relevMat <- array(0, c(length(idFix), ncol(xtrain)))
  for (ii in seq(idFix))
  {
      idx <- idFix[[ii]]
      x0 <- xtrain[idx,]; y0 <- y[idx]
      mod0 <- gbm.fit(x = x0, y = y0, n.trees = 100, interaction.depth = 12, shrinkage = 0.05,
          distribution = "bernoulli", verbose = T)
      relevMat[ii,] <-  summary(mod0, order = F, plot = F)[,2]
      print(ii)
  }
  
  # version 1: all non-zero
  subset1 <- which(apply(relevMat,2,prod) != 0)
  xtrain1 <- xtrain[,subset1]
  xtest1 <- xtest[,subset1]
  xtrain1$ID <- id_train; xtest1$ID <- id_test
  xtrain1$target <- y
  write_csv(xtrain1, path = paste("./input/xtrain_",which_version,"_r1.csv", sep = "") )
  write_csv(xtest1, path = paste("./input/xtest_",which_version,"_r1.csv", sep = "") )
  
  # version 2: non-zero 0.25 of the time
  subset1 <- which(apply(apply(relevMat,2,sign),2,sum) > 0.25 * length(idFix))
  xtrain1 <- xtrain[,subset1]
  xtest1 <- xtest[,subset1]
  xtrain1$ID <- id_train; xtest1$ID <- id_test
  xtrain1$target <- y
  write_csv(xtrain1, path = paste("./input/xtrain_",which_version,"_r2.csv", sep = "") )
  write_csv(xtest1, path = paste("./input/xtest_",which_version,"_r2.csv", sep = "") )
  
  # version 3: non-zero 0.5 of the time
  subset1 <- which(apply(apply(relevMat,2,sign),2,sum) > 0.5 * length(idFix))
  xtrain1 <- xtrain[,subset1]
  xtest1 <- xtest[,subset1]
  xtrain1$ID <- id_train; xtest1$ID <- id_test
  xtrain1$target <- y
  write_csv(xtrain1, path = paste("./input/xtrain_",which_version,"_r3.csv", sep = "") )
  write_csv(xtest1, path = paste("./input/xtest_",which_version,"_r3.csv", sep = "") )
}