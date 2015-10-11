## wd etc ####
library(readr)
library(h2o)
require(Matrix)
require(caret)
require(stringr)

xseed <- 10
vname <- "v9_r5"
todate <- str_replace_all(str_sub(Sys.time(), 0, 10), "-", "")

set.seed(xseed)

## extra functions ####
  # generate a partition of the data
  generateSplit <- function(nofSplits, xLength )
  {
    
    set.seed(1); idFix <- list()
    scrambler <- sample(1:xLength, xLength, replace = FALSE)
    
    nx <- xLength + nofSplits - xLength %% nofSplits
    for (uu in 1:nofSplits) {  idFix[[uu]] <- (1 : (nx/nofSplits)) + (uu - 1) * nx/nofSplits      }
    idFix[[nofSplits]] <- idFix[[nofSplits]][idFix[[nofSplits]] <= xLength]
    
    for (uu in 1:nofSplits)
    {
      idFix[[uu]] <- scrambler[idFix[[uu]]]
    }
    return(idFix)
  }
  
  
  
  # format an input data frame into xgboost-acceptable format
  xgformat <- function(x)
  {
    return(matrix(as.numeric(as.matrix(x)),nrow(x),ncol(x)))
  }
  
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
xfold <- read_csv(file = "./input/xfolds.csv")
subrange <- which(xfold$valid == 1)
xvalid <- xtrain[subrange,]; y_valid <- y[subrange]; id_valid <- id_train[subrange]
xtrain <- xtrain[-subrange,]; y <- y[-subrange]

## setup h2o and the datasets in appropriate format ####
h2oServer <- h2o.init(nthreads=-1, max_mem_size = "12g")
xtrain$target <- factor(y); xvalid$target <- y_valid
train.hex <- as.h2o(xtrain)
valid.hex <- as.h2o(xvalid)
test.hex <- as.h2o(xtest)
xrange <- 1:(ncol(xtrain)-1)

## fit a model ####
size1 <- ncol(xtrain)
size2 <- round(0.5 * size1)
dl.model <- h2o.deeplearning(x = xrange, y = ncol(xtrain), 
                             training_frame = train.hex, 
                 autoencoder = FALSE, 
                 #activation = c("Rectifier", "Tanh", "TanhWithDropout",
                #                "RectifierWithDropout", "Maxout", "MaxoutWithDropout"), 
                 activation = "Rectifier",
                 hidden = c(500, 100), epochs = 15, train_samples_per_iteration = -2, seed = xseed,
                 adaptive_rate = TRUE, rho = 0.99, epsilon = 1e-08, rate = 0.005,
                 rate_annealing = 1e-06, rate_decay = 1, momentum_start = 0,
                 nesterov_accelerated_gradient = TRUE, input_dropout_ratio = 0.02,
                 hidden_dropout_ratios = c(0.01, 0.01), l1 = 0, l2 = 0, 
                loss = c("CrossEntropy")
                 )

# generate predictions
pred_valid <- as.data.frame(predict(dl.model, valid.hex))$p1
pred_full <- as.data.frame(predict(dl.model, test.hex))$p1
xfor <- read_csv("./submissions/xmix3_20150908.csv")
xfor$target <- pred_full
write_csv(xfor, path = "./submissions/h2o_deeplearn_20151008.csv")

