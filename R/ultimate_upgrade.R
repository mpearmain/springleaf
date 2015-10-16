## wd etc ####
require(Metrics)
require(caret)
require(readr)
require(doParallel)
require(stringr)
require(lubridate)
require(lme4)

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

## read data ####
xtrain <- read_csv(file = "./input/train.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL; y <- xtrain$target; xtrain$target <- NULL
xtest <- read_csv(file = "./input/test.csv")
id_test <- xtest$ID; xtest$ID <- NULL

## preliminary preparation ####
# drop columns with nothing but NA
is_missing <- colSums(is.na(xtrain))
constant_columns <- which(is_missing == nrow(xtrain))
xtrain <- xtrain[,-constant_columns]; xtest <- xtest[,-constant_columns]
rm(is_missing, constant_columns)

# drop duplicated columns
duplicate_columns <- which(duplicated(lapply(xtrain, c)))
xtrain <- xtrain[,-duplicate_columns]; xtest <- xtest[,-duplicate_columns]
rm(duplicate_columns)

# check column types
col_types <- unlist(lapply(xtrain, class))
fact_cols <- which(col_types == "character")
# separate into (what seems like) numeric and categorical
xtrain_fc <- xtrain[,fact_cols]; xtrain <- xtrain[, -fact_cols]
xtest_fc <- xtest[,fact_cols]; xtest <- xtest[, -fact_cols]
# add zipcode
xtrain_fc$zipcode <- xtrain$VAR_0212; xtest_fc$zipcode <- xtest$VAR_0212
xtrain_fc$zipcode2 <- xtrain$VAR_0241; xtest_fc$zipcode2 <- xtest$VAR_0241

## factor handling: cleanup  ####
isTrain <- 1:nrow(xtrain_fc); xdat_fc <- rbind(xtrain_fc, xtest_fc); rm(xtrain_fc, xtest_fc)
xdat_fc$zipcode <- as.character(xdat_fc$zipcode)
xdat_fc$zipcode2 <- as.character(xdat_fc$zipcode2)
xdat_fc$zipcode[is.na(xdat_fc$zipcode)] <- ""
xdat_fc$zipcode2[is.na(xdat_fc$zipcode2)] <- ""
# drop timestamp columns - not needed here
time_cols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159",
               "VAR_0166","VAR_0167","VAR_0168","VAR_0169","VAR_0176","VAR_0177",
               "VAR_0178","VAR_0179","VAR_0204","VAR_0217")
time_cols <- time_cols[time_cols %in% colnames(xdat_fc)]
xdat_fc <- xdat_fc[,-which(colnames(xdat_fc) %in% time_cols)]

  # true / false cases
  {
  tf_columns <- c("VAR_0008","VAR_0009","VAR_0010","VAR_0011","VAR_0012",
                  "VAR_0043","VAR_0196","VAR_0226","VAR_0229","VAR_0230","VAR_0232","VAR_0236","VAR_0239")
  tf_columns <- tf_columns[tf_columns %in% colnames(xdat_fc)]
  for (ff in tf_columns)
  {
    x <- xdat_fc[,ff];   x[x == ""] <- "mis"
    x <- factor(x);  xdat_fc[,ff] <- x 
    msg(ff)
  }

  }

  # location columns
  {
  loc_columns <- c("VAR_0237", "VAR_0274", "VAR_0200", "zipcode", "zipcode2")
  for (ff in loc_columns)
  {
    x <- xdat_fc[,ff];   x[x == ""] <- "mis"; x[x == "-1"] <- "mis"
    x <- factor(x);  xdat_fc[,ff] <- x 
    msg(ff)
  }
  }

  # alphanumeric generic columns
  {
  an_columns <- c("VAR_0001","VAR_0005", "VAR_0044", "VAR_1934", "VAR_0202", "VAR_0222",
                  "VAR_0216","VAR_0283","VAR_0305","VAR_0325",
                  "VAR_0342","VAR_0352","VAR_0353","VAR_0354","VAR_0466","VAR_0467")
  for (ff in an_columns)
  {
    x <- xdat_fc[,ff];   x[x == ""] <- "mis"; x[x == ""] <- "mis"
    x <- factor(as.integer(factor(x)));  xdat_fc[,ff] <- x 
    msg(ff)
  }
  }

  # job columns => for bag of words later
  {
  job_columns <- c("VAR_0404", "VAR_0493")
  xjobs <- xdat_fc[,job_columns]; xdat_fc <- xdat_fc[,-which(colnames(xdat_fc) %in% job_columns)]
  for (ff in job_columns)
  {
    x <- xjobs[,ff];   x[x == ""] <- "mis"; x[x == "-1"] <- "mis"
    x <- factor(x);  xjobs[,ff] <- x 
    msg(ff)
  }
  }

  rm(xtrain, xtest, xjobs,x, ff )
  

## factor handling: create new ones  ####
xcomb <- combn(ncol(xdat_fc),2)
for (ii in 1:ncol(xcomb))
{
  xloc <- xdat_fc[,xcomb[,ii]]
  xname <- paste("bi",colnames(xloc)[1] , colnames(xloc)[2],sep = "_")
  xfac <- paste(xloc[,1], xloc[,2], sep = "_")
  xdat_fc[,xname] <- xfac
  msg(ii)
}
  

## factor handling: counts  ####
for (ii in 1:ncol(xdat_fc))
{
  xname <- colnames(xdat_fc)[ii]
  xtab <- data.frame(table(xdat_fc[,ii]))
  colnames(xtab)[1] <- xname
  colnames(xtab)[2] <- paste("ct", xname, sep = "")
  xdat_fc[,paste("ct", xname, sep = "")] <- xtab[match(xdat_fc[,ii], xtab[,1]),2]
  msg(ii)
}
  
## add response rates for zipcode, zipcode2 and new ones ####
# drop the raw factors
drop_list <- grep("^VAR", colnames(xdat_fc))
xdat_fc <- xdat_fc[,-drop_list]
# separate the count columns
count_list <- grep("^ct", colnames(xdat_fc))
xdat_count <- xdat_fc[,count_list]
xdat_fc <- xdat_fc[,-count_list]
xtrain_count <- xdat_count[isTrain,]
xtest_count <- xdat_count[-isTrain,]
rm(xdat_count)
# separate into xtrain / xtest
xtrain <- xdat_fc[isTrain,]
xtest <- xdat_fc[-isTrain,]
rm(xdat_fc)

  # setup the folds for cross-validation
  xfold <- read_csv(file = "./input/xfolds.csv")
  idFix <- list()
  for (ii in 1:10)
  {
    idFix[[ii]] <- which(xfold$fold10 == ii)
  }
  rm(xfold,ii)  
  
  # grab factor variables
  factor_vars <- colnames(xtrain)
  
  # loop over factor variables, create a response rate version for each
  for (varname in factor_vars)
  {
    # placeholder for the new variable values
    x <- rep(NA, nrow(xtrain))
    for (ii in seq(idFix))
    {
      # separate ~ fold
      idx <- idFix[[ii]]
      x0 <- xtrain[-idx, factor_vars]; x1 <- xtrain[idx, factor_vars]
      y0 <- y[-idx]; y1 <- y[idx]
      # take care of factor lvl mismatches
      x0[,varname] <- factor(as.character(x0[,varname]))
      # fit LMM model
      myForm <- as.formula (paste ("y0 ~ (1|", varname, ")"))
      myLME <- lmer (myForm, x0, REML=FALSE, verbose=F)
      myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
      # table to match to the original
      myLMERDF <- data.frame (levelName = as.character(levels(x0[,varname])), myDampVal = myRanEf+myFixEf)
      rownames(myLMERDF) <- NULL
      x[idx] <- myLMERDF[,2][match(xtrain[idx, varname], myLMERDF[,1])]
      x[idx][is.na(x[idx])] <- mean(y0)
    }
    rm(x0,x1,y0,y1, myLME, myLMERDF, myFixEf, myRanEf)
    # add the new variable
    xtrain[,paste(varname, "dmp", sep = "")] <- x
    
    # create the same on test set
    xtrain[,varname] <- factor(as.character(xtrain[,varname]))
    x <- rep(NA, nrow(xtest))
    # fit LMM model
    myForm <- as.formula (paste ("y ~ (1|", varname, ")"))
    myLME <- lmer (myForm, xtrain[,factor_vars], REML=FALSE, verbose=F)
    myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
    # table to match to the original
    myLMERDF <- data.frame (levelName = as.character(levels(xtrain[,varname])), myDampVal = myRanEf+myFixEf)
    rownames(myLMERDF) <- NULL
    x <- myLMERDF[,2][match(xtest[, varname], myLMERDF[,1])]
    x[is.na(x)] <- mean(y)
    xtest[,paste(varname, "dmp", sep = "")] <- x
    msg(varname)
  }
  
  # drop the factors
  ix <- which(colnames(xtrain) %in% factor_vars)
  xtrain <- xtrain[,-ix]
  xtest <- xtest[,-ix]
  
## aggregate and store ####
xtrain <- cbind(xtrain, xtrain_count)
xtest <- cbind(xtest, xtest_count)
rm(xtrain_count, xtest_count)


xtrain$ID <- id_train; xtrain$target <- y
colnames(xtrain) <- str_replace_all(colnames(xtrain), "_", "")
write_csv(xtrain, path = "./input/xtrain_v8a.csv")

xtest$ID <- id_test
colnames(xtest) <- str_replace_all(colnames(xtest), "_", "")
write_csv(xtest, path = "./input/xtest_v8a.csv")

  