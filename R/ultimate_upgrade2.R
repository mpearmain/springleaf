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

time_cols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159",
               "VAR_0166","VAR_0167","VAR_0168","VAR_0169","VAR_0176","VAR_0177",
               "VAR_0178","VAR_0179","VAR_0204","VAR_0217")
time_cols <- time_cols[time_cols %in% colnames(xdat_fc)]

# SFSG # 

## new factor creation ####
# trim timestamps to year only
for (ff in time_cols)
{
  xdat_fc[, ff] <- str_sub(xdat_fc[,ff],6,7)
}

# drop everything except for time_cols, zipcode2, VAR_0467 and VAR_0274 (state)
xdat_fc <- xdat_fc[,c(time_cols, "zipcode2", "VAR_0467", "VAR_0274")]
# mini cleanup
xdat_fc$VAR_0467[xdat_fc$VAR_0467  == ""] <- "unk"
xdat_fc$zipcode2[xdat_fc$zipcode2  == ""] <- "999999"
xdat_fc$VAR_0274[xdat_fc$VAR_0274  == ""] <- "unk"
xdat_fc$state <- xdat_fc$VAR_0274; xdat_fc$VAR_0274 <- NULL
xdat_fc$disch <- xdat_fc$VAR_0467; xdat_fc$VAR_0467 <- NULL

# combination of zipcode2 and all years
for (ff in time_cols)
{
  xname <- str_replace_all(paste("zip2", ff, sep = "_"), "_","")
  xfac <- paste(xdat_fc$zipcode2, xdat_fc[,ff], sep = "_")
  xdat_fc[,xname] <- xfac
  msg(xname)
}

# combination of state and all years
for (ff in time_cols)
{
  xname <- str_replace_all(paste("state", ff, sep = "_"), "_","")
  xfac <- paste(xdat_fc$state, xdat_fc[,ff], sep = "_")
  xdat_fc[,xname] <- xfac
  msg(xname)
}

# combination of status and all years
for (ff in time_cols)
{
  xname <- str_replace_all(paste("disch", ff, sep = "_"), "_","")
  xfac <- paste(xdat_fc$disch, xdat_fc[,ff], sep = "_")
  xdat_fc[,xname] <- xfac
  msg(xname)
}

# state and status
xdat_fc$state_disch <- paste(xdat_fc$state, xdat_fc$disch, sep = "")

# cleanup
xdat_fc <- xdat_fc[,-which(colnames(xdat_fc) %in% time_cols)]
xdat_fc$zipcode2 <- xdat_fc$state <- xdat_fc$disch <- NULL

## convert factors to response rates ####
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

# zip vs state proportions - ratios
for (ff in time_cols)
{
  name1 <- str_replace(paste("zip2", ff, "dmp", sep = ""), "_","") 
  name2 <- str_replace(paste("state", ff,"dmp", sep = ""), "_","") 
  name3 <- paste("zipst", ff, sep = "")
  xtrain[,name3] <- xtrain[, name2] / xtrain[,name1]
  xtest[,name3] <- xtest[, name2] / xtest[,name1]
  
  msg(name3)
}


## aggregate and store ####
xtrain$ID <- id_train; xtrain$target <- y
colnames(xtrain) <- str_replace_all(colnames(xtrain), "_", "")
write_csv(xtrain, path = "./input/xtrain_v8b.csv")

xtest$ID <- id_test
colnames(xtest) <- str_replace_all(colnames(xtest), "_", "")
write_csv(xtest, path = "./input/xtest_v8b.csv")

