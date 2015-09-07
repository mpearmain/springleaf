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
# cleanup constant columns
xsd <- apply(xtrain,2,sd, na.rm = T)
constant_columns <- which(xsd == 0)
xtrain <- xtrain[,-constant_columns]; xtest <- xtest[,-constant_columns]
rm(constant_columns, xsd)

# cleanup VAR_0212
xtrain$VAR_0212 <- xtrain$VAR_0212/10^9; xtest$VAR_0212 <- xtest$VAR_0212/10^9

# check the minima so that reasonable values can be used as replacement
xmin <- apply(xtrain,2,min, na.rm = T); xmax <- apply(xtrain,2,max, na.rm = T)

# take care of missing values
is_missing <- which(colSums(is.na(xtrain)) > 0)
xtrain[is.na(xtrain)] <- -1; xtest[is.na(xtest)] <- -9.999900e+04

## correlated pairs and derived features ####
  
  # identify highly correlated pairs
  {
      corr_pairs <- array(1,c(1,2))
      for (which_column in 1:ncol(xtrain))
      {
        cor_vec <- unlist(lapply(xtrain, function(s) cor(xtrain[,which_column],s)))
        cor_vec[which_column] <- 0
        dep_columns <- which(abs(cor_vec) > 0.95)
        if (length(dep_columns))
        {
          ref_columns <- rep(which_column, length(dep_columns))
          x <- cbind(ref_columns, dep_columns)  
          corr_pairs <- rbind(corr_pairs, x)
          print("corr found")
        }
        msg(which_column)
        # trim along the way
        if ((which_column %% 50) == 0)
        {
          corr_pairs <- corr_pairs[corr_pairs[,1] > corr_pairs[,2],]
        }
      }
      rm(which_column)
      # map the numbers to names
      ref_name <- colnames(xtrain)[corr_pairs[,1]]
      dep_name <- colnames(xtrain)[corr_pairs[,2]]
      corr_pairs <- data.frame(corr_pairs, ref_name, dep_name)
      rm(ref_name, dep_name)
      
      # amend with the correlation numbers
      corr_value <- rep(0, nrow(corr_pairs))
      for (ii in 1:nrow(corr_pairs))
      {
        corr_value[ii] <- cor(xtrain[,corr_pairs[ii,1]], xtrain[,corr_pairs[ii,2]])
        if ((ii %% 1000) == 0) msg(ii)
      }
      # match with actual correlations
      corr_pairs <- data.frame(corr_pairs, corr_value)
      rm(corr_value)
      # store the intermediate file
      write_csv(corr_pairs, path = "./input/correlated_pairs.csv")
      
      
}
  #corr_pairs <- read_csv(file = "./input/raw_correlated_pairs.csv")

  # suspicious columns - evaluate adequacy of pairwise differences of correlated ones
  xsum <- xdiff <- rep(0, nrow(corr_pairs))
  # create folds
  xfold <- read_csv(file = "./input/xfolds.csv")
  idFix <- list()
  for (ii in 1:10)
  {
    idFix[[ii]] <- which(xfold$fold10 == ii)
  }
  rm(xfold,ii)
  # loop over highly correlated pairs - check the information value of a sum
  # and difference of each correlated pair
  for (ii in 1:nrow(corr_pairs))
  {
    i1 <- corr_pairs[ii,1]; i2 <- corr_pairs[ii,2]
    loc_var <- xtrain[, i1] + xtrain[,i2]
    loc_res <- unlist(lapply(idFix, function(s) auc(y[s], loc_var[s])))
    xsum[ii] <- mean(loc_res) - 3 * sd(loc_res)
    loc_var <- xtrain[, i1] - xtrain[,i2]
    loc_res <- unlist(lapply(idFix, function(s) auc(y[s], loc_var[s])))
    xdiff[ii] <- mean(loc_res) - 3 * sd(loc_res)
    if ((ii %% 50) == 0) msg(ii)
  }
  rm(ii, i1, i2)
  corr_pairs <- data.frame(corr_pairs, xsum, xdiff)
  corr_pairs$ref_name <- as.character(corr_pairs$ref_name)
  corr_pairs$dep_name <- as.character(corr_pairs$dep_name)
  
  rm(xsum, xdiff)
  # loop over the pairs - add the useful ones to xtrain
  for (ii in 1:nrow(corr_pairs))
  {
    i1 <- corr_pairs[ii,1]; i2 <- corr_pairs[ii,2]
    if (corr_pairs$xsum[ii] > 0.57)
    {
      # attach new variable to train
      loc_var <- xtrain[, i1] + xtrain[,i2]
      xtrain <- data.frame(xtrain, loc_var)
      newname <- str_replace_all(paste("sum",corr_pairs$dep_name[ii], corr_pairs$ref_name[ii], sep = ""), "VAR","")
      colnames(xtrain)[ncol(xtrain)] <- newname
      # attach new variable to test
      loc_var <- xtest[, i1] + xtest[,i2]
      xtest <- data.frame(xtest, loc_var)
      newname <- str_replace_all(paste("sum",corr_pairs$dep_name[ii], corr_pairs$ref_name[ii], sep = ""), "VAR","")
      colnames(xtest)[ncol(xtest)] <- newname
    }
    
    if (corr_pairs$xdiff[ii] > 0.57)
    {
      # attach new variable to train
      loc_var <- xtrain[, i1] - xtrain[,i2]
      xtrain <- data.frame(xtrain, loc_var)
      newname <- str_replace_all(paste("diff",corr_pairs$dep_name[ii], corr_pairs$ref_name[ii], sep = ""), "VAR","")
      colnames(xtrain)[ncol(xtrain)] <- newname
      # attach new variable to test
      loc_var <- xtest[, i1] - xtest[,i2]
      xtest <- data.frame(xtest, loc_var)
      newname <- str_replace_all(paste("diff",corr_pairs$dep_name[ii], corr_pairs$ref_name[ii], sep = ""), "VAR","")
      colnames(xtest)[ncol(xtest)] <- newname
    }
    
    msg(ii)
  }
  rm(ii, i1,i2, loc_sum, loc_res, loc_var)     
 
  # drop the original highly correlated columns
  duplicate_columns <- unique(corr_pairs$ref_columns[abs(corr_pairs$corr_value) > 0.95])
  xtrain <- xtrain[,-duplicate_columns]
  xtest <- xtest[,-duplicate_columns]

## find linear combinations ####
  # shit is too big to fit in memory => doing it via sort of bagging
  # i.e. draw a subset N times, record variables to drop in each
  nBags <- 100
  set.seed(20150817)
  idFix <- createDataPartition(1:ncol(xtrain), times = nBags, p = 0.2, list = T)
  columns_to_drop <- list()
  for (ff in 1:length(idFix))
  {
    xloc <- xtrain[, idFix[[ff]]]
    flc <- findLinearCombos(xloc)
    if (length(flc$remove))
    {
      columns_to_drop[[ff]] <- colnames(xloc)[flc$remove]
    }
    msg(ff)
  }
  columns_to_drop <- unique(unlist(columns_to_drop))
  indices_to_drop <- which(colnames(xtrain) %in% columns_to_drop)
  xtrain <- xtrain[,-indices_to_drop]
  xtest <- xtest[,-indices_to_drop]
  rm(columns_to_drop, indices_to_drop)
  

## factor handling  ####
isTrain <- 1:nrow(xtrain_fc); xdat_fc <- rbind(xtrain_fc, xtest_fc); rm(xtrain_fc, xtest_fc)

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
    xtf <- xdat_fc[,tf_columns]; xdat_fc <- xdat_fc[,-which(colnames(xdat_fc) %in% tf_columns)]
    x <- xtf; for (ii in 1:ncol(x)) x[,ii] <- as.integer(x[,ii])
    flc <- findLinearCombos(x)
    if (length(flc$remove)) xtf <- xtf[,-flc$remove]
    xdat_fc <- data.frame(xdat_fc, xtf)
    rm(x, xtf, flc)
  
  }

  # timestamp columns
  {
    time_cols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159",
                   "VAR_0166","VAR_0167","VAR_0168","VAR_0169","VAR_0176","VAR_0177",
                   "VAR_0178","VAR_0179","VAR_0204","VAR_0217")
    time_cols <- time_cols[time_cols %in% colnames(xdat_fc)]
    
    # drop the time (leave date only) so you can actually see sth
    xtime <- xdat_fc[,time_cols]
    for (ff in time_cols)
    {
      x <- xtime[,ff];  x <- substr(x,1,7)
      x[x == ""] <- NA
      xtime[,ff] <- x
    }
    # summarize
    initial_year <- rep(0, nrow(xdat_fc))
    last_year <- rep(0, nrow(xdat_fc))
    nof_dates <- nof_unique_dates <- rep(0, nrow(xdat_fc))
    duration_contact <- rep(0, nrow(xdat_fc))
    for (ii in 1:nrow(xdat_fc))
    {
      formatted_dates <- as.Date(na.omit(unlist(xtime[ii,])), "%d%b%y")
      nof_dates[ii] <- length(formatted_dates)
      nof_unique_dates[ii] <- length(unique(formatted_dates))
      if (nof_dates[ii])
      {
        initial_year[ii] <- year(min(formatted_dates))
        last_year[ii] <- year(max(formatted_dates))
        duration_contact[ii] <- as.integer(max(formatted_dates) - min(formatted_dates))
      }
    }
    nof_years <- last_year - initial_year
    # cleanup time-related ones
    # initial/last year - drop the latter, as its either 0 or 2014 
    xdat_fc$initial_year <- factor(initial_year); rm(initial_year); rm(last_year)
    # number of dates and unique dates
    xdat_fc$nof_dates <- nof_dates; rm(nof_dates)
    xdat_fc$nof_unique_dates <- nof_unique_dates; rm(nof_unique_dates)
    # contact timespan
    xdat_fc$duration_contact <- duration_contact; rm(duration_contact)
    # contact - count the years
    xdat_fc$nof_years <- nof_years; rm(nof_years)
    rm(xtime)
    xdat_fc <- xdat_fc[,-which(colnames(xdat_fc) %in% time_cols)]
    rm(time_cols)
  
  }

  # location columns
  {
    loc_columns <- c("VAR_0237", "VAR_0274", "VAR_0200")
    for (ff in loc_columns)
    {
      x <- xdat_fc[,ff];   x[x == ""] <- "mis"; x[x == "-1"] <- "mis"
      x <- factor(x);  xdat_fc[,ff] <- x 
      msg(ff)
    }
    xdat_fc$same_state <- ( as.character(xdat_fc$VAR_0237) == as.character(xdat_fc$VAR_0274)) + 0
    xdat_fc$state1_mis <- (xdat_fc$VAR_0237 == "mis") + 0
    xdat_fc$state2_mis <- (xdat_fc$VAR_0274 == "mis") + 0
    xdat_fc$both_state_mis <- xdat_fc$state1_mis * xdat_fc$state2_mis
    # city
    xtab <- data.frame(table(xdat_fc$VAR_0200))
    record_count <- xtab[,2][match(xdat_fc$VAR_0200, xtab[,1])]
    mapped_city <- as.character(xdat_fc$VAR_0200)
    mapped_city[record_count < 1000] <- "RARE"
    xdat_fc$VAR_0200 <- factor(mapped_city)
    
    rm(xtab, record_count, mapped_city)
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
    xdat_fc$job1_mis <- (xjobs[,1] == "mis") + 0
    xdat_fc$job2_mis <- (xjobs[,2] == "mis") + 0
    xdat_fc$both_jobs_mis <- xdat_fc$job1_mis * xdat_fc$job2_mis
    xdat_fc <- data.frame(xdat_fc, xjobs)
    rm(xjobs)
  }

## reassign ####
  xnames <- c("initial_year","nof_dates","nof_unique_dates","duration_contact",
              "nof_years" ,"same_state","state1_mis","state2_mis","both_state_mis",
              "job1_mis", "job2_mis","both_jobs_mis")
  true_fac <- !(colnames(xdat_fc) %in% xnames)
  
  # full training set
  xtrain_fc <- xdat_fc[isTrain,]
  # move numerical ones fc -> regular
  colnames(xtrain_fc)[true_fac] <- paste("fac", colnames(xtrain_fc)[true_fac] , sep = "")
  xtrain <- data.frame(xtrain, xtrain_fc); rm(xtrain_fc)

  # full test set
  xtest_fc <- xdat_fc[-isTrain,]
  colnames(xtest_fc)[true_fac] <- paste("fac", colnames(xtest_fc)[true_fac] , sep = "")
  xtest <- data.frame(xtest, xtest_fc); rm(xtest_fc)

  # add a combination of state and year
  xtrain$fac_yearState <- paste(xtrain$initial_year, xtrain$facVAR_0237, sep = "_")
  xtest$fac_yearState <- paste(xtest$initial_year, xtest$facVAR_0237, sep = "_")
  
  rm(xdat_fc, corr_pairs, an_columns)
  

## convert factors to response rates ####
  # setup the folds for cross-validation
  xfold <- read_csv(file = "./input/xfolds.csv")
  idFix <- list()
  for (ii in 1:10)
  {
    idFix[[ii]] <- which(xfold$fold10 == ii)
  }
  rm(xfold,ii)  
  
  # grab factor variables
  factor_vars <- colnames(xtrain)[grep("fac", colnames(xtrain))]
  
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
  
  # SFSG # 
  
  
  
##  output formatted datasets ####
# store pure train set
xtrain$ID <- id_train; xtrain$target <- y
colnames(xtrain) <- str_replace_all(colnames(xtrain), "_", "")
write_csv(xtrain, path = "./input/xtrain_v6.csv")

xtest$ID <- id_test
colnames(xtest) <- str_replace_all(colnames(xtest), "_", "")
write_csv(xtest, path = "./input/xtest_v6.csv")
