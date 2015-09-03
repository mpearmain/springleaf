## wd etc ####
require(Metrics)
require(caret)
require(readr)
require(doParallel)
require(stringr)
require(lubridate)

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

# helper function for extracting max year < 99
extractYear <- function(vec)
{
  vec <- as.character(vec)
  x <- as.integer(str_sub(vec,-2))
  x <- x[x < 99]
  m <- max(x)
  if (is.infinite(m))
  {
    m <- 99
  }
  return(m)
}

## prepare dataset v1 ####
# all numbers => good basis for VW
# also: feed to Python gbm!
xtrain <- read_csv(file = "./input/train.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL

xtest <- read_csv(file = "./input/test.csv")
id_test <- xtest$ID; xtest$ID <- NULL

# drop constant columns
is_missing <- colSums(is.na(xtrain))
constant_columns <- which(is_missing == nrow(xtrain))
xtrain <- xtrain[,-constant_columns]
xtest <- xtest[,-constant_columns]
rm(is_missing, constant_columns)

# catch factor columns
fact_cols <- which(lapply(xtrain, class) == "character")
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


xtrain$ID <- id_train; xtrain$target <- y
write_csv(xtrain, path = "./input/xtrain_v1.csv")

xtest$ID <- id_test
write_csv(xtest, path = "./input/xtest_v1.csv")

## prepare dataset v5 ####
# clean factors + response rates + indicators of NA etc
xtrain <- read_csv(file = "./input/train.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL; y <- xtrain$target; xtrain$target <- NULL
xtest <- read_csv(file = "./input/test.csv")
id_test <- xtest$ID; xtest$ID <- NULL

## preliminary preparation
# drop columns with nothing but NA
is_missing <- colSums(is.na(xtrain))
constant_columns <- which(is_missing == nrow(xtrain))
xtrain <- xtrain[,-constant_columns]; xtest <- xtest[,-constant_columns]
rm(is_missing, constant_columns)

# check column types
is_missing <- which(colSums(is.na(xtrain)) > 0)
xtrain[is.na(xtrain)] <- -1; xtest[is.na(xtest)] <- -1
col_types <- unlist(lapply(xtrain, class))
fact_cols <- which(col_types == "character")
# separate into numeric and categorical
xtrain_fc <- xtrain[,fact_cols]; xtrain <- xtrain[, -fact_cols]
xtest_fc <- xtest[,fact_cols]; xtest <- xtest[, -fact_cols]
# cleanup constant columns
xsd <- apply(xtrain,2,sd)
constant_columns <- which(xsd == 0)
xtrain <- xtrain[,-constant_columns]; xtest <- xtest[,-constant_columns]

## check correlated pairs
{
  {
    corr_pairs <- array(1,c(1,2))
    for (which_column in 1:ncol(xtrain))
    {
      cor_vec <- rep(0, ncol(xtrain))
      for (ii in 1:ncol(xtrain))  cor_vec[ii] <- cor(xtrain[,which_column], xtrain[,ii])
      cor_vec[which_column] <- 0
      dep_columns <- which(abs(cor_vec) > 0.95)
      if (length(dep_columns))
      {
        ref_columns <- rep(which_column, length(dep_columns))
        x <- cbind(ref_columns, dep_columns)  
        corr_pairs <- rbind(corr_pairs, x)
      }
      msg(which_column)
    }
  }
  #corr_pairs <- read_csv(file = "./input/raw_correlated_pairs.csv")
  corr_pairs <- corr_pairs[corr_pairs[,1] > corr_pairs[,2],]
  corr_value <- rep(0, nrow(corr_pairs))
  for (ii in 1:nrow(corr_pairs))
  {
    corr_value[ii] <- cor(xtrain[,corr_pairs[ii,1]], xtrain[,corr_pairs[ii,2]])
    if ((ii %% 1000) == 0) msg(ii)
  }
  # match with actual correlations
  corr_pairs <- data.frame(corr_pairs, corr_value)
  
  # duplicate columns - for now drop everything that is too highly correlated
  duplicate_columns <- unique(corr_pairs$ref_columns[corr_pairs$corr_value > 0.99])
  xtrain <- xtrain[,-duplicate_columns]
  xtest <- xtest[,-duplicate_columns]

}

## find linear combinations
{
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
  # sliding window
  columns_to_drop2 <- list()
  xrange <-  ((0:10) * 100) + 1
  for (ii in seq(xrange))
  {
    idx <- xrange[ii]:(min(xrange[ii] + 100, ncol(xtrain)))
    flc <- findLinearCombos(xtrain[,idx])
    if (length(flc$remove))
    {
      columns_to_drop2[[ii]] <- colnames(xtrain[,idx])[flc$remove]
    }
    msg(ii)
  }
  columns_to_drop2 <- unique(unlist(columns_to_drop2))
  total_drop <- which(colnames(xtrain) %in% unique(c(columns_to_drop, columns_to_drop2)))
  xtrain <- xtrain[, -total_drop]
  xtest <- xtest[, -total_drop]
  # sliding window - once more
  columns_to_drop3 <- list()
  xrange <-  ((0:10) * 100) + 1
  for (ii in seq(xrange))
  {
    idx <- xrange[ii]:(min(xrange[ii] + 500, ncol(xtrain)))
    flc <- findLinearCombos(xtrain[,idx])
    if (length(flc$remove))
    {
      columns_to_drop3[[ii]] <- colnames(xtrain[,idx])[flc$remove]
    }
    msg(ii)
  }
  columns_to_drop3 <- unique(unlist(columns_to_drop3))
  total_drop <- which(colnames(xtrain) %in% columns_to_drop3)
  xtrain <- xtrain[, -total_drop]
  xtest <- xtest[, -total_drop]

}

## factor handling  
isTrain <- 1:nrow(xtrain_fc); xdat_fc <- rbind(xtrain_fc, xtest_fc); rm(xtrain_fc, xtest_fc)

# SFSG # 
# timestamp columns
{
  time_cols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159",
                 "VAR_0166","VAR_0167","VAR_0168","VAR_0169","VAR_0176","VAR_0177",
                 "VAR_0178","VAR_0179","VAR_0204","VAR_0217")
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
    msg(ii)
  }
  nof_years <- last_year - initial_year
  # cleanup time-related ones
  # initial/last year - drop the latter, as its either 0 or 2014 
  xdat_fc$initial_year <- initial_year; rm(initial_year); rm(last_year)
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

# true / false cases
{
  tf_columns <- c("VAR_0008","VAR_0009","VAR_0010","VAR_0011","VAR_0012",
                  "VAR_0043","VAR_0196","VAR_0226","VAR_0229","VAR_0230",
                  "VAR_0232","VAR_0236","VAR_0239")
  for (ff in tf_columns)
  {
    x <- xdat_fc[,ff];   x[x == ""] <- "mis"
    x <- factor(x);  xdat_fc[,ff] <- x 
    msg(ff)
  }
  xtf <- xdat_fc[,tf_columns]; xdat_fc <- xdat_fc[,-which(colnames(xdat_fc) %in% tf_columns)]
  x <- xtf; for (ii in 1:ncol(x)) x[,ii] <- as.integer(x[,ii])
  flc <- findLinearCombos(x); xtf <- xtf[,-flc$remove]
  xdat_fc <- data.frame(xdat_fc, xtf)
  rm(x, xtf, flc)
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
  xdat_fc$VAR_0200 <- mapped_city
  
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

## reassign
# full training set
xtrain_fc <- xdat_fc[isTrain,]
# move numerical ones fc -> regular
xnames <- c("initial_year","nof_dates","nof_unique_dates","duration_contact",
            "nof_years" ,"same_state","state1_mis","state2_mis","both_state_mis",
            "job1_mis", "job2_mis","both_jobs_mis")
true_fac <- !(colnames(xtrain_fc) %in% xnames)
colnames(xtrain_fc)[true_fac] <- paste("fac", colnames(xtrain_fc)[true_fac] , sep = "")
xtrain <- data.frame(xtrain, xtrain_fc); rm(xtrain_fc)
# full test set
xtest_fc <- xdat_fc[-isTrain,]
colnames(xtest_fc)[true_fac] <- paste("fac", colnames(xtest_fc)[true_fac] , sep = "")
xtest <- data.frame(xtest, xtest_fc)
rm(xtest_fc)

# output three chunks: train, validation and test

# load the validation split
xfolds <- read_csv(file = "./input/xfolds.csv")
isValid <- which(xfolds$valid == 1)
xtrain$ID <- id_train 
xtrain$target <- y

# store pure train set
colnames(xtrain) <- str_replace_all(colnames(xtrain), "_", "")
write_csv(xtrain, path = "./input/xtrain_v5_full.csv")

xvalid <- xtrain[isValid,]
colnames(xvalid) <- str_replace_all(colnames(xvalid), "_", "")
write_csv(xvalid, path = "./input/xvalid_v5.csv")
xtrain <- xtrain[-isValid,]
colnames(xtrain) <- str_replace_all(colnames(xtrain), "_", "")
write_csv(xtrain, path = "./input/xtrain_v5.csv")

xtest$ID <- id_test
colnames(xtest) <- str_replace_all(colnames(xtest), "_", "")
write_csv(xtest, path = "./input/xtest_v5.csv")
