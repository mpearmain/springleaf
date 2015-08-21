## wd etc ####
require(Metrics)
require(caret)
require(readr)
require(doParallel)
require(stringr)

## extra functions ####
# print a formatted message
msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
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
rm(xdat_fc, xtrain, xtrain_fc, xtest, xtest_fc)

## prepare dataset v2 ####
# all factors mapped to model matrices
xtrain <- read_csv(file = "./input/train.csv")
id_train <- xtrain$ID; xtrain$ID <- NULL
y <- xtrain$target; xtrain$target <- NULL

xtest <- read_csv(file = "./input/test.csv")
id_test <- xtest$ID; xtest$ID <- NULL

## preliminary preparation
# drop constant columns
is_missing <- colSums(is.na(xtrain))
constant_columns <- which(is_missing == nrow(xtrain))
xtrain <- xtrain[,-constant_columns]
xtest <- xtest[,-constant_columns]
rm(is_missing, constant_columns)

# check column types
is_missing <- which(colSums(is.na(xtrain)) > 0)
xtrain[is.na(xtrain)] <- -1; xtest[is.na(xtest)] <- -1
col_types <- unlist(lapply(xtrain, class))
fact_cols <- which(col_types == "character")

xtrain_fc <- xtrain[,fact_cols]; xtrain <- xtrain[, -fact_cols]
xtest_fc <- xtest[,fact_cols]; xtest <- xtest[, -fact_cols]

# SFSG # 
# handle factors - case by case, unfortunately, as shit is all over the place
# (quite literally in some cases)
isTrain <- 1:nrow(xtrain_fc)
xdat_fc <- rbind(xtrain_fc, xtest_fc); rm(xtrain_fc, xtest_fc)
tf_columns <- c("VAR_0008","VAR_0009","VAR_0010","VAR_0011","VAR_0012",
                "VAR_0043","VAR_0196","VAR_0226","VAR_0229","VAR_0230","VAR_0232","VAR_0236","VAR_0239")
# first: true / false cases
for (ff in tf_columns)
{
  x <- xdat_fc[,ff];   x[x == ""] <- "mis"
  x <- factor(x);   x <- model.matrix(~ x - 1)
  colnames(x) <- paste(ff, colnames(x), sep = "_")
  col_to_drop <- which.min(colSums(x))
  x <- x[,-col_to_drop, drop = F]
  xdat_fc <- data.frame(xdat_fc, x)
  msg(ff)
}
xdat_fc <- xdat_fc[, -which(colnames(xdat_fc) %in% tf_columns)]
# VAR_0001
ff <- "VAR_0001"
x <- xdat_fc[,ff];   x[x == ""] <- "mis"
x <- factor(x);   x <- model.matrix(~ x - 1)
colnames(x) <- paste(ff, colnames(x), sep = "_")
col_to_drop <- which.min(colSums(x))
x <- x[,-col_to_drop, drop = F]
xdat_fc <- data.frame(xdat_fc, x)
xdat_fc$VAR_0001 <- NULL
# VAR_0005
ff <- "VAR_0005"
x <- xdat_fc[,ff];   x[x == ""] <- "mis"
x <- factor(x);   x <- model.matrix(~ x - 1)
colnames(x) <- paste(ff, colnames(x), sep = "_")
col_to_drop <- which.min(colSums(x))
x <- x[,-col_to_drop, drop = F]
xdat_fc <- data.frame(xdat_fc, x)
xdat_fc$VAR_0005 <- NULL
# VAR_0044
ff <- "VAR_0044"
x <- xdat_fc[,ff];   x[x == ""] <- "mis"
x <- factor(as.integer(factor(x)));   x <- model.matrix(~ x - 1)
colnames(x) <- paste(ff, colnames(x), sep = "_")
col_to_drop <- which.min(colSums(x))
x <- x[,-col_to_drop, drop = F]
xdat_fc <- data.frame(xdat_fc, x)
xdat_fc$VAR_0044 <- NULL  
# VAR_0237 - state info - but it appears twice?
xdat_fc$equal_states <- (xdat_fc$VAR_0237 == xdat_fc$VAR_0274) + 0
ff <- "VAR_0237"
x <- xdat_fc[,ff];   x[x == ""] <- "mis"
x <- factor(x);   x <- model.matrix(~ x - 1)
colnames(x) <- paste(ff, colnames(x), sep = "_")
col_to_drop <- grep("mis", colnames(x))
x <- x[,-col_to_drop, drop = F]
xdat_fc <- data.frame(xdat_fc, x)
xdat_fc$VAR_0237 <- NULL  
# VAR_0274 - state info 
ff <- "VAR_0274"
x <- xdat_fc[,ff];   x[x == ""] <- "mis"; x[x == -1] <- "mis"
x <- factor(x);   x <- model.matrix(~ x - 1)
colnames(x) <- paste(ff, colnames(x), sep = "")
col_to_drop <- grep("mis", colnames(x))
x <- x[,-col_to_drop, drop = F]
xdat_fc <- data.frame(xdat_fc, x)
xdat_fc$VAR_0274 <- NULL  
# VAR_1934
ff <- "VAR_1934"
x <- xdat_fc[,ff];   x[x == ""] <- "mis"; x[x == -1] <- "mis"
x <- factor(x);   x <- model.matrix(~ x - 1)
colnames(x) <- paste(ff, colnames(x), sep = "")
col_to_drop <- which.min(colSums(x))
x <- x[,-col_to_drop, drop = F]
xdat_fc <- data.frame(xdat_fc, x)
xdat_fc$VAR_1934 <- NULL 
# skip city = VAR_0200
xdat_fc$VAR_0200 <- NULL
# VAR_0202
ff <- "VAR_0202"
x <- xdat_fc[,ff];   x[x == ""] <- "mis"; x[x == -1] <- "mis"
x <- factor(x);   x <- model.matrix(~ x - 1)
colnames(x) <- paste(ff, colnames(x), sep = "")
col_to_drop <- which.min(colSums(x))
x <- x[,-col_to_drop, drop = F]
xdat_fc <- data.frame(xdat_fc, x)
xdat_fc$VAR_0202 <- NULL 
# VAR_0222
ff <- "VAR_0222"
x <- xdat_fc[,ff];   x[x == ""] <- "mis"; x[x == -1] <- "mis"
x <- factor(x);   x <- model.matrix(~ x - 1)
colnames(x) <- paste(ff, colnames(x), sep = "")
col_to_drop <- which.min(colSums(x))
x <- x[,-col_to_drop, drop = F]
xdat_fc <- data.frame(xdat_fc, x)
xdat_fc$VAR_0222 <- NULL 
# drop VAR_0404 - title of some sort - 3144 levels
xdat_fc$VAR_0404 <- NULL
# drop VAR_0493 - 797 levels
xdat_fc$VAR_0493 <- NULL
# batch processing of enigma columns
some_columns <- c("VAR_0216","VAR_0283","VAR_0305","VAR_0325",
    "VAR_0342","VAR_0352","VAR_0353","VAR_0354","VAR_0466","VAR_0467")
for (ff in some_columns)
{
  x <- xdat_fc[,ff];   x[x == ""] <- "mis";  x[x == -1] <- "mis"
  x <- factor(x);   x <- model.matrix(~ x - 1)
  colnames(x) <- paste(ff, colnames(x), sep = "_")
  col_to_drop <- which.min(colSums(x))
  x <- x[,-col_to_drop, drop = F]
  xdat_fc <- data.frame(xdat_fc, x)
  msg(ff)
}
xdat_fc <- xdat_fc[, -which(colnames(xdat_fc) %in% some_columns)]
# timestamps
time_cols <- c("VAR_0073","VAR_0075","VAR_0156","VAR_0157","VAR_0158","VAR_0159",
  "VAR_0166","VAR_0167","VAR_0168","VAR_0169","VAR_0176","VAR_0177",
  "VAR_0178","VAR_0179","VAR_0204","VAR_0217")
xdat_fc$missing_dates <- rowSums(xdat_fc[,time_cols] == "")
# drop the time (leave date only) so you can actually see sth
for (ff in time_cols)
{
  x <- xdat_fc[,ff];  x <- str_sub(x,0,-10) 
  x[x == ""] <- "99JAN99"; xdat_fc[,ff] <- x
}
xdat_fc$nof_unique_dates <- apply(xdat_fc[,time_cols],1,function(s) length(unique(s)))
# smallest year within the dates
xdat_fc$first_year <- apply(xdat_fc[,time_cols],1,function(s) min(as.integer(str_sub(as.character(s),-2))))
xdat_fc$latest_year <- apply(xdat_fc[,time_cols],1, extractYear)
xdat_fc$latest_year[is.infinite(xdat_fc$latest_year)] <- 99
xdat_fc <- xdat_fc[, -which(colnames(xdat_fc) %in% time_cols)]

# combine with the numerical part
xtrain_fc <- xdat_fc[isTrain,]
xtrain <- data.frame(xtrain, xtrain_fc)
xtrain$ID <- id_train; xtrain$target <- y
write_csv(xtrain, path = "./input/xtrain_v2.csv")

xtest_fc <- xdat_fc[-isTrain,]
xtest <- data.frame(xtest, xtest_fc)
xtest$ID <- id_test
write_csv(xtest, path = "./input/xtest_v2.csv")
rm(xdat_fc, xtrain, xtrain_fc, xtest, xtest_fc)

# SFSG # 

## prepare dataset v3 ####
# create some more factors (combos of reasonable ones)
