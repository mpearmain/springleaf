# FULL Run for R ensemble 

library(data.table)

# Best model so far.
# Build a list of data sets.
ensemble.data <- list(fread('~/Downloads/baggin5-ThreeBag-25082015.csv'),
                      fread('~/Downloads/rf_minimix_20150824.csv'))
# Rename the datasets to avoid name collisions in the merge.
ensemble.data <- lapply(seq_along(ensemble.data), 
                        function(x) setnames(ensemble.data[[x]], c('ID', paste0('target', x))))

# Merge all the dataset together to ensemble.
full.joins <- Reduce(function(x, y) merge(x, y, by='ID'), ensemble.data)

alpha <- 0.99
full.joins[, target := (alpha * target1) + ((1 - alpha) * target2)]
full.joins <- full.joins[, c('ID', 'target'), with = F]
write.table(full.joins, file = "xgboost3-5-bag-rf-minimix-082515.csv", 
            row.names = F, col.names = T, sep = ",", quote = F)

