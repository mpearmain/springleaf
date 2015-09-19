# FULL Run for R ensemble 

library(data.table)

# Best model so far.
# Build a list of data sets.
# FTRL = 0.77507
# Keras = 0.76636
# RF = 0.77905
# XGB = 0.799
ensemble.data <- list(fread('output/Keras-Mix-10092515.csv'), 
                      fread('output/ftrl1sub.csv')) 
# Rename the datasets to avoid name collisions in the merge.
ensemble.data <- lapply(seq_along(ensemble.data), 
                        function(x) setnames(ensemble.data[[x]], c('ID', paste0('target', x))))

# Merge all the dataset together to ensemble.
full.joins <- Reduce(function(x, y) merge(x, y, by='ID'), ensemble.data)

alpha <- 0.2
full.joins[, target := (alpha * target1) + ((1 - alpha) * target2)]
full.joins <- full.joins[, c('ID', 'target'), with = F]
write.table(full.joins, file = "output/ftrl2-Keras2Bag-1709.csv", 
            row.names = F, col.names = T, sep = ",", quote = F)


# Best model so far.
# Build a list of data sets.
ensemble.data <- list(fread('output/ftrl2-Keras2Bag-1709.csv'),
                      fread('output/rf_minimix_20150824.csv'))
# Rename the datasets to avoid name collisions in the merge.
ensemble.data <- lapply(seq_along(ensemble.data), 
                        function(x) setnames(ensemble.data[[x]], c('ID', paste0('target', x))))

# Merge all the dataset together to ensemble.
full.joins <- Reduce(function(x, y) merge(x, y, by='ID'), ensemble.data)

alpha <- 0.7
full.joins[, target := (alpha * target1) + ((1 - alpha) * target2)]
full.joins <- full.joins[, c('ID', 'target'), with = F]
write.table(full.joins, file = "output/rf-minimix-ftrl2-Keras2Bag-1709.csv", 
            row.names = F, col.names = T, sep = ",", quote = F)


####################

# Best model so far.
# Build a list of data sets.
ensemble.data <- list(fread('output/rf-minimix-ftrl2-Keras2Bag-1709.csv'),
                      fread('output/xmix1_20150907.csv'))
# Rename the datasets to avoid name collisions in the merge.
ensemble.data <- lapply(seq_along(ensemble.data), 
                        function(x) setnames(ensemble.data[[x]], c('ID', paste0('target', x))))

# Merge all the dataset together to ensemble.
full.joins <- Reduce(function(x, y) merge(x, y, by='ID'), ensemble.data)

alpha <- 0.01
full.joins[, target := (alpha * target1) + ((1 - alpha) * target2)]
full.joins <- full.joins[, c('ID', 'target'), with = F]
write.table(full.joins, file = "output/xgboost-rf-minimix-ftrl2-Keras2Bag-1709.csv", 
            row.names = F, col.names = T, sep = ",", quote = F)





