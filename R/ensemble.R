# FULL Run for R ensemble 

library(data.table)

# Best model so far.
# Build a list of data sets.
ensemble.data <- list(fread('extratrees.csv'),
                      fread('xgb_autotune.csv'))
# Rename the datasets to avoid name collisions in the merge.
ensemble.data <- lapply(seq_along(ensemble.data), 
                        function(x) setnames(ensemble.data[[x]], c('ID', paste0('target', x))))

full.joins[, click := rowMeans(.SD), by = id]

# Only keep the mean and write out the file.
write.csv(full.joins[, list(id, click), with = TRUE],
          file = './python_ensemble/mp_CampaignMix_20150207.csv', 
          quote = F, 
          row.names = F)

alpha <- 0.8
full.joins[, target := (alpha * target1) + ((1 - alpha) * target2))]
full.joins <- full.joins[, c('ID', 'target'), with = T]
write.table(full.joins, file = "ensemble-17082015.csv", 
            row.names = F, col.names = T, sep = ",", quote = F)

