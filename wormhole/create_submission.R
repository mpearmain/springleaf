library(data.table)

ids <- fread('./input/test.csv', select = 'ID')
preds <- fread('./wormhole/pred.txt')

outfile <- data.table(ids$ID, preds$V1)
setnames(outfile, c('ID', 'target'))
write.table(outfile, file = "./output/asyn-FTRL-25082015.csv", 
            row.names = F, col.names = T, sep = ",", quote = F)