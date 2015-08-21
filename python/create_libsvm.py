# -*- coding: utf-8 -*-
__author__ = 'michaelpearmain'


from csv import DictReader
# target files

# Data locations
projPath = "/home/michaelpearmain/Kaggle/springleaf/"

with open(projPath + 'input/xtrain.libsvm',"wb") as outfile:
    for linenr, row in enumerate( DictReader(open(projPath + 'input/xtrain_v1.csv',"rb")) ):
        n_d = ""
        for kk in row.keys():
            if kk == 'ID':
                next
            else:
                if kk == 'target':
                    label = 2 * int(row["target"]) - 1
                else:
                    # one-hot encode everything with hash trick
                    n_d += " %s:1"%(abs(hash(kk + '_' + row[kk])))
        outfile.write("%s %s \n"%(label, n_d))

print("Finished Training data conversion")
print("Starting Test data conversion")

with open(projPath + 'input/xtest.libsvm',"wb") as outfile:
    for linenr, row in enumerate( DictReader(open(projPath + 'input/xtest_v1.csv',"rb")) ):
        n_d = ""
        for kk in row.keys():
            if kk == 'ID':
                next
            else:
                # one-hot encode everything with hash trick
                n_d += " %s:1"%(abs(hash(kk + '_' + row[kk])))
                label = 0
        outfile.write("%s %s \n"%(label, n_d))
