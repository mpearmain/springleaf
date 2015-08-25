# A simple shell script to shuffle the test data in libsvm format and split for train / validation purposes.

echo 'shuffling training data'
shuf --output train_shuf.libsvm $1
echo 'splitting to train and valid datasets'
split -l 133000 train_shuf.libsvm
rm train_shuf.libsvm
mv xaa train_shuf.libsvm
mv xab valid_shuf.libsvm
echo 'Complete'
