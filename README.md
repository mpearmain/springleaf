# springleaf

Repo for Kaggle contest

  * sklearn
  * pandas
  * numpy

Data lives in /input - but is part of .gitignore, download the data.
All feature engineering is contained with the make_data.py file. 
This should result in a train, validate and test set (with labels) so once can call:

<code>
train, train\_labels, valid, valid\_labels, test, test\_labels = make_data()
</code>

This makes it easy to replicate feature engineering and abstracts the work away from the modelling.


**Data processing**

Primarily conducted in R (due to easier parallelization, handling of factors and missing values)
