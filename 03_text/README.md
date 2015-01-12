# 3: Text Classification #

# Dependencies #
* Python 2.7.x
* Module: See `requirements.txt`

# Usage #
## Python Modules ##
```
pip install -r requirements.txt
```

## Preprocessing Data ##
Download data from [the official competition page](http://universityofbigdata.net/competition/5681717746597888) and rename it `data.tar.gz`.
```
tar xvzf data.tar.gz
make
```

## Run solvers ##
Run SVM and Gradient Boosting Classifier. This process requires more than 10 hours.
```
python solvers/svm.py data/bag-of-words/train_tfidf.svmlight data/bag-of-words/test_tfidf.svmlight -o pred_svm -v
python solvers/gbc.py data/bag-of-words/train_tfidf_lsi300.svmlight data/bag-of-words/test_tfidf_lsi300.svmlight -o pred_gbc -v
```

## Blend results ##
```
python scripts/blend.py pred_svm pred_gbc -o pred
```

Submit `pred` and be happy!
