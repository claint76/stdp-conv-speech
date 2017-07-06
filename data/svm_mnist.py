#!/usr/bin/env python

import gzip
import numpy as np
import pickle
from sklearn import svm


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
train_set = (np.append(train_set[0], valid_set[0], axis=0), np.append(train_set[1], valid_set[1], axis=0))

clf = svm.SVC(kernel='linear')
clf.fit(train_set[0], train_set[1])
score = clf.score(test_set[0], test_set[1])
print(score)
