from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score

train_X, train_y = datasets.load_svmlight_file("url_svmlight/Day17.svm", n_features=3231952)
clf = svm.SVC(kernel="linear")
clf.fit(train_X, train_y)

test_X, test_y = datasets.load_svmlight_file("url_svmlight/Day54.svm", n_features=3231952)

print(accuracy_score(test_y, clf.predict(test_X)))