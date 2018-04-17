from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sklearn.datasets as skl

x = skl.load_svmlight_file("url_svmlight/Day0.svm")

print(x.shape())



