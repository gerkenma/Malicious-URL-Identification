from sklearn import svm
import classifier_test as ct

clf = svm.SVC(kernel="linear")

ct.classifier_test(clf)