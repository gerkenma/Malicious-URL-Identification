from sklearn.metrics import accuracy_score
import scipy.sparse as sp
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors

TRAIN_LENGTH = 3
TEST_LENGTH = 1


def classifier_test(classifier, source):
    # training = np.random.random_integers(0, 89, TRAIN_LENGTH)
    training = [0, 1, 2]
    print("Training days:", training)
    X, y = datasets.load_svmlight_file(source + "/Day" + str(training[0]) + ".svm", n_features=3300000)
    for day in training[1:]:
        data = datasets.load_svmlight_file(source + "/Day" + str(day) + ".svm", n_features=3300000)
        X = sp.vstack((X, data[0]), format='csr')
        y = np.concatenate((y, data[1]), axis=0)

    print("Beginning fitting.")
    classifier.fit(X, y)
    print("Done fitting.")

    # testing = np.random.random_integers(90, 120, TEST_LENGTH)
    testing = [90]
    print("Testing days:", testing)
    test_X, test_y = datasets.load_svmlight_file(source + "/testData/Day" + str(testing[0]) + ".svm", n_features=3300000)
    for day in testing[1:]:
        data = datasets.load_svmlight_file(source + "/testData/Day" + str(day) + ".svm", n_features=3300000)
        test_X = sp.vstack((test_X, data[0]), format='csr')
        test_y = np.concatenate((test_y, data[1]), axis=0)

    prediction = classifier.predict(test_X)
    print("Accuracy:", accuracy_score(test_y, prediction))
    cm = metrics.confusion_matrix(test_y, prediction)
    print(cm)


if __name__ == "__main__":
    # clf = svm.SVC(kernel="linear")
    clf = neighbors.KNeighborsClassifier(n_neighbors=15)
    # Second argument decides if full sized or trimmed files are used
    # "url_svmlight" for full - "sample" for trimmed
    classifier_test(clf, "sample")