from sklearn.metrics import accuracy_score
import scipy.sparse as sp
import numpy as np
from sklearn import datasets
from sklearn import svm # SVM classifier
from sklearn import neighbors # KNN classifier
from sklearn import ensemble # Random forest classifier
from sklearn import metrics # Confusion matrix


TRAIN_LENGTH = 3
TEST_LENGTH = 1


def classifier_test(classifiers, source):
    # training = np.random.random_integers(0, 89, TRAIN_LENGTH)
    training = [0, 1, 2]
    print("Training days:", training)
    X, y = datasets.load_svmlight_file(source + "/Day" + str(training[0]) + ".svm", n_features=3300000)
    for day in training[1:]:
        data = datasets.load_svmlight_file(source + "/Day" + str(day) + ".svm", n_features=3300000)
        X = sp.vstack((X, data[0]), format='csr')
        y = np.concatenate((y, data[1]), axis=0)

    print("Beginning fitting.")
    count = 0
    for clf in classifiers:
        print(count)
        clf.fit(X, y)
    print("Done fitting.")

    # testing = np.random.random_integers(90, 120, TEST_LENGTH)
    testing = [90]
    print("Testing days:", testing)
    test_X, test_y = datasets.load_svmlight_file(source + "/testData/Day" + str(testing[0]) + ".svm", n_features=3300000)
    for day in testing[1:]:
        data = datasets.load_svmlight_file(source + "/testData/Day" + str(day) + ".svm", n_features=3300000)
        test_X = sp.vstack((test_X, data[0]), format='csr')
        test_y = np.concatenate((test_y, data[1]), axis=0)

    # Use line below if using voting classifier
    # prediction = classifiers[len(classifiers)-1].predict(test_X)
    prediction = classifiers[0].predict(test_X)
    print("Accuracy:", accuracy_score(test_y, prediction))
    cm = metrics.confusion_matrix(test_y, prediction)
    print(cm)


if __name__ == "__main__":
    # linSVM = svm.SVC(kernel="linear", probability=True)
    # knn = neighbors.KNeighborsClassifier(n_neighbors=15)
    # rfc = ensemble.RandomForestClassifier(n_estimators=8, n_jobs=-1)
    # vote = ensemble.VotingClassifier(estimators=[('svc', linSVM), ('knn', knn), ('rf', rfc)], voting='soft', weights=[3, 1, 1], n_jobs=-1)

    # Second argument decides if full sized or trimmed files are used
    # "url_svmlight" for full - "sample" for trimmed
    classifier_test("hello!", "sample")