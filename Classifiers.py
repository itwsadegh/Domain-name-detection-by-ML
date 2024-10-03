import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)


def chooseClassifier(Xtrain, Ytrain, Xtest, Ytest, classifier):
    sc = StandardScaler()
    Xtrain = sc.fit_transform(Xtrain)
    Xtest = sc.transform(Xtest)

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)

    if classifier == 'logistic regression':
        model = LogisticRegression()

    elif classifier == 'Naive Bayes':
        model = GaussianNB()

    elif classifier == 'Decision Tree':
        model = DecisionTreeClassifier()

    elif classifier == 'AdaBoost':
        model = AdaBoostClassifier()

    elif classifier == 'Random Forest':
        model = RandomForestClassifier(n_estimators = 100)

    model.fit(Xtrain, Ytrain)
    expected = Ytest
    predicted = model.predict(Xtest)
    accuracy = accuracy_score(expected, predicted)
    recall = recall_score(expected, predicted, average = 'macro')
    precision = precision_score(expected, predicted, average = 'macro')
    f1 = f1_score(expected, predicted, average = 'macro')
    cm = metrics.confusion_matrix(expected, predicted)
    #print(cm)

    print("----------------------------------------------")
    print("accuracy")
    print("%.3f" %accuracy)
    print("precision")
    print("%.3f" %precision)
    print("racall")
    print("%.3f" %recall)
    print("f1score")
    print("%.3f" %f1)

