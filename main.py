# This is a sample Python script.

import ipaddress

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import seaborn
import Classifiers
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from matplotlib import pyplot as plt

def preprocessing_():
    train = pd.read_csv('Project2_training.csv')
    validation = pd.read_csv('project2_validation.csv')
    Xtrain = train.iloc[:,0:-1]
    Ytrain = train.iloc[:,-1]
    Xvalid= validation.iloc[:,0:-1]
    Yvalid = validation.iloc[:,-1]
    Xtrain['c_ip'] = Xtrain['c_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
    Xvalid['c_ip'] = Xvalid['c_ip'].apply(lambda x: int(ipaddress.ip_address(x)))
    le = preprocessing.LabelEncoder()
    le.fit(Ytrain)
    Ytrain = le.transform(Ytrain)
    Yvalid = le.transform(Yvalid)

    return Xtrain, Ytrain, Xvalid, Yvalid, le

def EDA(X, Y):
    head = X.head()
    desc = X.describe()
    print(desc)
    seaborn.heatmap(X.corr())
    ifnull_ = X.isnull().sum()  # no null attributes

    # feature selection
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(5).plot.bar()
    plt.show()

    feat_selected = feat_importances.nlargest(5).keys()
    Xnew = X[feat_selected]
    seaborn.heatmap(Xnew.corr())
    headnew = Xnew.head()
    descnew = Xnew.describe()
    print(descnew)

if __name__ == '__main__':
    Xtrain, Ytrain, Xvalid, Yvalid, labelEncoder = preprocessing_()
    EDA(Xtrain, Ytrain)

    # select classifier
    classifier = 'Decision Tree'
    #Classifiers.chooseClassifier(Xtrain, Ytrain, Xvalid, Yvalid, classifier)

