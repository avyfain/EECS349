import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clfs = [LogisticRegression, MultinomialNB]
clfs_names = map(lambda x: type(x()).__name__ , clfs)


def hail_mary(x_train, x_test, y_train, y_test):
    from feature_builder import init_vectorizer, fit_sklearn_model
    full_x_train, vectorizer = init_vectorizer(x_train)
    full_x_test = vectorizer.transform(x_test)
    num_features = full_x_train.shape[1]
    for clf, clf_name in zip(clfs,clfs_names):
        for k_percentage in [0.25, 0.5, 0.75, 1]:
            k = int(num_features * k_percentage)
            ch2 = SelectKBest(chi2 , k=k)
            vectorizer = ch2
            x_train = vectorizer.fit_transform(full_x_train, y_train)
            x_test = vectorizer.transform(full_x_test)

            model = fit_sklearn_model(x_train, y_train, vectorizer, classifier=clf)
            #print "Number of features: %s" % len(vectorizer.get_feature_names())
            y_pred = model.predict(x_train)
            print "=="*20, "\nTraining Set Report. Model: %s. k= %s\n" % (clf_name, k),  "--"*20
            report = metrics.classification_report(y_train, y_pred)
            print report

            #vectorizer is already fitted with train data
            y_pred = model.predict(x_test)
            print "=="*20, "\nTest Set Report. Model: %s. k= %s\n" % (clf_name,k), "--"*20
            report = metrics.classification_report(y_test, y_pred)
            print report