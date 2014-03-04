import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, RandomizedLogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

clfs = [LogisticRegression, MultinomialNB, SVC, LinearSVC, RidgeClassifier, Perceptron,KNeighborsClassifier,NearestCentroid]
clfs_names = map(lambda x: type(x()).__name__ , clfs)


def hail_mary(x_train, x_test, y_train, y_test):
    from feature_builder import init_vectorizer, fit_sklearn_model

    full_x_train, vectorizer = init_vectorizer(x_train)
    full_x_test = vectorizer.transform(x_test)
    num_features = full_x_train.shape[1]
    score_funs = [metrics.f1_score,
        metrics.precision_score,
        metrics.recall_score,
        metrics.accuracy_score]
    k_percentages = [0.25, 0.5, 0.75, 1]
    scores = np.zeros((len(clfs), len(score_funs), 2,  len(k_percentages)))
    for clf, clf_name in zip(clfs,clfs_names):
        for percentage_i, k_percentage in enumerate(k_percentages):
            k = int(num_features * k_percentage)
            ch2 = SelectKBest(chi2 , k=k)
            vectorizer = ch2
            #x_train = vectorizer.fit_transform(full_x_train, y_train)
            dsets = [("Training", full_x_train, y_train), ("Test",full_x_test, y_test)]
            vectorized_x_train = vectorizer.fit_transform(full_x_train, y_train)
            model = fit_sklearn_model(vectorized_x_train, y_train, classifier=clf)
            for i, (name, x, y) in enumerate(dsets):
                transformed_x = vectorizer.transform(x)

                y_pred = model.predict(transformed_x)
                for ii, score_fun in enumerate(score_funs):
                    score = metrics.f1_score(y, y_pred)
                    scores[clfs_names.index(clf_name),ii,i,percentage_i] = score

    for fun_i, fun in enumerate(score_funs):
        print "===" * 20, "\n%s\n"%fun.__name__, "---"*20
        for dset_i, (dset_name,x,y) in enumerate(dsets):
            print "-" * 20, "\n%s\n"%dset_name, "-"*20
            print "%20s"%""," ".join(["%5.2f" % p for p in k_percentages])
            for i, clf_name in enumerate(clfs_names):
                print("%20s"%clf_name)," ".join(["%5.2f" % scores[i][fun_i][dset_i][k] for k in range(len(k_percentages))])
