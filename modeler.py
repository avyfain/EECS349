import random, logging, warnings

from pymongo import MongoClient

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
from sklearn import cross_validation
from sklearn import metrics

from feature_builder import init_vectorizer, fit_sklearn_model, clean_articles_for_model


clfs = [
    LogisticRegression,
    MultinomialNB,
    LinearSVC,
    RidgeClassifier,
    Perceptron,
    NearestCentroid,
    KNeighborsClassifier
]

clfs_names = map(lambda x: type(x()).__name__ , clfs)

score_funs = [
    metrics.f1_score,
    metrics.precision_score,
    metrics.recall_score,
    metrics.accuracy_score
]

k_percentages = np.arange(0.05,.35,0.05)

random.seed(4)

def cross_validate(all_articles):
    """Repeated random sub-sampling validation
    """
    n = 10 #Number of repetitions

    scores = np.zeros((len(clfs), len(score_funs), 2,  len(k_percentages)))

    dset  = clean_articles_for_model(all_articles)
    data = np.array(dset['data'])
    targets = np.array(dset['target'])
    kf = cross_validation.KFold(len(data), n_folds=n)
    for i, (train_indeces, test_indeces) in enumerate(kf):

        x_train = data[train_indeces]
        x_test =  data[test_indeces]
        y_train = targets[train_indeces]
        y_test = targets[test_indeces]

        print "Starting iter %s" % i
        #add (elementwise) results from each CV iteration
        scores = np.add(scores, cv_iteration(x_train,x_test,y_train,y_test))

    #take average of each-score
    scores = scores / float(n)
    dsets_names = ["training", "test"]
    for fun_i, fun in enumerate(score_funs):
        print "===" * 20, "\n%s\n"%fun.__name__, "---"*20
        for dset_i, dset_name in enumerate(dsets_names):
            print "-" * 20, "\n%s\n"%dset_name, "-"*20
            print "%28s"%""," ".join(["%5.2f" % p for p in k_percentages])
            for i, clf_name in enumerate(clfs_names):
                print("%28s"%clf_name)," ".join(["%5.2f" % scores[i][fun_i][dset_i][k] for k in range(len(k_percentages))])


def cv_iteration(x_train, x_test, y_train, y_test):
    print "Train set: %s with %s favorites. Test set: %s with %s favorites" % \
    (len(x_train), sum([int(t) for t in y_train]), len(x_test), sum([int(t) for t in y_test]))
    full_x_train, vectorizer = init_vectorizer(x_train)
    print "Initial Featureset length: %s" % len(vectorizer.get_feature_names())
    feature_names = vectorizer.get_feature_names()
    full_x_test = vectorizer.transform(x_test)
    num_features = full_x_train.shape[1]
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
                    with warnings.catch_warnings(record=True) as w:
                        score = score_fun(y, y_pred)
                        scores[clfs_names.index(clf_name),ii,i,percentage_i] = score


    #print top ranked features
    top_ranked_features = sorted(enumerate(ch2.scores_),key=lambda x:x[1], reverse=True)[:1000]
    top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
    #print np.array(feature_names)[top_ranked_features_indices]

    return scores



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    client = MongoClient('mongodb://localhost:27017/')
    db = client.phoenix
    Articles = db.articles

    ids = db.users.find_one({'_id':'avyfain'})['articles_ids']
    training_set  = Articles.find({'_id': {'$in': ids},
        'extracted_raw_content':{'$exists':True},
        'status':"1"
        })

    all_articles = list(training_set)[:]
    print "Number of articles: %s" % len(all_articles)

    cross_validate(all_articles)