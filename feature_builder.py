import math, operator, logging, collections, random
from collections import Counter

import numpy as np
from nltk import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from pymongo import MongoClient
from bson.objectid import ObjectId
from sklearn import metrics
from scipy.sparse import lil_matrix

from extractor import clean_text
from featurizer import Featurizer
from naive_bayes import NaiveBayes
from analyzer import ArticleAnalyzer

def all_features(feature_list):
    all_keys = set()
    length = 0
    for features in feature_list:
        length += len(features.keys())
        keys = set(features.keys())
        all_keys.update(keys)
    print "## Non deduped length: %s, Deduped length: %s" % (length, len(all_keys))

    return all_keys

def build_matrix(feature_list,all_keys):
    num_rows, num_cols = len(feature_list), len(all_keys)
    matrix = lil_matrix((num_rows,num_cols),dtype=int)
    for row, features in enumerate(feature_list):
        for feature_key, feature_count in features.iteritems():
            col_index = all_keys.index(feature_key)

    from pympler.asizeof import asizeof
    print "Size of matrix", asizeof(matrix)
    print "Shape: ", matrix.shape
    return matrix

def scikit_models(articles):
    feature_list = []
    c = 0
    targets = []
    for art in training_set:
        c+=1
        try:
            features = Featurizer(art).extract_features()
            targets.append(art["favorite"])
            feature_list.append(features)
        except Exception as e:
            print e

        if c > 2000:
            break

    #all_features_keys = list(all_features(feature_list))
    #matrix = build_matrix(feature_list,all_features_keys)
    print len(targets), matrix.shape
    #from sklearn.naive_bayes import GaussianNB
    #gnb = GaussianNB()
    #y_pred = gnb.fit(matrix.toarray(), targets)

def score(model, training_set):
    true_pos = 1
    true_neg = 1
    false_pos = 1
    false_neg = 1
    predicted_classes = []

    for article in training_set:
        predicted_class = model.predict(article)
        if predicted_class is not None:
            predicted_classes.append(predicted_class)
            real_class = int(article['favorite'])
            if predicted_class == 1 and real_class == 1:
                true_pos += 1
            elif predicted_class == 1 and real_class == 0:
                false_pos += 1
            elif predicted_class == 0 and real_class == 1:
                false_neg += 1
            else:
                true_neg += 1

    print true_pos,"true positives"
    print true_neg, "true negatives"
    print false_pos, "false positives"
    print false_neg,"false negatives"

    precision = true_pos/float(true_pos+false_pos)
    recall = true_pos/float(true_pos+false_neg)

    f_score = precision*recall/float(precision+recall)

    return f_score

def fit_sklearn_model(x_train, y_train, classifier= MultinomialNB):
    vectorizer = CountVectorizer(stop_words='english', analyzer=ArticleAnalyzer().as_callable())
    X_train = vectorizer.fit_transform(x_train)
    model = classifier().fit(X_train, y_train)
    return model

def clean_articles_for_model(articles):
    """Cleans articles so the output of this can be used by sklearn
    returns documents, targets
    """

    docs = []
    y = []

    for article in articles:
        try:
            text = clean_text(article['extracted_raw_content']['content'])
            docs.append(article)
            y.append(int(article['favorite']))
        except Exception as e:
            logging.info("Exception %s" % e)
            continue

    return {"data": docs, "target": y}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = MongoClient('mongodb://localhost:27017/')
    db = client.phoenix
    Articles = db.articles
    #training_set = Articles.find({'extracted_raw_content':{'$exists':True}})#.limit(100)

    ids = db.users.find_one({'_id':'avyfain'})['articles_ids']
    training_set  = Articles.find({'_id': {'$in': ids},
        'extracted_raw_content':{'$exists':True}})

    all_articles = list(training_set)
    random.seed(4)
    random.shuffle(all_articles)
    dataset_size = len(all_articles)
    partition_index = int(dataset_size * 0.8)
    data = clean_articles_for_model(all_articles)
    x_train = data['data'][:partition_index]
    x_test =  data['data'][partition_index:]

    y_train = data['target'][:partition_index]
    y_test = data['target'][partition_index:]

    print "Training size: %s, test size: %s" % (len(x_train), len(x_test))

    model = fit_sklearn_model(x_train, y_train, classifier=LogisticRegression)


    vectorizer = CountVectorizer(stop_words='english', analyzer=ArticleAnalyzer().as_callable())
    x_train = vectorizer.fit_transform(x_train)
    print "Number of features: %s" % len(vectorizer.get_feature_names())
    y_pred = model.predict(x_train)
    print "=="*20, "\nTraining Set Report. Model: MultinomialNB \n", "--"*20
    report = metrics.classification_report(y_train, y_pred)
    print report

    #vectorizer is already fitted with train data
    x_test = vectorizer.transform(x_test)
    y_pred = model.predict(x_test)
    print "="*20, "\nTest Set Report. Model: MultinomialNB\n", "-"*20
    report = metrics.classification_report(y_test, y_pred)
    print report
    #model = NaiveBayes(articles_to_score)
    # print "We have built a model"

    # initial_score = score(model, articles_to_score)
    # print "Our model has an F-score of", initial_score