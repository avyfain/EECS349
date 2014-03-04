from collections import Counter
import collections
from extractor import clean_text
from nltk import word_tokenize
import numpy as np
from scipy.sparse import lil_matrix
import math
import operator
import logging
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from pymongo import MongoClient
from bson.objectid import ObjectId
from featurizer import Featurizer
from naive_bayes import NaiveBayes
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
        predicted_class = model.classify_offline(article)
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

def sklearn_model(articles):
    docs = []
    y = []
    from sklearn.datasets import fetch_20newsgroups
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)
    for article in articles:
        try:
            text = clean_text(article['extracted_raw_content']['content'])
            docs.append(text)
            y.append(int(article['favorite']))
        except Exception as e:
            logging.info("Exception %s" % e)
            continue

    vectorizer = TfidfVectorizer(stop_words='english')
    X_train = vectorizer.fit_transform(docs)

    model = MultinomialNB().fit(X_train, y)
    pred = model.predict(X_train)
    report = metrics.classification_report(y, pred)
    score = metrics.f1_score(y, pred)
    print report
    print "F-Score: %s" % score

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = MongoClient('mongodb://localhost:27017/')
    db = client.phoenix
    Articles = db.articles
    #training_set = Articles.find({'extracted_raw_content':{'$exists':True}})#.limit(100)

    ids = db.users.find_one({'_id':'avyfain'})['articles_ids']
    training_set  = Articles.find({'_id': {'$in': ids},
        'extracted_raw_content':{'$exists':True}})

    articles_to_score = list(training_set)
    print "Number of articles with raw content (before featurizing):", training_set.count()

    sklearn_model(articles_to_score)

    model = NaiveBayes(articles_to_score)
    print "We have built a model"

    initial_score = score(model, articles_to_score)
    print "Our model has an F-score of", initial_score