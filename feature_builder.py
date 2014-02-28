from collections import Counter
import collections
from extractor import clean_text
from nltk import word_tokenize
import numpy as np
from scipy.sparse import lil_matrix

class Featurizer(object):
    def __init__(self, story):
        self.story = story
        self.features = {}

    def extract_features(self):
        self.extract_title_features()
        self.extract_text_features()

        return self.features

    def extract_title_features(self):
        # * title_contains_{word}
        # * title_contains_{bigram}
        # * domain_{domain of url}
        # * author_{author}
        # * title_has_dollar_amount
        # * title_has_number_of_years
        # * word count

        author = self.story['extracted_raw_content']['author']
        if author:
            author = author.lower()
            feature_key = "author_" + author.encode('ascii', errors='ignore')
            self.features.setdefault(feature_key, 1)

        word_count = self.story['extracted_raw_content']['word_count']
        if word_count:
            word_count = self.story['word_count']
            self.features.setdefault("word_count", word_count)

        domain = self.story['extracted_raw_content']['domain']

        if domain:
            domain_key = "domain_name_" + domain.encode('ascii', errors='ignore')
            self.features.setdefault(domain_key, 1)

        title = clean_text(self.story['extracted_raw_content']['title'])
        words = word_tokenize(title)
        word_freqs = Counter(words)
        for k,v in word_freqs.iteritems():
            feature_key = "title_contains_" + k
            self.features.setdefault(feature_key, v)

        has_dollar_sign_or_word = '$' in title or 'dollar' in title
        self.features.setdefault('has_dollar_sign_or_word', int(has_dollar_sign_or_word))


    def extract_text_features(self):
        text = clean_text(self.story['extracted_raw_content']['content'])
        words = word_tokenize(text)
        word_freqs = Counter(words)
        for k,v in word_freqs.iteritems():
            feature_key = "content_contains_" + k
            self.features.setdefault(feature_key, v)





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
    for art in arts_with_content:
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


def nb(articles):
    #frequency table, initialize all to 1
    #keys will be tuples (feature_key, target_class), values will be frequency+1
    feature_counts = collections.defaultdict(lambda: 1)
    class_counts = collections.defaultdict(lambda: 0)
    for article in articles:
        try:
            features = Featurizer(article).extract_features()
        except:
            #Some error extracting article, so skip it.
            continue
        target_class = article["favorite"]
        class_counts[target_class] +=1

        for feature, count in features.iteritems():
            feature_counts[(feature, target_class)] += 1


if __name__ == "__main__":
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client.phoenix
    Articles = db.articles
    arts_with_content = Articles.find({'extracted_raw_content':{'$exists':True}})
    print "Number of articles with raw content (before featurizing):", arts_with_content.count()
    #scikit_model(arts_with_content)
    nb(arts_with_content)




