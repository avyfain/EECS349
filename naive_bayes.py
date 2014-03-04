import math, collections
from featurizer import Featurizer
import logging
class NaiveBayes():
    def __init__(self, articles):
        self.articles = articles
        self.fit()

    def fit(self, offline=True):
        self.feature_names = set([])
        #frequency table, initialize all to 1
        #keys will be tuples (feature_key, target_class), values will be frequency+1
        self.feature_counts = collections.defaultdict(lambda: 1)
        self.class_counts = collections.defaultdict(lambda: 1)
        logging.info("Article count %s" % len(self.articles))
        for article in self.articles:
            try:
                features = Featurizer(article).extract_features()
                for f in features:
                    self.feature_names.add(f)
                target_class = article["favorite"]
                self.class_counts[target_class] += 1
                # print target_class, self.class_counts
            except Exception as e:
                #Some error extracting article, so skip it.
                print "Exception:",e
                continue

            for feature, count in features.iteritems():
                self.feature_counts[(feature, target_class)] += 1
        self.feature_set_length = len(self.feature_names)

        if offline:
            self.build_probabilities()

    def build_probabilities(self):
        self.probs = {}
        for target_class in ['0','1']:

            for feature in self.feature_names:
                numerator = self.feature_counts[(feature, target_class)]
                denominator = (self.class_counts[target_class] + self.feature_set_length)
                self.probs[(target_class, feature)] = numerator / float(denominator)

    def predict(self,article_list):
        predicted_classes = []
        for article in article_list:
            try:
                article_features = Featurizer(article).extract_features()
            except:
                return None

            p_article = {}
            for target_class in self.class_counts.keys():
                multiplication_total = 0
                for feature in article_features:
                    if feature in self.feature_names:
                        multiplication_total += math.log(self.probs[(target_class, feature)])

                all_class_counts = float(sum(self.class_counts.values()))
                prior = self.class_counts[target_class] / all_class_counts
                p_article[target_class] = math.log(prior) + multiplication_total

            estimated_class = min(p_article.iteritems(), key=operator.itemgetter(1))[0]
            return int(estimated_class)



    def clasify_online(self, article):
        try:
            article_features = Featurizer(article).extract_features()
        except:
            return None
        #MAP Estimate per feature (number of favorites with feature + 1)/(number of favorites + featureset length)
        p_article = {}
        for target_class in ['0','1']:
            probs = {}
            prior = self.class_counts[target_class]/float((self.class_counts['1']+self.class_counts['0']))
            for feature in article_features:
                if feature in self.feature_names:
                    numerator = self.feature_counts[(feature, target_class)]
                    denominator = (self.class_counts[target_class]+self.feature_set_length)
                    probs[feature] = numerator/float(denominator)
            tot = 0
            for p in probs.values():
                tot += math.log(p)
            p_article[target_class] = math.log(prior) + tot
            estimated_class = min(p_article.iteritems(), key=operator.itemgetter(1))[0]
        return int(estimated_class)
