from collections import Counter
from extractor import clean_text
from nltk import word_tokenize

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