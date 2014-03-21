import re, array
from sklearn.feature_extraction.text import _check_stop_list
from nltk.stem import SnowballStemmer

class ArticleAnalyzer():
    """Used to add custom functionality to sklearn's CountVectorizer.
    Besides analyzing article content, also analyzes title
    """
    def __init__(self, excluded_features=[]):
        self.stop_words = self.get_stop_words()
        self.tokenize = self.build_tokenizer()
        self.preprocess = self.build_preprocessor()
        stemmer = stemmer = SnowballStemmer('english')
        self.stem = lambda tokens: [stemmer.stem(token) for token in tokens]
        self.excluded_features = excluded_features

    def extract_features(self, article):
        features = []
        content_text = article['extracted_raw_content']['content']
        content_features = self._word_ngrams(self.stem(self.tokenize(
                                self.preprocess(self.decode(content_text)))),
                                "content_contains_",
                                self.stop_words)
        content_features = [f for f in content_features if f not in self.excluded_features]
        features += content_features


        #add author to features
        try:
            author = article['extracted_raw_content']['author']
            if author:
                author = author.lower()
                author_feature = "author_" + self.decode(author)
                features += author_feature
        except Exception as e:
            print e

        #add title features
        try:
            title = article['extracted_raw_content']['title']
            if title:
                title = title.lower()
                title_features = self._word_ngrams(self.tokenize(self.preprocess
                            (self.decode(title))), 'title_contains_', self.stop_words)
                features += title_features
        except Exception as e:
            print e

        #add domain feature
        try:
            domain = article['extracted_raw_content']['domain']
            if domain:
                domain_feature = "domain_is_" + domain
            features += domain_feature
        except Exception as e:
            print e

        #add word lenght as a feature
        try:
            word_count = int(article['extracted_raw_content']['word_count'])
            if word_count:
                if word_count < 500:
                    word_count_feature = "article_is_short"
                elif word_count < 1200:
                    word_count_feature = "article_is_medium"
                else:
                    word_count_feature = "article_is_long"

                features += word_count_feature
        except Exception as e:
            print e

        return features


    def as_callable(self):
        return lambda article: self.extract_features(article)

    def get_stop_words(self):
          """Build or fetch the effective stop words list"""
          return _check_stop_list('english')

    def build_tokenizer(self):
        re_token_pattern=r"(?u)\b\w\w+\b"
        token_pattern = re.compile(re_token_pattern)
        return lambda doc: token_pattern.findall(doc)

    def build_preprocessor(self):
        """Return a function to preprocess the text before tokenization"""
        noop = lambda x: x
        lowercase = True
        # accent stripping
        strip_accents = noop

        if lowercase:
            return lambda x: strip_accents(x.lower())
        else:
            return strip_accents

    def decode(self, doc):
        encoding = "ascii"
        decode_error='ignore'
        return doc.encode(encoding, decode_error)

    def _word_ngrams(self, tokens, feature_prefix, stop_words=None,):
        """Turn tokens into a sequence of n-grams after stop words filtering
            https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L114
        """
        # handle stop words
        if stop_words is not None:
            tokens = [feature_prefix+w for w in tokens if w not in stop_words]

        # handle token n-grams
        self.ngram_range=(1, 1)
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens.append(" ".join(original_tokens[i: i + n]))

        return tokens