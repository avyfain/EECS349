# -*- coding: utf-8 -*-
import nltk
import string
from nltk.collocations import *
from nltk.stem.wordnet import WordNetLemmatizer

from bs4 import BeautifulSoup
from readability import ParserClient

parser_client = ParserClient('0ae1d8bed72a91ed706dcf9f354a0db4b430cb47')
parser_response = parser_client.get_article_content('http://www.theatlantic.com/entertainment/archive/2014/02/russias-gold-medal-figure-skaters-celeb-relationship-status-pioneers/283804')
article = parser_response.content['content']

soup = BeautifulSoup(article, "lxml")
text = soup.get_text()

for k, v in parser_response.content.iteritems():
	if k in ['title', 'dek']:
		text = text + v

exclude = set(string.punctuation+'”'+'’')
text = ''.join(ch for ch in text if ch not in exclude and ch in string.printable).lower()

words = nltk.word_tokenize(text)
filtered_words = [w for w in words if not w in nltk.corpus.stopwords.words('english')]

for w in filtered_words:
	print w