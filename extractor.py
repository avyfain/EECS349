import requests
from pymongo import MongoClient
from bs4 import BeautifulSoup
from readability import ParserClient

client = MongoClient('mongodb://localhost:27017/')
db = client.phoenix
Articles = db.articles

def batch_extractor():
    """
        This will go through the entire list of articles in the DB, and will try to get Aylien info for each
    """
    articles = [Articles.find_one()]
    for a in articles:
        print "In", a
        if not Article.been_extracted(a):
            data = Article.extract(a)
            print data
            if data:
                Article.store_text_extraction(a, data)


class Article(object):

    @staticmethod
    def been_extracted(a):
        return "extracted_results" in a

    @staticmethod
    def extract(a):
        #data = Aylien().extract(a["resolved_url"])
        parser_client = ParserClient('0ae1d8bed72a91ed706dcf9f354a0db4b430cb47')
        parser_response = parser_client.get_article_content('http://www.theatlantic.com/entertainment/archive/2014/02/russias-gold-medal-figure-skaters-celeb-relationship-status-pioneers/283804')
        article = parser_response.content['content']

        soup = BeautifulSoup(article, "lxml")
        text = soup.get_text()
        #TODO

for k, v in parser_response.content.iteritems():
    if k in ['title', 'dek']:
        text = text + v
        return data

    @staticmethod
    def store_text_extraction(a, data):
        _id = a["_id"]
        db.articles.update({'_id':_id},
            {"$set":{"extracted_results":data}})



if __name__ == "__main__":
    batch_extractor()
