import requests

from pymongo import MongoClient
from bs4 import BeautifulSoup
from readability import ParserClient

client = MongoClient('mongodb://localhost:27017/')
db = client.phoenix
Articles = db.articles

def batch_extractor():
    """
        This will go through the entire list of articles in the DB and extract
        content data from readability
    """
    articles = Articles.find()
    for a in articles:
        if not Article.been_extracted(a):
            try:
                raw_extracted = Article.extract_raw_content(a)
                if raw_extracted:
                    Article.store_raw_extracted(a, raw_extracted)
                    print "Stored", a['title'] if 'title' in a else a['_id']
            except Exception as e:
                print e


class Article(object):

    @staticmethod
    def been_extracted(a):
        return "extracted_raw_content" in a

    @staticmethod
    def extract_raw_content(a):
        #data = Aylien().extract(a["resolved_url"])
        parser_client = ParserClient('0ae1d8bed72a91ed706dcf9f354a0db4b430cb47')
        parser_response = parser_client.get_article_content(a['resolved_url'])
        try:
            content = parser_response.content
            if 'error' in content:
                raise Exception
            return content
        except Exception as e:
            print parser_response
            print parser_response.content
            print e
            return False
        #soup = BeautifulSoup(article)
        #text = soup.get_text()



    @staticmethod
    def store_raw_extracted(a, data):
        _id = a["_id"]
        db.articles.update({'_id':_id},
            {"$set":{"extracted_raw_content":data}})

    @staticmethod
    def clean_story(a):
        clean_title = clean_text(a['extracted_raw_content']['title'])

def clean_text(text):
    """Input is a body of parsed text (no HTML).
    Will be converted to ASCII, punctuation will be removed, lowercase, etc
    """

    #avoid encoding errors. Probably a better way to handle this.
    text = text.encode('ascii', errors='ignore')
    text = text.lower()

    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    text = text.split()
    filtered_words = [w for w in text if not w in stop_words]

    cleaned_text = ' '.join(filtered_words)
    #TODO punctuation

    return cleaned_text

if __name__ == "__main__":
    batch_extractor()
