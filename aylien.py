import requests

class Aylien(object):
    def __init__(self):
        self.MASHAPE_KEY = "KEY HERE"
        self.base_url = "https://aylien-text.p.mashape.com/"

    def extract(self, url, store_locally=True):
        endpoint = "extract"
        data = {"url": url}
        try:
            req = self._make_request(endpoint,data)
            data = req.json()
            if store_locally:
                pass
            return data
        except Exception as e:
            print e
            return False

    def _make_request(self, endpoint,data):
        headers = {"X-Mashape-Authorization": self.MASHAPE_KEY}
        url = self.base_url + endpoint
        return requests.get(url, data=data, headers=headers)