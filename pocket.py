import json, csv, requests


"""
   First run authorize() to get a url and browse to that to authorize app
"""


CONSUMER_KEY = "23417-b55bf5445152c892b4dbbe22"

TOKENS = {
    #"leon":"204ab5b2-4e64-015c-453e-65268e",
    "avy": "0d3c6a1d-c3ea-2f05-53cb-d23ea8"
}

def get_request_token():
    data = {
        "consumer_key": CONSUMER_KEY,
        "redirect_uri": "http://google.com"
        }
    endpoint = 'v3/oauth/request'
    req = make_request(endpoint, data)
    rtext = req.text
    truncate_at = rtext.find("=") + 1
    code = rtext[truncate_at:]
    print "Code:", code
    return code


def make_request(endpoint, data, type="POST"):
    base_url = "http://getpocket.com/"
    new_url = base_url + endpoint
    if type=="POST":
        return requests.post(new_url,data=data)
    elif type=="GET":
        return requests.get(new_url,params=data)

def authorize():
    token = get_request_token()
    endpoint = 'auth/authorize'
    data = {'request_token': token,
        'redirect_uri':"http://www.google.com"
        }
    url = make_request(endpoint,data, type="GET").url
    print "Go to the following URL to authorize pocket:"
    print url

def get_all_articles(access_token):
    data = {
        "access_token":access_token,
        "state": "all",
        "consumer_key":CONSUMER_KEY
    }
    
    endpoint = "v3/get"
    req = make_request(endpoint,data, type="POST")
    print req.text, req.url, data
    return json.loads(req.text)

def list_to_csv():
    filename = "leon_all_pocket.json"
    op = open("leon_pocket.csv","w")
    with open(filename) as f:
        data = json.loads(f.read())
        article_list = data["list"]
        headers = ['item_id','resolved_id','given_url']
        writer = csv.DictWriter(op,fieldnames=headers, extrasaction='ignore')
        
        #each article is a dict "id":data
        for art in article_list.values():
            writer.writerow(art)

    op.close()

if __name__ == "__main__":
    #authorize()
    
    for n,t in TOKENS.iteritems():
        data = get_all_articles(t)
        json.dump(data,open("{0}_all_pocket.json".format(n),"w"))

    
    #list_to_csv()