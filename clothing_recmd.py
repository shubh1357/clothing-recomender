from urllib.request import urlopen as u
from bs4 import BeautifulSoup as s
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from numpy.linalg import norm


def clean_text(
        string: str,
        punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
        stop_words=['the', 'a', 'and', 'is', 'be', 'will']) -> str:
    """
    A method to clean text
    """

    string = string.split('|')[0]
    # Cleaning the urls
    string = re.sub(r'https?://\S+|www\.\S+', '', string)
    # Cleaning the html elements
    string = re.sub(r'<.*?>', '', string)
    # Removing the punctuations
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
            # Converting the text to lower
    string = string.lower()
    string = re.sub(r'\w*\d\w*', '', string)
    # Removing stop words
    string = ' '.join([word for word in string.split() if word not in stop_words])
    # Cleaning the whitespaces
    string = re.sub(r'\s+', ' ', string).strip()
    return string


def keyword_process(keyword):
    """

    Parameters
    ----------
    keyword : String
        Search string.

    Returns
    -------
    key : String
        Returns cleaned search string in "key1+key2+key3..." format for products search.

    """
    keyword = clean_text(keyword)

    keyword1 = keyword.split(' ')
    key = ''
    for k in keyword1:
        key = key + k
        key = key + '+'

    return key


def scrap_url(url):
    """

    Parameters
    ----------
    url : string
        Target url for scrapping.

    Returns
    -------
    soup : Json object
        BS4 beautifulSoup json object parsed via html parser.

    """

    client = u(url)
    page = client.read()
    client.close()
    soup = s(page, 'html.parser')
    return soup


def extract_raw_data(keyword):
    """


    Parameters
    ----------
    keyword : string
        clothing search string.

    Returns
    -------
    prod_desc : list of list
        product url and its description.

    """

    # base url
    url = 'https://www.amazon.in/'
    # preprocess keyword
    search_key = keyword_process(keyword)
    # generating target url based on search string
    url = url + 's?k=' + search_key[:-1]
    # scraping data
    soup = scrap_url(url)

    prod_desc = []
    # finding all the cards containing product details and product link.
    containers = soup.findAll("div", {
        "class": "sg-col-4-of-24 sg-col-4-of-12 s-result-item s-asin sg-col-4-of-16 sg-col s-widget-spacing-small sg-col-4-of-20"})

    for con in containers:

        prod = con.findAll('a', {"class": "a-link-normal s-no-outline"})
        p_url = prod[0]['href']
        print(p_url)
        p_url = "https://www.amazon.in" + p_url
        try:
            p_soup = scrap_url(p_url)
        except:
            print("unable to access the url")
            continue

        des = p_soup.findAll('div', {'id': 'feature-bullets'})
        des_text = des[0].text
        product_data = []
        product_data.append(p_url)
        product_data.append(des_text)
        prod_desc.append(product_data)

    return prod_desc


def prod_desc_comp(data):
    """
    input:- search string and product description
    output:- Similarity between the search string and the product string.
    """

    sentence = clean_text(data['desc'])
    keyword = clean_text(data['search_key'])
    # Obtain sentence embeddings through sentence transformers
    model = SentenceTransformer('paraphrase-mpnet-base-v2')
    # Encode all sentences
    sen_embedding = model.encode(sentence)
    key_embedding = model.encode(keyword)
    return np.dot(sen_embedding, key_embedding) / (norm(sen_embedding) * (norm(key_embedding)))


def product_list(search_string):
    """

    Parameters
    ----------
    search_string : string
        Clothing search string.

    Returns
    -------
    TYPE
        List of urls.

    """

    if len(search_string) == 0:
        print("please enter a valid search string")
        return
    raw_data = extract_raw_data(search_string)
    df = pd.DataFrame(raw_data)
    df.columns = ['product_url', 'desc']
    df['search_key'] = search_string
    df['similarity'] = df.apply(prod_desc_comp, axis=1)
    df = df.sort_values(by=['similarity'], ascending=False)

    match_products_url = df.product_url
    return match_products_url[:20]

# while(True):
#    input_string = input("Enter the search string")
#    if len(input_string)==0:
#        print("please enter a valid search string")
#    else:
#        urls = product_list(input_string)
#        print(urls)
#    print("If you want to search another product press 1 else 0")
#    flag = input()
#    if(flag=='0'):
#        break