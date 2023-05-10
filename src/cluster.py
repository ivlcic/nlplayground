import os
import openai

from typing import List
from kl.articles import Articles, Article
from ttnx.cluster import cluster_louvain

openai.organization = os.environ['OPENAI_ORG']
openai.api_key = os.environ['OPENAI_API_KEY']


def openai_embed(articles: List[Article], embed_field_name: str):
    for a in articles:
        if a.from_cache('data'):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                continue
        embedding = openai.Embedding.create(  # call openai
            input=a.title + ' ' + a.body, model="text-embedding-ada-002"
        )
        a.data[embed_field_name] = embedding["data"][0]["embedding"]  # extract vector from response
        a.to_cache('data')  # cache article to file


requests = Articles()\
    .filter_customer('7fd935a6-a1f5-42d1-8b5f-048dd54c07d1')\
    .filter_country('SI')

articles: List[Article] = requests.gets('2023-05-09')
openai_embed(articles, 'openai_embd')
clusters = cluster_louvain(articles, 'openai_embd', 0.92)
for k in clusters.keys():
    articles: List[Article] = clusters[k]
    print("Cluster " + str(k))
    for a in articles:
        print('\t|---' + str(a))



