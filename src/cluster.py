import os
import openai

from typing import List
from kl.articles import Articles, Article
from oai.embed import openai_embed
from ttnx.cluster import cluster_louvain

openai.organization = os.environ['OPENAI_ORG']
openai.api_key = os.environ['OPENAI_API_KEY']


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



