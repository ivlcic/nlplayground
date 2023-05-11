#!/usr/bin/env python

from typing import List
from kl.articles import Articles, Article
from oai.embed import openai_embed
from ttnx.cluster import cluster_louvain, cluster_print
from ttnx.embed import ttnx_embed

if __name__ == "__main__":
    requests = Articles()\
        .filter_customer('7fd935a6-a1f5-42d1-8b5f-048dd54c07d1')\
        .filter_country('SI')

    articles: List[Article] = requests.gets('2023-05-10T08:00:00', '2023-05-11T08:00:00')

    openai_embed(articles, 'openai_embd')
    ttnx_embed(articles, 'ttnx_embd')

    oai_l_clusters = cluster_louvain(articles, 'openai_embd', 0.92)
    ttnx_l_clusters = cluster_louvain(articles, 'ttnx_embd', 0.92)

    print('')
    print('========================== OpenAI ========================== ')
    cluster_print(oai_l_clusters)

    print('')
    print('========================== Textonic ========================== ')
    cluster_print(ttnx_l_clusters)
