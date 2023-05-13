#!/usr/bin/env python

from typing import List
from kl.articles import Articles, Article
from oai.embed import openai_embed
from ttnx.cluster import cluster_louvain, cluster_print, cluster_compare
from ttnx.constants import TTNX_AVG_SQUEEZE, TTNX_AVG_SENTENCE, TTNX_AVG_TRUNCATE, TTNX_AVG_NONE
from ttnx.embed import ttnx_embed

if __name__ == "__main__":
    # .filter_customer('7fd935a6-a1f5-42d1-8b5f-048dd54c07d1')\
    # .filter_customer('a65c7372-9fbe-410c-93d7-4613d26488e7')\
    requests = Articles()\
        .filter_customer('a6f60ee6-f990-4620-8508-c3a6f6cc0dc1')\
        .filter_country('SI')

    articles: List[Article] = requests.gets('2023-05-10T08:00:00', '2023-05-12T08:00:00')

    openai_embed(articles, 'openai_embd')
    # ttnx_embed(articles, 'ttnx_embd', cache=False, average_t=TTNX_AVG_NONE)  # old mode
    ttnx_embed(articles, 'ttnx_embd', cache=False)

    oai_l_clusters = cluster_louvain(articles, 'openai_embd', 0.92)
    ttnx_l_clusters = cluster_louvain(articles, 'ttnx_embd', 0.74)

    print('')
    print('========================== OpenAI ========================== ')
    cluster_print(oai_l_clusters)

    print('')
    print('========================== Textonic ========================== ')
    cluster_print(ttnx_l_clusters)

    print('')
    print('========================== Compare clustering ========================== ')
    cluster_compare(oai_l_clusters, ttnx_l_clusters, 'OpenAI', 'Textonic')
