#!/usr/bin/env python

from typing import List
from kl.articles import Articles, Article
from oai.embed import openai_embed
from ttnx.cluster import cluster_louvain, cluster_print, cluster_compare, cluster_ttxn
from ttnx.constants import TTNX_AVG_SQUEEZE, TTNX_AVG_SENTENCE, TTNX_AVG_TRUNCATE, TTNX_AVG_NONE, TTNX_WEIGHT_NEG_LIN
from ttnx.embed import ttnx_embed

if __name__ == "__main__":
    requests = Articles()
    requests.filter_customer('a65c7372-9fbe-410c-93d7-4613d26488e7')
    requests.filter_country('SI')
    requests.field('vector_768___textonic_v1')

    articles: List[Article] = requests.gets('2023-05-31T07:50:00', '2023-06-01T07:50:00')

    openai_embed(articles, 'openai_embd')
    # ttnx_embed(articles, 'ttnx_embd', cache=False, average_t=TTNX_AVG_NONE)  # old mode
    # ttnx_embed(articles, 'ttnx_embd', cache=False, average_t=TTNX_AVG_SQUEEZE, weight_t=TTNX_WEIGHT_NEG_LIN)  # alt mode
    ttnx_embed(articles, 'ttnx_embd', cache=False)

    oai_l_clusters = cluster_louvain(articles, 'openai_embd', 0.92)
    ttnx_l_clusters = cluster_louvain(articles, 'ttnx_embd', 0.79)
    #ttnx_l_clusters = cluster_ttxn(articles, 'vector_768___textonic_v1', 0.84)

    print('')
    print('========================== OpenAI ========================== ')
    cluster_print(oai_l_clusters)

    print('')
    print('========================== Textonic ========================== ')
    cluster_print(ttnx_l_clusters)

    print('')
    print('========================== Compare clustering ========================== ')
    cluster_compare(oai_l_clusters, ttnx_l_clusters, 'OpenAI', 'Textonic')