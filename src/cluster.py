import os
import openai
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm

from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from kl.articles import Articles, Article

openai.organization = os.environ['OPENAI_ORG']
openai.api_key = os.environ['OPENAI_API_KEY']


requests = Articles()\
    .filter_customer('7fd935a6-a1f5-42d1-8b5f-048dd54c07d1')\
    .filter_country('SI')

articles: List[Article] = requests.gets('2023-05-09')
embeddings: np.array = np.array([])

for a in articles:
    if not a.from_cache('data'):
        embedding = openai.Embedding.create(
            input=a.title + ' ' + a.body, model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        a.data['openai_embd'] = embedding
        a.openai_embd = np.array(embedding)
        a.to_cache('data')

    np.append(embeddings, a.openai_embd, axis=0)

similarity_threshold = 0.75
labels = [0] * len(embeddings)
# print(labels)
# print(embeddings.size())
similarity_matrix = cosine_similarity(embeddings, embeddings) > similarity_threshold
# print(similarity_matrix.size())
similarity_matrix = similarity_matrix.cpu()
G = nx.from_numpy_array(similarity_matrix.numpy())
communities = nx_comm.louvain_communities(G, resolution=0.1)
# print(communities)
for community in communities:
    initial_member = min(community)
    for member in community:
        labels[member] = initial_member

clusters = [dict(id=a.uuid, titlelabel=str(lbl)) for a, lbl in zip(articles, labels)]

