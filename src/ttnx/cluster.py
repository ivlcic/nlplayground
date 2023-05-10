import networkx as nx
import numpy as np

from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from kl.articles import Article


def cluster_louvain(articles: List[Article], embed_field_name: str, similarity_threshold: float = 0.90):
    embeddings = []
    [embeddings.append(x.data[embed_field_name]) for x in articles]
    embeddings = np.array(embeddings)
    labels = [0] * len(embeddings)
    x = cosine_similarity(embeddings, embeddings)
    similarity_matrix = x > similarity_threshold
    G = nx.from_numpy_array(similarity_matrix)
    communities = nx.algorithms.community.louvain_communities(G, resolution=0.1)
    for community in communities:
        initial_member = min(community)
        for member in community:
            labels[member] = initial_member

    clusters = {}
    for a, lbl in zip(articles, labels):
        if lbl not in clusters:
            clusters[lbl] = [a]
        else:
            clusters[lbl].append(a)
    clusters = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))
    return clusters