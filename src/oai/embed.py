import logging
import openai

from typing import List
from kl.articles import Article
from oai.tokenize import truncate_text_tokens

logger = logging.getLogger('oai.embed')

EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CTX_LENGTH = 8191


def openai_embed(articles: List[Article], embed_field_name: str):
    for a in articles:
        if a.from_cache('data'):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                logger.debug('Loaded %s article OpenAI embedding from cache.', a)
                continue
        logger.debug('Loading %s article OpenAI embedding ...', a)
        tokens = truncate_text_tokens(a.title + ' ' + a.body, EMBEDDING_ENCODING, EMBEDDING_CTX_LENGTH)
        embedding = openai.Embedding.create(  # call OpenAI
            input=tokens, model="text-embedding-ada-002"
        )
        logger.info('Loaded %s article OpenAI embedding.', a)
        a.data[embed_field_name] = embedding["data"][0]["embedding"]  # extract vector from response
        a.to_cache('data')  # cache article to file
