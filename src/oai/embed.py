import logging
import openai
import oai.constants as oai_const

from typing import List
from kl.articles import Article
from oai.tokenize import truncate_text_tokens

logger = logging.getLogger('oai.embed')


def openai_embed(articles: List[Article], embed_field_name: str, cache: bool = True):
    for a in articles:
        if a.from_cache('data'):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                logger.debug('Loaded %s article OpenAI embedding from cache.', a)
                continue
        logger.debug('Loading %s article OpenAI embedding ...', a)
        tokens = truncate_text_tokens(
            a.title + ' ' + a.body,
            oai_const.EMBEDDING_ENCODING,
            oai_const.EMBEDDING_CTX_LENGTH
        )
        embedding = openai.Embedding.create(  # call OpenAI
            input=tokens, model="text-embedding-ada-002"
        )
        logger.info('Loaded %s article OpenAI embedding.', a)
        a.data[embed_field_name] = embedding["data"][0]["embedding"]  # extract vector from response
        if cache:
            a.to_cache('data')  # cache article to file
