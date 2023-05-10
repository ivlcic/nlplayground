import openai
import tiktoken

from typing import List
from kl.articles import Article

EMBEDDING_ENCODING = 'cl100k_base'
EMBEDDING_CTX_LENGTH = 8191


def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def openai_embed(articles: List[Article], embed_field_name: str):
    for a in articles:
        if a.from_cache('data'):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                continue
        tokens = truncate_text_tokens(a.title + ' ' + a.body)
        embedding = openai.Embedding.create(  # call openai
            input=tokens, model="text-embedding-ada-002"
        )
        a.data[embed_field_name] = embedding["data"][0]["embedding"]  # extract vector from response
        a.to_cache('data')  # cache article to file