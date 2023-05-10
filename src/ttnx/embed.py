import uuid

from typing import List
from kl.articles import Article
from ttnx.api import call_textonic


def __call_ttxn_embed(articles: List[Article], embed_field_name: str):
    request = {
        'requestId': str(uuid.uuid4()),
        'process': {
            'analysis': {
                'steps': [
                    {
                        'step': 'doc_embed'
                    }
                ]
            }
        },
        'documents': []
    }
    for a in articles:
        document = {
            'id': a.uuid,
            'title': a.title,
            'lang': a.language,
            'sections': [
                {
                    'outline': 'headline',
                    'data': a.title
                },
                {
                    'outline': 'body',
                    'data': a.body
                }
            ]
        }
        request['documents'].append(document)
    result = call_textonic(request)
    for res_item, a in zip(result, articles):
        for res in res_item['result']:
            if 'c' in res and 'v' in res and 'doc_embed' in res['c']:
                a.data[embed_field_name] = res['v']


def ttnx_embed(articles: List[Article], embed_field_name: str):
    embed = []
    for a in articles:
        if a.from_cache('data'):  # read from file
            if embed_field_name in a.data:  # we already did the embedding
                continue
        embed.append(a)

    __call_ttxn_embed(embed, embed_field_name)
    for a in embed:
        a.to_cache('data')  # cache article to file



