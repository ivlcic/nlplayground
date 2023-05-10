import os
import openai

from typing import List
from kl.articles import Articles, Article
from oai.summarize import summarize_article

openai.organization = os.environ['OPENAI_ORG']
openai.api_key = os.environ['OPENAI_API_KEY']


requests = Articles()\
    .filter_customer('7fd935a6-a1f5-42d1-8b5f-048dd54c07d1')\
    .filter_country('SI')

articles: List[Article] = requests.gets('2023-05-09')

#summarize_article(articles, summary_field_name='summary_d3_512_np')

prompt = f'''napiši povzetek v slovenščini za sledeči dokument:\n\"\"\"<document>\"\"\"\n\n'''
summarize_article(articles, max_tokens=512, summary_field_name='summary_d3_512', prompt_template=prompt)

prompt = f'''napiši povzetek v slovenščini za sledeči dokument:\n\"\"\"<document>\"\"\"\n\n'''
summarize_article(articles, max_tokens=256, summary_field_name='summary_g4_512', prompt_template=prompt, model='gpt-4')

brejk = 'point'