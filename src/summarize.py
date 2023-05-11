#!/usr/bin/env python
import oai.constants as oai_const

from typing import List
from kl.articles import Articles, Article
from oai.summarize import summarize_article


if __name__ == "__main__":
    requests = Articles()\
        .filter_customer('7fd935a6-a1f5-42d1-8b5f-048dd54c07d1')\
        .filter_country('SI')

    articles: List[Article] = requests.gets('2023-05-09')

    # language problematic no prompt TL:DR sample:
    # summarize_article(articles, summary_field_name='summary_d3_512_np')

    prompt = f'''napiši povzetek v slovenščini za sledeči dokument:\n\"\"\"<document>\"\"\"\n\n'''
    summarize_article(
        articles,
        max_tokens=512,
        summary_field_name='summary_d3_512',
        prompt_template=prompt,
        model=oai_const.DEFAULT_MODEL
    )

    brejk = 'point'
