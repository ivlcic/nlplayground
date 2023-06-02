#!/usr/bin/env python
import logging
import torch
import transformers

from typing import List
from kl.articles import Articles, Article
from local.constants import MODEL_CACHE_DIR

logger = logging.getLogger('mosaic')

cache_model_dir = MODEL_CACHE_DIR

if __name__ == "__main__":
    requests = Articles()
    # requests.filter_customer('7fd935a6-a1f5-42d1-8b5f-048dd54c07d1')
    # requests.filter_customer('a6f60ee6-f990-4620-8508-c3a6f6cc0dc1')
    requests.filter_customer('a65c7372-9fbe-410c-93d7-4613d26488e7')
    requests.filter_country('SI')
    requests.limit(3)

    # articles: List[Article] = requests.gets('2023-05-10T08:00:00', '2023-05-12T08:00:00')
    # articles: List[Article] = requests.gets('2023-05-11T08:00:00', '2023-05-12T08:00:00')
    articles: List[Article] = requests.gets('2023-05-13T08:00:00', '2023-05-14T08:00:00')

    config = transformers.AutoConfig.from_pretrained(
        'mosaicml/mpt-7b-chat',
        trust_remote_code=True,
        cache_dir=cache_model_dir
    )
    #config.attn_config['attn_impl'] = 'triton'
    logger.info("Loading model")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        'mosaicml/mpt-7b-chat',
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=cache_model_dir
    )
    model.to(device='cuda:0')
    # model.to(device='cpu')
    logger.info("Model loaded")

