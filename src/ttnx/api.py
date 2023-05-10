import logging
import os
import json
import requests

from typing import Any, Dict, List

logger = logging.getLogger('ttnx.api')


def call_textonic(json_object: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = 'https://textonic.io/api/public/ml/process'
    api_key = os.environ['TTNX_API_KEY']
    query = json.dumps(json_object)
    result = []
    logger.debug('Invoking Textonic [%s]...', url)
    try:
        # make HTTP verb parameter case-insensitive by converting to lower()
        resp = requests.post(url,
                             headers={
                                 'Content-Type': 'application/json',
                                 'Accept': 'application/vnd.dropchop.result+json',
                                 'Accept-Encoding': 'gzip,deflate',
                                 'Authorization': 'Bearer ' + api_key
                             },
                             data=query)
    except Exception as error:
        logger.error('Textonic request [%s] error [%s]:', query, error)
        return result
    logger.info('Invoked Textonic [%s]', url)
    try:
        resp_text: Dict[str, Any] = json.loads(resp.text)
        if resp_text['status']['code'] == 'success':
            for item in resp_text['data']:
                result.append(item)
        return result
    except:
        logger.error('Textonic parse error [%s]:', resp.text)
    logger.info('Parsed Textonic [%s] articles.', len(result))
    return result
