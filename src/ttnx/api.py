import os
import json
import requests

from typing import Any, Dict


def call_textonic(json_object: Dict[str, Any]) -> Dict[str, Any]:
    url = 'https://textonic.io/api/public/ml/process'
    api_key = os.environ['TTNX_API_KEY']
    query = json.dumps(json_object)
    result = []
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
        print('\nTextonic request error:', error)
        return result

    try:
        resp_text: Dict[str, Any] = json.loads(resp.text)
        if resp_text['status']['code'] == 'success':
            for item in resp_text['data']:
                result.append(item)
        return result
    except:
        print('\nTextonic parse error:', resp.text)

    return result