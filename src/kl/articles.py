import os
import requests
import json
import uuid

from mergedeep import merge
from uuid import uuid3
from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta
from typing import Any, List, TypeVar, Dict

TArticle = TypeVar("TArticle", bound="Article")
TArticles = TypeVar("TArticles", bound="Articles")


class Article:

    def __init(self):
        if 'uuid' in self.data:
            self.uuid = self.data['uuid']
        if 'created' in self.data:
            self.created = datetime.fromisoformat(self.data['created'].replace('Z', '+00:00'))
        if 'published' in self.data:
            self.published = datetime.fromisoformat(self.data['published'].replace('Z', '+00:00'))
        if 'language' in self.data:
            self.language = self.data['language']
            if 'translations' in self.data and self.language in self.data['translations']:
                self.title = self.data['translations'][self.language]['title']
                self.body = self.data['translations'][self.language]['body']

        if 'country' in self.data:
            self.country = self.data['country']['name']

        if 'mediaReach' in self.data:
            self.mediaReach = self.data['mediaReach']
        if 'advertValue' in self.data:
            self.advertValue = self.data['advertValue']

        if 'media' in self.data:
            self.media = self.data['media']['name']
            if 'tags' in self.data['media']:
                for tag in self.data['media']['tags']:
                    if 'org.dropchop.jop.beans.tags.MediaType' == tag['class']:
                        self.media_type = tag

        if 'tags' in self.data:
            for tag in self.data['tags']:
                if 'org.dropchop.jop.beans.tags.CustomerTopicGroup' == tag['class']:
                    self.customers.append(tag)
                if 'org.dropchop.jop.beans.tags.CustomerTopic' == tag['class']:
                    self.topics.append(tag)

    def __init__(self, json_object: Dict[str, Any]):
        self.data: Dict[str, Any] = json_object
        self.uuid: str = ''
        self.language: str = ''
        self.title: str = ''
        self.body: str = ''
        self.media: str = ''
        self.mediaReach: int = 0
        self.advertValue: float = 0.0
        self.media_type: Dict[str, Any] = {}
        self.customers: List = []
        self.topics: List = []
        self.country: str = ''
        self.created: datetime = datetime(1999, 1, 1, 0, 0, 0, 0)
        self.published: datetime = datetime(1999, 1, 1, 0, 0, 0, 0)
        self.__init()

    def __str__(self) -> str:
        return '[' + self.uuid + '][' + self.created.astimezone().isoformat(timespec='seconds') \
            + '][' + self.country + '][' + self.media + '][' + self.title + ']'

    def to_cache(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        with open(os.path.join('data', self.uuid + '.json'), 'w') as fp:
            json.dump(self.data, fp)

    def from_cache(self, data_path) -> bool:
        if not os.path.exists(data_path):
            return False
        fname = os.path.join('data', self.uuid + '.json')
        if not os.path.exists(fname):
            return False
        with open(fname) as file:
            json_object = json.loads(file.read())
            merge(json_object, self.data)
            self.data = json_object
            self.__init()
            return True


class Articles:

    def __init__(self, url: str = None, user: str = None, passwd: str = None):
        self.user: str = user if user else os.environ['CPTM_SUSER']
        self.passwd: str = passwd if passwd else os.environ['CPTM_SPASS']
        self.url: str = url if url else os.environ['CPTM_SURL']
        self.filters = []
        self.limit = 100
        self.offset = 0
        self.filter = {
            'topics': [],
            'customers': [],
            'media_tags': [],
            'country': '',
            'language': '',
        }
        self.query_tpl: str = '''
        {
          "query": {
            "bool": {
              "filter": [
                {
                  "range": {
                    "created": {
                      "gte": "<date_start>",
                      "lt": "<date_end>"
                    }
                  }
                }
                <filters>
              ]
            }
          },
          "_source": [
            "uuid",
            "created",
            "published",
            "tags",
            "media",
            "mediaReach",
            "advertValue",
            "media.tags",
            "country",
            "language",
            "translations"
          ],
          "from": <from>,
          "size": <size>
        }
        '''

    def _inject_filters(self, query: str) -> str:
        tags = self.filter['topics'] + self.filter['customers']
        if tags:
            self.filters.append(
                '''
                {
                  "terms": {
                    "tags.uuid": ''' + json.dumps(tags) + '''
                  }
                }
                '''
            )
        if self.filter['country']:
            self.filters.append(
                '''
                {
                  "term": {
                    "country.name": "''' + self.filter['country'] + '''"
                  }
                }
                '''
            )
        if self.filter['language']:
            self.filters.append(
                '''
                {
                  "term": {
                    "language": "''' + self.filter['language'] + '''"
                  }
                }
                '''
            )
        if self.filter['media_tags']:
            self.filters.append(
                '''
                {
                  "terms": {
                    "media.tags.uuid": ''' + json.dumps(self.filter['media_tags']) + '''
                  }
                }
                '''
            )
        filters = ''
        if self.filters:
            filters = ',' + ','.join(self.filters)
        return query.replace('<filters>', filters)

    def filter_customer(self, customer_uuid: str) -> TArticles:
        if not customer_uuid:
            self.filter['customers'] = []
            return self
        self.filter['customers'].append(str(uuid3(uuid.NAMESPACE_URL, 'CustomerTopicGroup.' + customer_uuid)))
        return self

    def filter_topic(self, topic_uuid: str) -> TArticles:
        if not topic_uuid:
            self.filter['topics'] = []
            return self

        self.filter['topics'].append(topic_uuid)
        return self

    def filter_country(self, code: str) -> TArticles:
        self.filter['country'] = code
        return self

    def filter_media_type(self, tag: str) -> TArticles:
        if not tag:
            self.filter['media_tags'] = []
            return self

        self.filter['media_tags'].append(tag)
        return self

    def limit(self, limit: int) -> TArticles:
        self.limit = limit
        return self

    def offset(self, offset: int) -> TArticles:
        self.offset = offset
        return self

    def get(self, start: datetime, end: datetime = None) -> List[Article]:
        if not end:
            end = start + timedelta(hours=24)
        query = self.query_tpl.replace('<from>', str(self.offset))
        query = query.replace('<size>', str(self.limit))
        query = query.replace('<date_start>', start.astimezone().isoformat())
        query = query.replace('<date_end>', end.astimezone().isoformat())
        query = self._inject_filters(query)
        # print(query)

        result = []
        try:
            # make HTTP verb parameter case-insensitive by converting to lower()
            resp = requests.post(self.url,
                                 headers={'Content-Type': 'application/json'},
                                 auth=HTTPBasicAuth(self.user, self.passwd),
                                 data=query)
        except Exception as error:
            print('\nElasticsearch request error:', error)
            return result

        try:
            resp_text: Dict[str, Any] = json.loads(resp.text)
            for hit in resp_text['hits']['hits']:
                result.append(Article(hit['_source']))
        except:
            print('\nElasticsearch parse error:', resp.text)

        return result

    def gets(self, start: str, end: str = None) -> List[Article]:
        start_date = datetime.fromisoformat(start)
        end_date = None
        if end:
            end_date = datetime.fromisoformat(end)

        return self.get(start_date, end_date)
