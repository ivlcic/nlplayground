import logging
import logging.config


class KLFilter(logging.Filter):
    def filter(self, record):
        record.levelname = '[%s]' % record.levelname
        record.funcName = '[%s]' % record.funcName
        record.lineno = '[%s]' % record.lineno
        return True


#logging.basicConfig(
#    format='%(asctime)s %(levelname)-7s %(name)s %(lineno)-3s: %(message)s',
#    level=logging.DEBUG,
#    datefmt='%Y-%m-%d %H:%M:%S'
#)

LOGGING = {
    'version': 1,
    'formatters': {
        'my_formatter': {
          'format': '%(asctime)s %(levelname)-7s %(name)s %(lineno)-3s: %(message)s',
          'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'filters': {
        'myfilter': {
            '()': KLFilter
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'filters': ['myfilter'],
            'formatter': 'my_formatter',
        }
    },
    'root': {
        'level': 'DEBUG',
        'filters': ['myfilter'],
        'handlers': ['console']
    },
}

logging.config.dictConfig(LOGGING)
#logging.getLogger().addFilter(fmt_filter)