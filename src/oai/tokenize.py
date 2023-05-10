import logging
import tiktoken

from itertools import islice

logger = logging.getLogger('oai.tokenize')

MODEL_TOKENS = {
    'gpt-4': 8192,
    'gpt-4-32k': 32768,
    'gpt-3.5-turbo': 4096,
    'text-davinci-003': 4097,
    'davinci': 2049,
}

MODEL_ABBREV = {
    'gpt-4': 'g4',
    'gpt-4-32k': 'g4l',
    'gpt-3.5-turbo': 'g35',
    'text-davinci-003': 'd3',
    'davinci': 'd',
}

DEFAULT_ENCODING = 'cl100k_base'
DEFAULT_LENGTH = 2048


def truncate_text_tokens(text, encoding_name=DEFAULT_ENCODING, max_tokens=DEFAULT_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def chunked_tokens(text, encoding_name=DEFAULT_ENCODING, chunk_length=DEFAULT_LENGTH):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator


def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        end = min(i + int(n), len(tokens))
        j = end
        found = False
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                found = True
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if not found:
            j = end
        yield tokens[i:j]
        i = j

