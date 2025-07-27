from os import environ

SENTENCE_TRANSFORMER = environ.get(
    'SENTENCE_TRANSFORMER', 'paraphrase-multilingual-MiniLM-L12-v2')

OLLAMA_MODEL = environ.get('OLLAMA_MODEL', 'gemma3:12b')

QUERY_POINTS_LIMIT = environ.get('QUERY_POINTS_LIMIT', 1)

QDRANT_URL = environ.get('QDRANT_URL', 'localhost')
