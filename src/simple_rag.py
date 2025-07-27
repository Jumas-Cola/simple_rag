import os
from itertools import count

import click
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import chat
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

import config
from vector_db import client

cli = click.Group()

model = SentenceTransformer(config.SENTENCE_TRANSFORMER)


@cli.command(help='Проиндексировать obsidian vault.')
@click.option('--path', help='Путь к obsidian vault.', required=True)
def index(path: str):
    path = path.rstrip('/')

    if not os.path.isdir(path):
        raise ValueError(f'Путь {path} не является директорией')

    collection_name = os.path.basename(path)

    if client.collection_exists(collection_name=collection_name):
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=rest.Distance.COSINE,
        ),
    )

    id_generator = count()

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    book_text = f.read()
                    chunks = splitter.split_text(book_text)
                    for chunk in chunks:
                        vector = model.encode(chunk)
                        client.upsert(
                            collection_name=collection_name,
                            points=[
                                {
                                    'id': next(id_generator),
                                    'vector': vector.tolist(),
                                    'payload': {
                                        'file': file,
                                        'text': chunk
                                    }
                                }
                            ]
                        )


@cli.command(help='Отправить поисковый запрос.')
@click.option('--collection', help='Имя коллекции.', required=True)
@click.option('--query', help='Поисковый запрос.', required=True)
def search(collection: str, query: str):

    results = client.query_points(
        collection_name=collection,
        query=model.encode(query).tolist(),
        limit=config.QUERY_POINTS_LIMIT
    )

    prompt = """
    Ответь на вопрос, используя ТОЛЬКО данные ниже.
    В ответе укажи ссылки на источники из поля "Контекст".
    Если информации нет, скажи «Не нашел данных».

    Контекст:
    """

    for point in results.points:
        chunk_text = point.payload['text']
        file = point.payload['file']
        prompt += f'{chunk_text} [Источник: {file}]\n'

    stream = chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {
                'role': 'system',
                'content': prompt,
            },
            {
                'role': 'user',
                'content': query
            }
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)


if __name__ == '__main__':
    cli()
