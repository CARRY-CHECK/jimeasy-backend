"""Create the same configured embedding function for ingestion and retrieval."""
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

from app.settings import MODEL_CONFIG


def create_embedding_function():
    config = MODEL_CONFIG['embedding']
    provider = config['provider']
    if provider == 'chroma_default':
        # Keep the persisted Chroma function identity compatible with existing DBs.
        if config['model'] != 'all-MiniLM-L6-v2' or config['parameters']:
            raise ValueError('chroma_default supports only all-MiniLM-L6-v2 with empty parameters.')
        return DefaultEmbeddingFunction()
    if provider == 'sentence_transformer':
        return SentenceTransformerEmbeddingFunction(
            model_name=config['model'], **config['parameters'],
        )
    raise ValueError(f'Unsupported embedding provider: {provider}')
