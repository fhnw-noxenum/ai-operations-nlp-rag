from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from settings import COLLECTION_NAME, EMBEDDING_MODEL, database_url


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store(pre_delete_collection: bool = False) -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=database_url(),
        pre_delete_collection=pre_delete_collection,
        use_jsonb=True,
    )
