import os
import chromadb
import torch
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from loader import load_all_text_files
from logger import logger

# Setting up the directory paths for important folders
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
VECTORDB_DIR = os.path.join(OUTPUTS_DIR, "vector_db")


# Select the device: CUDA (GPU) if available, otherwise Apple MPS (Mac GPU), otherwise fall back on CPU
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)



def initialize_db(
    persist_directory: str = VECTORDB_DIR,
    collection_name: str = "wiki_pages",
    delete_existing: bool = False,
) -> chromadb.Collection:
    """
    Initialize a ChromaDB instance and persist it to disk.

    Args:
        persist_directory (str): The directory where ChromaDB will persist data. Defaults to "./vector_db"
        collection_name (str): The name of the collection to create/get. Defaults to "wiki_pages"
        delete_existing (bool): Whether to delete the existing database if it exists. Defaults to False
    Returns:
        chromadb.Collection: The ChromaDB collection instance
    """
    if os.path.exists(persist_directory) and delete_existing:
        shutil.rmtree(persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    # Initialize ChromaDB client with persistent storage
    client = chromadb.PersistentClient(path=persist_directory)

    # Create or get a collection
    try:
        # Try to get existing collection first
        collection = client.get_collection(name=collection_name)
        print(f"Retrieved existing collection: {collection_name}")
    except Exception:
        # If collection doesn't exist, create it
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:batch_size": 10000,
            },  # Use cosine distance for semantic search
        )
        print(f"Created new collection: {collection_name}")

    print(f"ChromaDB initialized with persistent storage at: {persist_directory}")

    return collection


def get_db_collection(
    persist_directory: str = VECTORDB_DIR,
    collection_name: str = "wiki_pages",
) -> chromadb.Collection:
    """
    Get a ChromaDB client instance.

    Args:
        persist_directory (str): The directory where ChromaDB persists data
        collection_name (str): The name of the collection to get

    Returns:
        chromadb.PersistentClient: The ChromaDB client instance
    """
    return chromadb.PersistentClient(path=persist_directory).get_collection(
        name=collection_name
    )

def chunk_pages(
    pages: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """
    Chunk the wikipedia pages into smaller documents.
    """
    # The splittler model tries to split text at natural boundaries (paragraphs, sentences) 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(pages)


def embed_documents(documents: list[str]) -> list[list[float]]:
    """
    Converts a list of text chunks into embeddings (vectors) using the pre-loaded model.

    Args:
        documents (List[str]): List of text chunks to embed.

    Returns:
        List[List[float]]: Corresponding embeddings for each text chunk.
    """
    # Guard clause: if the input is empty, return an empty list
    if not documents:
        return []
    # Initialize the embedding model
    # This model converts text into numerical vectors (embeddings) suitable for semantic search

    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
    )
    # Use the initialized embedding model to compute embeddings
    # Each text chunk becomes a numerical vector that can be stored in a vector database
    embeddings = embedding_model.embed_documents(documents)

    # Return the list of embeddings
    return embeddings

def insert_pages(collection: chromadb.Collection, pages: list[str]):
    """
    Insert the wikipedia documents into a ChromaDB collection.

    Args:
        collection (chromadb.Collection): The collection to insert documents into
        documents (list[str]): The documents to insert

    Returns:
        None
    """
    next_id = collection.count()

    for page in pages:
        # Chunking each wikipedia page
        chunked_pages = chunk_pages(page)
        # Embedding the chunked pages into vectors
        embeddings = embed_documents(chunked_pages)
        # Assigning Ids
        ids = list(range(next_id, next_id + len(chunked_pages)))
        ids = [f"document_{id}" for id in ids]
        # Adding to the ChromaDB collection
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunked_pages,
        )
        next_id += len(chunked_pages)


def main():
    logger.info(f"Initializing the Vector database in {VECTORDB_DIR} directory")
    collection = initialize_db(
        persist_directory=VECTORDB_DIR,
        collection_name="wiki_pages",
        delete_existing=True,
    )
    logger.info(f"Loading the wikipedia pages from the data folder")
    all_pages = load_all_text_files()
    logger.info(f"Inserting chunked and embedded wikipedia pages into the initialized ChromaDB collection")
    insert_pages(collection, all_pages)

    print(f"Total documents in collection: {collection.count()}")


if __name__ == "__main__":
    main()

