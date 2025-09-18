from logger import logger
import os
from vectordb_and_ingestion import get_db_collection, embed_documents
from loader import load_yaml_config
from langchain_groq import ChatGroq
from prompt import build_prompt_from_config
from dotenv import load_dotenv

# Loading the environment variables
load_dotenv()

# Pulling the Groq API key from .env file
api_key = os.getenv("GROQ_API_KEY")

collection = get_db_collection(collection_name="wiki_pages")
def retrieve_relevant_documents(
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> list[str]:
    """
    Query the ChromaDB database with a string query.

    Args:
        query (str): The search query string
        n_results (int): Number of results to return (default: 5)
        threshold (float): Threshold for the cosine similarity score (default: 0.3)

    Returns:
        dict: Query results containing ids, documents, distances, and metadata
    """
    logger.info(f"Retrieving relevant documents for query: {query}")
    relevant_results = {
        "ids": [],
        "documents": [],
        "distances": [],
    }
    # Embed the query using the same model used for documents
    logger.info("Embedding query...")
    query_embedding = embed_documents([query])[0]  # Get the first (and only) embedding

    logger.info("Querying collection...")
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances"],
    )

    logger.info("Filtering results...")
    keep_item = [False] * len(results["ids"][0])
    for i, distance in enumerate(results["distances"][0]):
        if distance < threshold:
            keep_item[i] = True

    relevant_results = {"ids": [], "documents": [], "distances": []}

    for i, keep in enumerate(keep_item):
        if keep:
            relevant_results["ids"].append(results["ids"][0][i])
            relevant_results["documents"].append(results["documents"][0][i])
            relevant_results["distances"].append(results["distances"][0][i])

    # keeping two parallel lists
    return {
        "documents": relevant_results["documents"],     # list of strings
        "distances": relevant_results["distances"]      # list of floats
    }

def respond_to_query(
    prompt_config: dict,
    query: str,
    n_results: int = 5,
    threshold: float = 0.3,
) -> str:
    """
    Respond to a query using the ChromaDB database.
    """

    relevant_files = retrieve_relevant_documents(
        query, n_results=n_results, threshold=threshold
    )
    

    # logger.info("-" * 100)
    # logger.info("Relevant documents: \n")
    # for doc in relevant_files['documents']:
    #     logger.info(doc)
    #     logger.info("-" * 100)
    # logger.info("")

    # logger.info("User's question:")
    # logger.info(query)
    # logger.info("")
    # logger.info("-" * 100)
    # logger.info("")
    input_data = (
        f"Relevant documents:\n\n{relevant_files['documents']}\n\nUser's question:\n\n{query}"
    )

    rag_assistant_prompt = build_prompt_from_config(
        prompt_config, input_data=input_data
    )

    logger.info('Relevant documents and their cosine distances:')
    for i in range(n_results):
        logger.info(f"Cosine distance: {relevant_files['distances'][i]:.3f}" )
        logger.info(f"Article: {relevant_files['documents'][i]}")
        logger.info("")

    #logger.info(f"RAG assistant prompt: {rag_assistant_prompt}")
    #logger.info("")

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=api_key
    )

    response = llm.invoke(rag_assistant_prompt)
    return response.content

if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of this script
    APP_CONFIG_FPATH = os.path.join(BASE_DIR, "config", "config.yaml")
    PROMPT_CONFIG_FPATH = os.path.join(BASE_DIR, "config","prompt_config.yaml")

    app_config = load_yaml_config(APP_CONFIG_FPATH)
    prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

    rag_assistant_prompt = prompt_config["rag_wiki_assistant_prompt"]

    vectordb_params = app_config["vectordb"]

    exit_app = False
    while not exit_app:
        query = input(
            "Enter a question, 'config' to change the parameters, or 'exit' to quit: "
        )
        if query == "exit":
            exit_app = True
            exit()

        elif query == "config":
            threshold = float(input("Enter the retrieval threshold: "))
            n_results = int(input("Enter the Top K value: "))
            vectordb_params = {
                "threshold": threshold,
                "n_results": n_results,
            }
            continue

        response = respond_to_query(
            prompt_config=rag_assistant_prompt,
            query=query,
            **vectordb_params,
        )
        logger.info("-" * 100)
        logger.info("LLM response:")
        logger.info(response)
