import sys
import os
import streamlit as st

# add the code folder to the path so we can import from it
BASE_DIR = os.path.dirname(__file__)            # app/
CODE_DIR = os.path.join(BASE_DIR, '..', 'code') # ../code
sys.path.append(CODE_DIR)

from loader import load_yaml_config
from retrieval_and_response import respond_to_query, retrieve_relevant_documents  

# Load configs from code/config
APP_CONFIG_FPATH = os.path.join(CODE_DIR, 'config', 'config.yaml')
PROMPT_CONFIG_FPATH = os.path.join(CODE_DIR, 'config', 'prompt_config.yaml')

app_config = load_yaml_config(APP_CONFIG_FPATH)
prompt_config = load_yaml_config(PROMPT_CONFIG_FPATH)

rag_assistant_prompt = prompt_config["rag_wiki_assistant_prompt"]

# Sidebar: retrieval settings
st.sidebar.header("RAG Settings")
n_results = st.sidebar.number_input(
    "Number of results (Top K)",
    min_value=1, max_value=50,
    value=app_config["vectordb"].get("n_results", 5)
)
threshold = st.sidebar.slider(
    "Retrieval Threshold (cosine distance)",
    0.0, 1.0,
    app_config["vectordb"].get("threshold", 0.5)
)

# --- Navigation bar ---
page = st.sidebar.radio(
    "Go to page:",
    ("Ask & Answer", "Relevant Documents", "RAG Prompt")
)

# --- Page 1: Ask & Answer ---
if page == "Ask & Answer":
    st.title("📚 RAG Wiki Assistant")
    st.write("Ask any question below. Adjust the retrieval settings from the sidebar.")

    query = st.text_input("Enter your question:")

    if st.button("Ask") and query.strip():
        with st.spinner("Retrieving answer..."):
            # Get response from the LLM
            response = respond_to_query(
                prompt_config=rag_assistant_prompt,
                query=query,
                threshold=threshold,
                n_results=n_results,
            )
            st.session_state["last_query"] = query
            st.session_state["llm_response"] = response

            # Also store retrieved documents for page 2
            docs_result = retrieve_relevant_documents(query=query, threshold=threshold, n_results=n_results)
            st.session_state["retrieved_docs"] = docs_result

    if "llm_response" in st.session_state:
        st.success("Response:")
        st.write(st.session_state["llm_response"])

# --- Page 2: Relevant Documents ---
elif page == "Relevant Documents":
    st.title("📄 Relevant Documents & Cosine Distances")
    st.write(f"Relevant documents for the given query")
    if "retrieved_docs" not in st.session_state:
        st.info("No documents yet. Go to 'Ask & Answer' first.")
    else:
        docs_result = st.session_state["retrieved_docs"]
        docs = docs_result["documents"]
        dists = docs_result["distances"]

        # Sort by distance descending
        combined = sorted(zip(docs, dists), key=lambda x: x[1])

        for idx, (doc, dist) in enumerate(combined, 1):
            st.markdown(f"**{idx}. Cosine distance: {dist:.f}**")
            st.write(doc)
            st.markdown("---")

# --- Page 3: RAG Prompt ---
elif page == "RAG Prompt":
    st.title("📝 Current RAG System Prompt ")
    st.warning(
    "⚠️ **Reminder:** This section is displayed only to demonstrate how the RAG-generated "
    "answer behaves for the given query. In a production environment, exposing the system "
    "prompt would pose a security and integrity risk and should be strictly avoided."
)
    st.markdown(rag_assistant_prompt)

