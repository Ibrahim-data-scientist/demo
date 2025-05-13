# Environment fixes FIRST
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Now other imports
import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load the Q&A data
@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx")[["Input", "Response"]].dropna()

df = load_data()

# Setup embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Setup ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("faq_bot")

# Store data if not already 
if len(collection.get()["documents"]) == 0:
    for idx, row in df.iterrows():
        collection.add(
            documents=[row["Response"]],
            metadatas=[{"source": "faq"}],
            ids=[str(idx)],
            embeddings=[embedder.encode(row["Input"]).tolist()]
        )

# --- Page Configuration ---
st.set_page_config(page_title="Bulipe Tech FAQ Bot", page_icon="🤖", layout="centered")

# --- Initialize Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Assalamu alaikum 🍁, I'm your Bulipe Tech FAQ chatbot, ready to assist you! 😊"
    }]

# Sidebar for clearing chat history
with st.sidebar:
    st.title('Chat Settings')

    def clear_chat_history():
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Assalamu alaikum 🍁, I'm your Bulipe Tech FAQ chatbot, ready to assist you! 😊"
        }]
    
    st.button('Clear Chat History', on_click=clear_chat_history)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- Response Generation ---
def get_faq_response(input_text):
    query_embedding = embedder.encode(input_text).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    return results["documents"][0][0] if results["documents"] else "Sorry, I couldn't find an answer to that."

# Chat input handling
if prompt := st.chat_input("Ask me about our services..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Fetching answer..."):
                response = get_faq_response(prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
