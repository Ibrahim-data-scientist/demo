import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# Load Q&A Data
@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx")[["Input", "Response"]].dropna()

df = load_data()

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB (in-memory, Streamlit Cloud safe)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("bulipe_faq")

# Populate the collection (if empty)
if len(collection.get()["documents"]) == 0:
    for idx, row in df.iterrows():
        collection.add(
            documents=[row["Response"]],
            metadatas=[{"source": "faq"}],
            ids=[str(idx)],
            embeddings=[embedder.encode(row["Input"]).tolist()]
        )

# UI config
st.set_page_config("Bulipe Tech FAQ Bot", layout="centered")
st.title("🤖 Bulipe Tech FAQ Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Assalamu Alaikum 🌿! I'm your Bulipe Tech FAQ chatbot. Ask me anything about our digital skills programs."
    }]

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Search function
def get_faq_response(query):
    query_vec = embedder.encode(query).tolist()
    result = collection.query(query_embeddings=[query_vec], n_results=1)
    return result["documents"][0][0] if result["documents"] else "Sorry, I couldn't find an answer for that."

# Chat input
if user_input := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    response = get_faq_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
