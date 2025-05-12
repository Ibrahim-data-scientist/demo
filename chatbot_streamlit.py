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

# Store data if not already stored
if len(collection.get()["documents"]) == 0:
    for idx, row in df.iterrows():
        collection.add(
            documents=[row["Response"]],
            metadatas=[{"source": "faq"}],
            ids=[str(idx)],
            embeddings=[embedder.encode(row["Input"]).tolist()]
        )

# Streamlit UI
st.set_page_config(page_title="Bulipe Tech FAQ Bot", page_icon="🤖")
st.title("🤖 Bulipe Tech Service Chatbot")
st.write("Ask a question about our company services!")

user_input = st.text_input("You:", placeholder="What services do you offer?")

if user_input:
    query_embedding = embedder.encode(user_input).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    answer = results["documents"][0][0] if results["documents"] else "Sorry, I couldn't find an answer to that."
    st.markdown(f"**Bot:** {answer}")
