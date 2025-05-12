# Environment fixes
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Imports
from flask import Flask, request, render_template, jsonify
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Load the Q&A data
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

# Response generation
def get_faq_response(input_text):
    query_embedding = embedder.encode(input_text).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    return results["documents"][0][0] if results["documents"] else "Sorry, I couldn't find an answer to that."

# Routes
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input")
    if user_input:
        response = get_faq_response(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "Please provide a question."})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)