import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx")[["Input", "Response"]].dropna()

df = load_data()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all questions once
@st.cache_resource
def embed_questions(questions):
    return model.encode(questions, convert_to_tensor=True)

question_embeddings = embed_questions(df["Input"].tolist())

# UI config
st.set_page_config("Bulipe Tech FAQ Bot", layout="centered")
st.title("🤖 Bulipe Tech FAQ Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Assalamu Alaikum 🌿! I'm your Bulipe Tech FAQ chatbot. Ask me anything about our digital skills programs."
    }]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Search function
def get_best_response(query):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, question_embeddings)[0]
    best_idx = scores.argmax()
    return df.iloc[best_idx]["Response"]

# Chat input
if user_input := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer = get_best_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
