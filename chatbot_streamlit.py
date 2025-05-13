import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64


def set_custom_style(background_image_path):
    with open(background_image_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
# Load data
@st.cache_data
def load_data():
    return pd.read_excel("data.xlsx")[["Input", "Response"]].dropna()

df = load_data()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Embed all questions once
@st.cache_resource
def embed_questions(questions):
    return model.encode(questions, convert_to_tensor=True)

question_embeddings = embed_questions(df["Input"].tolist())

st.set_page_config(
    page_title="Bulipe chatbot",
    page_icon="r.png",
    layout="wide" 
)
st.title("🤖 Bulipe Tech Services based Chatbot")
set_custom_style("B.jpg")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Assalamu Alaikum 🌿! I'm your Bulipe Tech  chatbot. Ask me anything about our digital skills programs."
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
