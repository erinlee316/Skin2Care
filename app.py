import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

FAISS_PATH = "faiss_index"

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.environ.get("GROQ_API_KEY")

st.set_page_config(page_title="Skin2Care", page_icon="🧴", layout="centered")
st.title("🧴 Skin2Care")
st.caption("Ask me anything about skincare products — I'll recommend based on real ingredient data.")

# Load embedding model and FAISS index once
@st.cache_resource
def load_vectorstore():
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.load_local(FAISS_PATH, embed_model, allow_dangerous_deserialization=True)
    return vectorstore

if not os.path.exists(FAISS_PATH):
    st.error("FAISS index not found. Run `python ml.py` first to build the index.")
    st.stop()

with st.spinner("Loading skincare knowledge base..."):
    vectorstore = load_vectorstore()

# Example prompts
st.markdown("**Try asking:**")
cols = st.columns(3)
examples = [
    "Best moisturizer for dry skin",
    "Recommend a gentle cleanser for sensitive skin",
    "What products have hyaluronic acid?",
]
for i, example in enumerate(examples):
    if cols[i].button(example):
        st.session_state["query"] = example

query = st.text_input("Your question:", value=st.session_state.get("query", ""), placeholder="e.g. What's a good sunscreen for oily skin?")

if query:
    with st.spinner("Finding the best products for you..."):
        docs = vectorstore.similarity_search(query, k=5)
        context_string = "\n\n-----\n\n".join(doc.page_content for doc in docs)
        augmented_query = f"<CONTEXT>\n{context_string}\n</CONTEXT>\n\nMY QUESTION:\n{query}"

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a skincare expert. Answer clearly using only the context provided. Only recommend products if they are mentioned in the context."},
                {"role": "user", "content": augmented_query}
            ]
        )
        answer = response.choices[0].message.content

    st.markdown("### Recommendation")
    st.markdown(answer)

    with st.expander("View source products used"):
        for i, doc in enumerate(docs):
            st.markdown(f"**Product {i+1}**")
            st.code(doc.page_content, language="json")
