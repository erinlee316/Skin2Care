import os
import json
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from groq import Groq

# ------------------- Load JSON -------------------
def read_json(filepath: str) -> List[Dict]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return []

# ------------------- Prepare Docs -------------------
def json_to_documents(json_data: List[Dict]) -> List[Document]:
    documents = []
    for i, item in enumerate(json_data):
        text = json.dumps(item, indent=2)
        doc = Document(page_content=text, metadata={"source": f"product_{i}"})
        documents.append(doc)
    return documents

# ------------------- Index to FAISS -------------------
def index_to_faiss(json_path: str, save_path: str = "faiss_index"):
    print("Reading skincare data...")
    data = read_json(json_path)
    documents = json_to_documents(data)

    print("Generating embeddings...")
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local(save_path)

    print(f"Saved FAISS index to '{save_path}'")

# ------------------- RAG Query -------------------
def perform_rag(query: str, embedding_model, userdata: Dict, save_path: str = "faiss_index", model="llama-3.3-70b-versatile"):
    print("Loading FAISS index...")
    vectorstore = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(query, k=5)

    context_string = "\n\n-----\n\n".join(doc.page_content for doc in docs)
    augmented_query = f"<CONTEXT>\n{context_string}\n</CONTEXT>\n\nMY QUESTION:\n{query}"

    system_prompt = """You are a skincare expert. Answer clearly using only the context provided.
Only recommend products if they are mentioned in the context."""

    client = Groq(api_key=userdata["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )
    return response.choices[0].message.content

# ------------------- Example Usage -------------------
if __name__ == "__main__":
    with open(".env") as f:
        for line in f:
            key, _, val = line.strip().partition("=")
            if key:
                os.environ[key] = val
    api_key = os.environ.get("GROQ_API_KEY")

    userdata = {
        "GROQ_API_KEY": api_key
    }

    json_path = "./scraped_products/all_products.json"
    faiss_path = "faiss_index"

    # 1. Index the JSON file to FAISS (skip if already built)
    if not os.path.exists(faiss_path):
        index_to_faiss(json_path, faiss_path)
    else:
        print("FAISS index already exists, skipping indexing.")

    # 2. Ask a question
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    response = perform_rag("Pick one product that works best if my skin is dry", embed_model, userdata)
    print("RAG Response:\n", response)
