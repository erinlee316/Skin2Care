import os # reads .env file for api key
import json # reads JSON file
from langchain_community.embeddings import HuggingFaceEmbeddings  # converts text to vectors
from langchain_community.vectorstores import FAISS  # stores and searches vectors
from langchain_core.documents import Document  # wraps text into a format FAISS understands
from groq import Groq  # llm api for generating recommendations
# openrouter and gemini not free...


# load product data from a json file
def read_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# convert each product dict into a LangChain Document so FAISS can index it
# each product becomes its own document with its full json as the content
def json_to_documents(data):
    documents = []
    for i, item in enumerate(data):
        # turn the product dict into a readable string
        text = json.dumps(item, indent=2)
        doc = Document(page_content=text, metadata={"source": f"product_{i}"})
        documents.append(doc)
    return documents

# generate embeddings for all products and save the FAISS index to disk
# this only needs to run once 
# after running, we can just load the saved index unless new data is scrapped
def index_to_faiss(json_path, save_path="faiss_index"):
    print("Reading skincare data...")
    data = read_json(json_path)
    documents = json_to_documents(data)

    print("Generating embeddings...")
    # all-mpnet-base-v2 converts each product into a 768-dimensional vector
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    print("Creating FAISS index...")
    # FAISS indexes all the vectors so we can search them quickly later
    vectorstore = FAISS.from_documents(documents, embedding)
    vectorstore.save_local(save_path)
    print(f"Saved FAISS index to '{save_path}'")

# function allows us to ask for recommendations from Groq
# given a user query, find the most similar products using RAG
def perform_rag(query, embedding_model, api_key, save_path="faiss_index"):
    # load the saved FAISS index from disk
    vectorstore = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)

    # retrieve top 5 most relevant products based on semantic similarity to the query
    docs = vectorstore.similarity_search(query, k=5)

    # combine the 5 products into one context string to pass to the LLM
    context = "\n\n-----\n\n".join(doc.page_content for doc in docs)

    # build the full prompt 
    # context first, then the user's question
    prompt = f"<CONTEXT>\n{context}\n</CONTEXT>\n\nMY QUESTION:\n{query}"

    # send to Groq (llama 3.3 70b) and get a recommendation based on the context
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            # system prompt tells the llm to only recommend products from the context
            {"role": "system", "content": "You are a skincare expert. Answer clearly using only the context provided. Only recommend products if they are mentioned in the context."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # load api key from .env file
    with open(".env") as f:
        for line in f:
            key, _, val = line.strip().partition("=")
            if key:
                os.environ[key] = val

    api_key = os.environ.get("GROQ_API_KEY")
    json_path = "./scraped_products/all_products.json"
    faiss_path = "faiss_index"

    # build the FAISS index if it doesn't exist yet, otherwise skip
    # skipping saves 30-60 minutes of re-embedding on every run
    if not os.path.exists(faiss_path):
        index_to_faiss(json_path, faiss_path)
    else:
        print("FAISS index already exists, skipping.")

    # load the embedding model and test with a sample query
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    response = perform_rag("Pick one product that works best if my skin is dry", embed_model, api_key)
    print("RAG Response:\n", response)
