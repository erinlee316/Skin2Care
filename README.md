# Skin2Care

An AI-powered skincare product recommendation system built with Retrieval-Augmented Generation (RAG).

**Live App:** [skin2care.streamlit.app](https://skin2care.streamlit.app)

## What It Does

Skin2Care lets users ask natural language questions about skincare and receive personalized product recommendations grounded in real ingredient data from over 11,000 products scraped from Incidecoder.

**Example queries:**
- "Best moisturizer for dry skin"
- "Recommend a gentle cleanser for sensitive skin"
- "What products have hyaluronic acid?"

## How It Works

1. **Data Collection** — Scraped 11,000+ skincare products from Incidecoder using Selenium, capturing product names, brands, ingredients, and descriptions
2. **Embedding** — Product data is converted into vector embeddings using `sentence-transformers/all-mpnet-base-v2`
3. **Vector Search** — Embeddings are indexed with FAISS for fast semantic similarity search
4. **RAG Pipeline** — User queries retrieve the top 5 most relevant products, which are passed as context to an LLM (Llama 3.3 70B via Groq) to generate a recommendation
5. **Frontend** — Streamlit app for interactive querying

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **FAISS** — Vector similarity search
- **Sentence Transformers** — Text embeddings
- **Groq** — LLM inference (Llama 3.3 70B)
- **Streamlit** — Web app frontend
- **Selenium + BeautifulSoup** — Web scraping

## Run Locally

```bash
git clone https://github.com/erinlee316/Skin2Care.git
cd Skin2Care
pip install -r requirements.txt
```

Add a `.env` file with your Groq API key:
```
GROQ_API_KEY=your-key-here
```

Build the FAISS index (one-time):
```bash
python ml.py
```

Run the app:
```bash
streamlit run app.py
```
