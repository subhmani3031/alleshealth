An AI-powered assistant to help hospital staff with reimbursement-related questions. This chatbot uses LangChain's RAG (Retrieval-Augmented Generation) system with FAISS vector store and supports CSV, JSON Tariff documents.

# Features

- Upload structured/semi-structured Tariff documents (CSV, Excel, PDF)
- Asks natural language questions about medical reimbursements
- Uses FAISS + HuggingFace embeddings to retrieve top-k relevant chunks
- Uses a LLM (Flan-T5 via HuggingFaceHub) to answer based on context

# How It Works

1. Upload a tariff document.
2. Click "Process Document".
3. Ask your question in natural language.
4. The assistant retrieves relevant context and generates a grounded answer.

# Setup

# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run streamlit_app.py