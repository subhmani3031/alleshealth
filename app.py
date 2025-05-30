# app.py
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader, JSONLoader
from langchain.llms import HuggingFaceHub
import tempfile
import os

def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name

    if uploaded_file.name.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif uploaded_file.name.endswith(".json"):
        loader = JSONLoader(file_path, jq_schema=".[]", text_content=False)
    else:
        st.error("Unsupported file type.")
        return None, None

    data = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(data, embeddings)
    os.unlink(file_path)
    return vectordb, data

def main():
    st.set_page_config(page_title="Hospital Reimbursement Assistant", layout="wide")
    st.title("Hospital Reimbursement Assistant")

    uploaded_file = st.file_uploader("Upload Tariff Document (CSV or JSON)", type=["csv", "json"])
    process_button = st.button("Process Document")

    if uploaded_file and process_button:
        vectordb, _ = process_uploaded_file(uploaded_file)

        if vectordb is None:
            return

        llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.3, "max_length": 512})
        st.success("Document processed. Ask your reimbursement questions below.")

        user_query = st.text_input("Ask a question about hospital reimbursement:")
        ask_button = st.button("Ask")

        if ask_button and user_query:
            with st.spinner("Fetching relevant info and generating answer..."):
                retriever = vectordb.as_retriever(search_kwargs={"k": 4})
                docs = retriever.get_relevant_documents(user_query)
                context = "\n\n".join([doc.page_content for doc in docs])

                prompt = f"""
                You are an expert hospital reimbursement assistant.
                Use the following context to answer the user question as accurately as possible.
                If you don't know the answer, just say you don't know, don't make it up.

                Context:
                {context}

                Question:
                {user_query}

                Answer:
                """
                response = llm(prompt)
            st.markdown(f"Answer:\n{response}")
    else:
        st.info("Please upload a Tariff document and click 'Process Document' to get started.")

if __name__ == "__main__":
    main()
