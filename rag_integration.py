# rag_integration.py
from typing import List, Dict, Any
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

class FSAgentRAG:
    """
    RAG entegrasyonu: FS Agent sonuçları + PubMed sonuçlarını 
    vektör veritabanına indexleyip doğal dil sorgusu yapılmasını sağlar.
    """
    def __init__(self, persist_dir: str = "fsagent_rag"):
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings
        )
        self.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")),
            retriever=self.vectordb.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff"
        )

    def index_documents(self, documents: List[Dict[str, Any]]):
        """
        Belgeleri RAG vektör veritabanına ekler.
        documents: [{"text": "...", "metadata": {...}}, ...]
        """
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        self.vectordb.add_texts(texts, metadatas=metadatas)
        self.vectordb.persist()

    def query(self, question: str) -> str:
        """
        Doğal dil sorusu için RAG cevabı döndürür.
        """
        return self.qa.run(question)
