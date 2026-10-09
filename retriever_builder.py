from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

class RetrieverBuilder:
    def __init__(self):
        # Uses standard OpenAI embeddings (can be swapped with Hugging Face sentence-transformers)
        self.embeddings = OpenAIEmbeddings()

    def build_hybrid_retriever(self, docs):
        """Builds a hybrid retriever combining BM25 keyword search and ChromaDB vector search."""
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        return hybrid_retriever
