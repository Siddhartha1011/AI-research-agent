from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_core.documents import Document

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.doc_map = []

    def build_index(self, documents: list[Document]):
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        print("Embeddings shape:", embeddings.shape)
        if embeddings.size == 0:
            raise ValueError("No embeddings generated. Check your input data.")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.doc_map = documents

    def search(self, query: str, top_k=3) -> list[Document]:
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        return [self.doc_map[i] for i in indices[0]]