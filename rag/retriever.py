from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

class FinQARetriever:
    def __init__(self):
        """
        Initializes the dense embeddings model (all-MiniLM-L6-v2) for FAISS vector search.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def example_to_documents(self, example):
        """
        Converts the pre_text, post_text, and table of a FinQA example
        into a list of formatted LangChain Document objects.
        """
        documents = []
        
        # Add pre_text sentences
        for i, text in enumerate(example.get("pre_text", [])):
            if text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={"source": "pre_text", "index": i}
                ))
                
        # Add post_text sentences
        for i, text in enumerate(example.get("post_text", [])):
            if text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={"source": "post_text", "index": i}
                ))
                
        # Add table rows with Header-Aware Chunking (Option A)
        table = example.get("table", [])
        if len(table) > 0:
            headers = table[0]
            headers_str = " | ".join(headers)
            # Prepend headers to all data rows (starting from index 1 if available)
            start_idx = 1 if len(table) > 1 else 0
            for i in range(start_idx, len(table)):
                row = table[i]
                row_str = " | ".join(row)
                if row_str.strip():
                    page_content = f"Columns: {headers_str} -> Row: {row_str}"
                    documents.append(Document(
                        page_content=page_content,
                        metadata={"source": "table", "index": i}
                    ))
        return documents

    def build_ensemble_retriever(self, documents, k=5):
        """
        Builds and returns a LangChain EnsembleRetriever (FAISS + BM25) 
        from a list of Documents.
        """
        if not documents:
            return None
            
        # Build FAISS vector store
        db = FAISS.from_documents(documents, self.embeddings)
        dense_retriever = db.as_retriever(search_kwargs={"k": k})
        
        # Build BM25 retriever
        sparse_retriever = BM25Retriever.from_documents(documents)
        sparse_retriever.k = k
        
        # Combine using EnsembleRetriever
        return EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )

    def retrieve_context(self, example, query, k=5):
        """
        Builds a transient hybrid retriever and returns the top-k deduplicated documents.
        """
        documents = self.example_to_documents(example)
        ensemble_retriever = self.build_ensemble_retriever(documents, k=k)
        
        if not ensemble_retriever:
            return []
            
        retrieved_docs = ensemble_retriever.invoke(query)
        
        # Deduplicate retrieved documents to maintain order and uniqueness
        seen = set()
        unique_docs = []
        for doc in retrieved_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)
                
        return unique_docs[:k]

