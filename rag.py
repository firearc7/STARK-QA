import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import torch
from torch import Tensor
import kagglehub

# For text processing
import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# For embeddings and retrieval
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# For chunking strategies
from sklearn.cluster import KMeans

# For RAG pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM


@dataclass
class Document:
    """Represents a document or chunk of text with its metadata and embedding."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class TextProcessor:
    """Handles text processing tasks like cleaning, chunking, and embedding."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the text processor.
        
        Args:
            embedding_model_name: Name of the Sentence Transformer model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
        text = re.sub(r'\n+', ' ', text)  # Remove newlines
        return text.strip()
    
    def chunk_by_sentences(self, text: str, chunk_size: int = 5, overlap: int = 2) -> List[str]:
        """
        Chunk text by sentences with overlap.
        
        Args:
            text: Text to chunk
            chunk_size: Number of sentences per chunk
            overlap: Number of sentences to overlap between chunks
            
        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        chunks = []
        
        if chunk_size >= len(sentences):
            return [text]
        
        for i in range(0, len(sentences) - chunk_size + 1, chunk_size - overlap):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        
        # Add the last chunk if it's not already included
        if (len(sentences) - chunk_size) % (chunk_size - overlap) != 0:
            last_chunk = ' '.join(sentences[-(chunk_size):])
            chunks.append(last_chunk)
        
        return chunks

    def chunk_by_semantic_similarity(self, text: str, target_chunks: int = 10) -> List[str]:
        """
        Chunk text based on semantic similarity using clustering.
        
        Args:
            text: Text to chunk
            target_chunks: Target number of chunks to create
            
        Returns:
            List of semantically coherent chunks
        """
        # First get sentence-level chunks
        sentences = sent_tokenize(text)
        
        # Handle case where there are fewer sentences than target chunks
        if len(sentences) <= target_chunks:
            return sentences
        
        # Embed sentences
        embeddings = self.embedding_model.encode(sentences)
        
        # Cluster sentences
        kmeans = KMeans(n_clusters=target_chunks, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group sentences by cluster
        cluster_to_sentences = {i: [] for i in range(target_chunks)}
        for i, cluster_id in enumerate(clusters):
            cluster_to_sentences[cluster_id].append(sentences[i])
        
        # Join sentences within each cluster
        chunks = [' '.join(sents) for sents in cluster_to_sentences.values() if sents]
        
        return chunks
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        return self.embedding_model.encode(texts)


class VectorStore:
    """Manages vector storage and retrieval using FAISS."""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings
        """
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity for normalized vectors)
        self.documents: List[Document] = []
        self.embedding_dim = embedding_dim
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        embeddings = np.vstack([doc.embedding for doc in documents])
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using the query embedding.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        # Normalize query embedding
        query_embedding_norm = query_embedding.copy()
        faiss.normalize_L2(query_embedding_norm.reshape(1, -1))
        
        # Search
        scores, indices = self.index.search(query_embedding_norm.reshape(1, -1), k)
        
        # Return documents and scores
        results = [(self.documents[idx], score) for idx, score in zip(indices[0], scores[0]) if idx < len(self.documents)]
        
        return results


# ...existing code...

class RAGPipeline:
    """Main RAG pipeline that orchestrates the entire process."""
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-base",
        chunk_size: int = 5,
        chunk_overlap: int = 2,
        chunk_method: str = "sentences",
        reranking_enabled: bool = True
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the LLM to use for generation
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chunk_method: Chunking method ("sentences" or "semantic")
            reranking_enabled: Whether to use query-based reranking
        """
        self.text_processor = TextProcessor(embedding_model_name)
        self.vector_store = VectorStore()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_method = chunk_method
        self.reranking_enabled = reranking_enabled
        
        # Initialize LLM for generation
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        # Change from AutoModelForCausalLM to AutoModelForSeq2SeqLM for T5 models
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
        self.generation_pipeline = pipeline(
            "text2text-generation",  # Change pipeline type for encoder-decoder models
            model=self.model, 
            tokenizer=self.tokenizer,
            max_length=512
        )

    def ingest_data(self, data: Union[pd.DataFrame, List[str]], text_column: Optional[str] = None):
        """
        Ingest data into the RAG system.
        
        Args:
            data: DataFrame or list of texts to ingest
            text_column: Column name containing text (for DataFrame)
        """
        texts = []
        metadata_list = []
        
        # Extract texts and metadata
        if isinstance(data, pd.DataFrame):
            if text_column is None or text_column not in data.columns:
                raise ValueError(f"Text column '{text_column}' not found in DataFrame")
            
            texts = data[text_column].tolist()
            # Create metadata from other columns
            for i, row in data.iterrows():
                metadata = {col: row[col] for col in data.columns if col != text_column}
                # Ensure we have an ID for each text
                if 'id' not in metadata:
                    metadata['id'] = str(i)
                metadata_list.append(metadata)
        else:
            texts = data
            metadata_list = [{'id': str(i)} for i in range(len(texts))]
        
        documents = []
        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            # Clean text
            clean_text = self.text_processor.clean_text(text)
            
            # Chunk text
            if self.chunk_method == "sentences":
                chunks = self.text_processor.chunk_by_sentences(clean_text, 
                                                            self.chunk_size, 
                                                            self.chunk_overlap)
            else:
                chunks = self.text_processor.chunk_by_semantic_similarity(clean_text)
            
            # Create document objects for each chunk
            for j, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = j
                chunk_metadata['original_text_id'] = metadata.get('id', str(i))
                
                # Generate embedding
                embedding = self.text_processor.generate_embeddings([chunk])[0]
                
                # Create document
                doc = Document(
                    id=f"{metadata.get('id', i)}-{j}",
                    text=chunk,
                    metadata=chunk_metadata,
                    embedding=embedding
                )
                documents.append(doc)
        
        # Add documents to vector store
        self.vector_store.add_documents(documents)
        
        return len(documents)
    
    def query(self, query: str, k: int = 5) -> List[Document]:
        """
        Query the RAG system.
        
        Args:
            query: Query text
            k: Number of results to retrieve
            
        Returns:
            List of retrieved documents
        """
        # Generate query embedding
        query_embedding = self.text_processor.generate_embeddings([query])[0]
        
        # Retrieve similar documents
        results = self.vector_store.search(query_embedding, k=k)
        
        if self.reranking_enabled and len(results) > 1:
            # Rerank results based on query relevance
            reranked_results = self._rerank_results(query, results)
            return [doc for doc, _ in reranked_results]
        
        return [doc for doc, _ in results]
    
    def _rerank_results(self, query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Rerank results using a more sophisticated method.
        
        Args:
            query: Original query
            results: Initial retrieval results
            
        Returns:
            Reranked results
        """
        # For cross-encoder reranking, we'd use a cross-encoder model here
        # For demonstration, we'll use a simple query term overlap heuristic
        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]
        
        # Simple query term matching boost
        query_terms = set(query.lower().split())
        
        for i, doc in enumerate(documents):
            # Count query terms in document
            doc_terms = set(doc.text.lower().split())
            term_overlap = len(query_terms.intersection(doc_terms)) / max(1, len(query_terms))
            
            # Boost score with term overlap
            scores[i] = scores[i] * (1 + term_overlap)
        
        # Sort by new scores
        reranked_results = [(doc, score) for doc, score in sorted(
            zip(documents, scores), 
            key=lambda x: x[1],
            reverse=True
        )]
        
        return reranked_results
    
    def generate(self, query: str, k: int = 3, prompt_template: Optional[str] = None) -> str:
        """
        Generate a response using retrieved context.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            prompt_template: Optional template for generation prompt
            
        Returns:
            Generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.query(query, k=k)
        
        # Construct context from retrieved documents
        context = "\n\n".join([doc.text for doc in retrieved_docs])
        
        # Construct prompt
        if prompt_template:
            prompt = prompt_template.format(query=query, context=context)
        else:
            prompt = f"Context information is below.\n\n{context}\n\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query}\nAnswer:"
        
        # Generate response
        response = self.generation_pipeline(prompt)[0]['generated_text']
        
        # Extract just the answer part (after the prompt)
        response = response[len(prompt):].strip()
        
        return response


class StoryRAG(RAGPipeline):
    """Specialized RAG system optimized for short stories."""
    
    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "google/flan-t5-base"
    ):
        """
        Initialize the Story RAG system with parameters optimized for narrative text.
        
        Args:
            embedding_model_name: Name of the embedding model
            llm_model_name: Name of the LLM model
        """
        super().__init__(
            embedding_model_name=embedding_model_name,
            llm_model_name=llm_model_name,
            chunk_size=3,  # Smaller chunks for narrative coherence
            chunk_overlap=1,  # Some overlap to maintain context
            chunk_method="sentences",  # Sentence-based chunking for stories
            reranking_enabled=True  # Enable reranking for better results
        )
    
    def analyze_story_themes(self, story_id: str) -> Dict[str, float]:
        """
        Analyze themes in a story using retrieved chunks.
        
        Args:
            story_id: ID of the story to analyze
            
        Returns:
            Dictionary of themes and their relevance scores
        """
        # Find all chunks for this story
        story_docs = [doc for doc in self.vector_store.documents if doc.metadata.get("original_text_id") == story_id]
        
        if not story_docs:
            return {}
        
        # Common literary themes to analyze
        themes = [
            "love", "adventure", "tragedy", "comedy", "mystery",
            "family", "friendship", "betrayal", "growth", "conflict"
        ]
        
        # Generate embeddings for themes
        theme_embeddings = self.text_processor.generate_embeddings(themes)
        
        # Calculate relevance scores
        story_embeddings = np.vstack([doc.embedding for doc in story_docs])
        
        # Average the story embeddings
        avg_story_embedding = np.mean(story_embeddings, axis=0)
        
        # Normalize embeddings
        faiss.normalize_L2(avg_story_embedding.reshape(1, -1))
        faiss.normalize_L2(theme_embeddings)
        
        # Calculate similarity scores
        similarities = cosine_similarity(avg_story_embedding.reshape(1, -1), theme_embeddings)[0]
        
        # Create dictionary of theme scores
        theme_scores = {theme: float(score) for theme, score in zip(themes, similarities)}
        
        return theme_scores
    
    def find_similar_stories(self, story_id: str, k: int = 3) -> List[str]:
        """
        Find stories similar to the given story.
        
        Args:
            story_id: ID of the reference story
            k: Number of similar stories to return
            
        Returns:
            List of similar story IDs
        """
        # Find all chunks for this story
        story_docs = [doc for doc in self.vector_store.documents if doc.metadata.get("original_text_id") == story_id]
        
        if not story_docs:
            return []
        
        # Calculate average embedding for the story
        story_embeddings = np.vstack([doc.embedding for doc in story_docs])
        avg_story_embedding = np.mean(story_embeddings, axis=0)
        
        # Search similar documents
        results = self.vector_store.search(avg_story_embedding, k=k*3)  # Retrieve more for diversity
        
        # Get unique story IDs (excluding the original story)
        similar_story_ids = []
        for doc, _ in results:
            doc_story_id = doc.metadata.get("original_text_id")
            if doc_story_id != story_id and doc_story_id not in similar_story_ids:
                similar_story_ids.append(doc_story_id)
                if len(similar_story_ids) >= k:
                    break
        
        return similar_story_ids


# Main execution code
# Main execution code
def main():
    """Main function to demonstrate the RAG system."""
    # Load the dataset
    print("Loading dataset...")
    # Using a sample dataset since kagglehub.load_dataset isn't available
    df = pd.DataFrame({
        "text": [
            "Once upon a time, there lived a friendship between a fox and a rabbit. They helped each other through many challenges.",
            "The loyal dog waited for his owner to return, day after day, at the same train station.",
            "In the kingdom of dreams, adventure awaited those brave enough to close their eyes and believe."
        ],
        "id": ["0", "1", "2"]  # Adding IDs for the stories
    })
    print("Using sample dataset.")
    
    # Initialize RAG system
    print("Initializing Story RAG system...")
    rag = StoryRAG(
        embedding_model_name="all-MiniLM-L6-v2",  # Smaller, faster model
        llm_model_name="google/flan-t5-base"      # Efficient LLM
    )
    
    # Ingest data
    print("Ingesting stories...")
    num_documents = rag.ingest_data(df, text_column="text")
    print(f"Ingested {num_documents} document chunks from {len(df)} stories")
    
    # Demo query
    print("\nDemo query:")
    query = "Find me a story about friendship and loyalty"
    results = rag.query(query, k=3)
    
    print(f"Query: {query}")
    print("\nTop 3 relevant story chunks:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Story ID: {doc.metadata.get('original_text_id')}")
        print(f"Text: {doc.text[:200]}...")
    
    # Generate a response
    print("\nGenerating response...")
    response = rag.generate(query)
    print(f"Generated response: {response}")
    
    # Analyze themes in a story
    story_id = "0"  # First story
    print(f"\nAnalyzing themes in story {story_id}...")
    themes = rag.analyze_story_themes(story_id)
    print("Theme analysis:")
    for theme, score in sorted(themes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {theme}: {score:.4f}")
    
    print("\nRAG system demo complete!")

if __name__ == "__main__":
    main()