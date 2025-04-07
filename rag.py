import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import torch
from torch import Tensor

# For text processing
import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)  # Added for improved text processing
from nltk.corpus import stopwords

# For embeddings and retrieval
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# For chunking strategies
from sklearn.cluster import KMeans

# For RAG pipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Enhanced cleaning
        text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
        text = re.sub(r'\n+', ' ', text)  # Remove newlines
        text = re.sub(r'[^\w\s.,?!-]', '', text)  # Remove special characters except punctuation
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
        if not text:
            return []
            
        sentences = sent_tokenize(text)
        chunks = []
        
        if len(sentences) <= chunk_size:
            return [text]
        
        for i in range(0, len(sentences) - chunk_size + 1, max(1, chunk_size - overlap)):
            chunk = ' '.join(sentences[i:i + chunk_size])
            chunks.append(chunk)
        
        # Add the last chunk if it's not already included
        if chunks and sentences and i + chunk_size < len(sentences):
            last_chunk = ' '.join(sentences[-(chunk_size):])
            # Check if the last chunk is sufficiently different from the previous one
            if not chunks or not self._is_significant_overlap(chunks[-1], last_chunk):
                chunks.append(last_chunk)
        
        return chunks

    def _is_significant_overlap(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two text chunks have significant overlap.
        
        Args:
            text1: First text chunk
            text2: Second text chunk
            threshold: Similarity threshold for considering overlap significant
            
        Returns:
            True if significant overlap exists
        """
        # Quick check using token sets
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return False
            
        jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
        return jaccard > threshold

    def chunk_by_semantic_similarity(self, text: str, target_chunks: int = 10) -> List[str]:
        """
        Chunk text based on semantic similarity using clustering.
        
        Args:
            text: Text to chunk
            target_chunks: Target number of chunks to create
            
        Returns:
            List of semantically coherent chunks
        """
        if not text:
            return []
            
        # First get sentence-level chunks
        sentences = sent_tokenize(text)
        
        # Handle case where there are fewer sentences than target chunks
        if len(sentences) <= target_chunks:
            return sentences
        
        try:
            # Embed sentences
            embeddings = self.embedding_model.encode(sentences, show_progress_bar=False)
            
            # Cluster sentences
            kmeans = KMeans(n_clusters=min(target_chunks, len(sentences)), 
                           random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            # Group sentences by cluster
            cluster_to_sentences = {i: [] for i in range(min(target_chunks, len(sentences)))}
            for i, cluster_id in enumerate(clusters):
                cluster_to_sentences[cluster_id].append(sentences[i])
            
            # Join sentences within each cluster
            chunks = []
            for cluster_sents in cluster_to_sentences.values():
                if cluster_sents:
                    # Sort sentences by their original order in the text
                    sorted_sents = sorted(cluster_sents, key=lambda s: sentences.index(s))
                    chunks.append(' '.join(sorted_sents))
            
            return chunks
        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            # Fallback to sentence chunking
            return self.chunk_by_sentences(text)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
            
        try:
            return self.embedding_model.encode(texts, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Return zero vectors in case of failure
            return np.zeros((len(texts), self.embedding_model.get_sentence_embedding_dimension()))

    def expand_query(self, query: str) -> str:
        """
        Expand query with keywords to improve retrieval.
        
        Args:
            query: Original query text
            
        Returns:
            Expanded query
        """
        # Remove stopwords to focus on key terms
        query_terms = [word for word in query.lower().split() if word not in self.stop_words]
        
        # For demonstration purposes, a simple expansion technique
        # In a real implementation, this could be replaced with a more sophisticated
        # method like WordNet expansion, word embedding similarity, etc.
        return query + " " + " ".join(query_terms)


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
        if not documents:
            logger.warning("No documents provided to add to vector store")
            return
            
        try:
            embeddings = np.vstack([doc.embedding for doc in documents if doc.embedding is not None])
            if embeddings.shape[0] == 0:
                logger.warning("No valid embeddings found in documents")
                return
                
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            self.index.add(embeddings)
            self.documents.extend([doc for doc in documents if doc.embedding is not None])
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using the query embedding.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if query_embedding is None or len(query_embedding) == 0:
            logger.error("Invalid query embedding provided")
            return []
            
        try:
            # Safety check for k
            k = min(k, len(self.documents))
            if k == 0:
                return []
                
            # Normalize query embedding
            query_embedding_norm = query_embedding.copy().astype(np.float32)
            faiss.normalize_L2(query_embedding_norm.reshape(1, -1))
            
            # Search
            scores, indices = self.index.search(query_embedding_norm.reshape(1, -1), k)
            
            # Return documents and scores
            results = [(self.documents[idx], float(score)) for idx, score in zip(indices[0], scores[0]) 
                      if 0 <= idx < len(self.documents)]
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []


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
        
        try:
            # Initialize LLM for generation
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
            self.generation_pipeline = pipeline(
                "text2text-generation",
                model=self.model, 
                tokenizer=self.tokenizer,
                max_length=512
            )
        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            raise ValueError(f"Failed to initialize language model: {str(e)}")

    def ingest_data(self, data: Union[pd.DataFrame, List[str]], text_column: Optional[str] = None):
        """
        Ingest data into the RAG system.
        
        Args:
            data: DataFrame or list of texts to ingest
            text_column: Column name containing text (for DataFrame)
            
        Returns:
            Number of document chunks ingested
        """
        texts = []
        metadata_list = []
        
        # Extract texts and metadata
        try:
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
                if not isinstance(data, list):
                    raise ValueError("Data must be a DataFrame or list of texts")
                texts = data
                metadata_list = [{'id': str(i)} for i in range(len(texts))]
        except Exception as e:
            logger.error(f"Error processing input data: {str(e)}")
            return 0
        
        documents = []
        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            # Clean text
            clean_text = self.text_processor.clean_text(text)
            if not clean_text:
                continue
                
            # Chunk text
            chunks = []
            try:
                if self.chunk_method == "sentences":
                    chunks = self.text_processor.chunk_by_sentences(
                        clean_text, 
                        self.chunk_size, 
                        self.chunk_overlap
                    )
                elif self.chunk_method == "semantic":
                    chunks = self.text_processor.chunk_by_semantic_similarity(clean_text)
                else:
                    logger.warning(f"Unknown chunk method '{self.chunk_method}', falling back to sentence chunking")
                    chunks = self.text_processor.chunk_by_sentences(
                        clean_text, 
                        self.chunk_size, 
                        self.chunk_overlap
                    )
            except Exception as e:
                logger.error(f"Error chunking text: {str(e)}")
                continue
                
            # Skip if no chunks were created
            if not chunks:
                continue
            
            # Create document objects for each chunk
            for j, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = j
                chunk_metadata['original_text_id'] = metadata.get('id', str(i))
                chunk_metadata['chunk_method'] = self.chunk_method
                
                try:
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
                except Exception as e:
                    logger.error(f"Error processing chunk {j} of document {i}: {str(e)}")
        
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
        if not query or not isinstance(query, str):
            logger.error("Invalid query provided")
            return []
            
        try:
            # Expand query for better retrieval
            expanded_query = self.text_processor.expand_query(query)
            
            # Generate query embedding
            query_embedding = self.text_processor.generate_embeddings([expanded_query])[0]
            
            # Retrieve similar documents
            results = self.vector_store.search(query_embedding, k=max(k, 10))  # Get more for reranking
            
            if not results:
                logger.warning("No results found for query")
                return []
                
            if self.reranking_enabled and len(results) > 1:
                # Rerank results based on query relevance
                reranked_results = self._rerank_results(query, results)
                return [doc for doc, _ in reranked_results[:k]]
            
            return [doc for doc, _ in results[:k]]
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return []
    
    def _rerank_results(self, query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """
        Rerank results using a more sophisticated method.
        
        Args:
            query: Original query
            results: Initial retrieval results
            
        Returns:
            Reranked results
        """
        if not results:
            return []
            
        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]
        
        try:
            # Get query embedding again for direct comparison
            query_embedding = self.text_processor.generate_embeddings([query])[0]
            
            # BM25-style term frequency scoring
            query_terms = [term.lower() for term in query.split() if term.lower() not in self.text_processor.stop_words]
            
            for i, doc in enumerate(documents):
                # 1. Term frequency scoring
                doc_text = doc.text.lower()
                term_score = 0
                for term in query_terms:
                    # Count occurrences of term in document (simple TF)
                    term_count = doc_text.count(term)
                    # Simple TF-IDF style weighting
                    if term_count > 0:
                        term_score += (1 + np.log(term_count))
                
                # 2. Position bias - terms appearing earlier get higher weight
                position_score = 0
                for term in query_terms:
                    pos = doc_text.find(term)
                    if pos >= 0:
                        # Terms appearing earlier get higher scores
                        position_score += 1.0 / (1.0 + pos / 100.0)
                
                # 3. Exact phrase matching
                phrase_score = 1.0
                if query.lower() in doc_text:
                    phrase_score = 1.5  # Boost for exact phrase match
                
                # Combine scores with the original similarity
                combined_score = scores[i] * (1.0 + 0.2 * term_score + 0.1 * position_score) * phrase_score
                scores[i] = combined_score
            
            # Sort by new scores
            reranked_results = [(doc, score) for doc, score in sorted(
                zip(documents, scores), 
                key=lambda x: x[1],
                reverse=True
            )]
            
            return reranked_results
        except Exception as e:
            logger.error(f"Error reranking results: {str(e)}")
            # Return original results if reranking fails
            return results
    
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
        if not query:
            return "Please provide a valid query."
            
        try:
            # Retrieve relevant documents
            retrieved_docs = self.query(query, k=k)
            
            if not retrieved_docs:
                return "I couldn't find any relevant information to answer your query."
            
            # Construct context from retrieved documents
            context = "\n\n".join([doc.text for doc in retrieved_docs])
            
            # Construct prompt
            if prompt_template:
                prompt = prompt_template.format(query=query, context=context)
            else:
                prompt = f"Context information is below.\n\n{context}\n\nGiven the context information and not prior knowledge, answer the query.\nQuery: {query}\nAnswer:"
            
            # Generate response
            response = self.generation_pipeline(prompt)[0]['generated_text']
            
            # For T5 models, the output is already just the answer
            if response.startswith(prompt):
                # For models that might repeat the prompt, extract just the answer
                response = response[len(prompt):].strip()
                
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while generating a response."


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
        if not story_id:
            logger.error("Invalid story ID provided")
            return {}
            
        try:
            # Find all chunks for this story
            story_docs = [doc for doc in self.vector_store.documents 
                         if doc.metadata.get("original_text_id") == story_id]
            
            if not story_docs:
                logger.warning(f"No documents found for story ID {story_id}")
                return {}
            
            # Enhanced literary themes to analyze
            themes = [
                "love", "adventure", "tragedy", "comedy", "mystery",
                "family", "friendship", "betrayal", "growth", "conflict",
                "redemption", "journey", "heroism", "sacrifice", "identity"
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
        except Exception as e:
            logger.error(f"Error analyzing story themes: {str(e)}")
            return {}
    
    def find_similar_stories(self, story_id: str, k: int = 3) -> List[str]:
        """
        Find stories similar to the given story.
        
        Args:
            story_id: ID of the reference story
            k: Number of similar stories to return
            
        Returns:
            List of similar story IDs
        """
        if not story_id:
            logger.error("Invalid story ID provided")
            return []
            
        try:
            # Find all chunks for this story
            story_docs = [doc for doc in self.vector_store.documents 
                         if doc.metadata.get("original_text_id") == story_id]
            
            if not story_docs:
                logger.warning(f"No documents found for story ID {story_id}")
                return []
            
            # Calculate average embedding for the story
            story_embeddings = np.vstack([doc.embedding for doc in story_docs])
            avg_story_embedding = np.mean(story_embeddings, axis=0)
            
            # Search similar documents
            results = self.vector_store.search(avg_story_embedding, k=k*5)  # Retrieve more for diversity
            
            # Get unique story IDs (excluding the original story)
            similar_story_ids = []
            for doc, _ in results:
                doc_story_id = doc.metadata.get("original_text_id")
                if doc_story_id != story_id and doc_story_id not in similar_story_ids:
                    similar_story_ids.append(doc_story_id)
                    if len(similar_story_ids) >= k:
                        break
            
            return similar_story_ids
        except Exception as e:
            logger.error(f"Error finding similar stories: {str(e)}")
            return []


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
