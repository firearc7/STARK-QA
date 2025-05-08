import os
import pandas as pd
import networkx as nx
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import json
import pickle
from pathlib import Path
# Import the Knowledge Graph Generator
from kgg import KnowledgeGraphGenerator

# Import the latest Mistral client (1.0.0+)
from mistralai import Mistral

# Add these imports at the top
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from tqdm import tqdm  # Add for progress bars
import hashlib  # Add for generating story IDs

class KnowledgeAugmentedGenerator:
    """
    A class that augments LLM responses with information from a knowledge graph.
    """
    def __init__(self, model="mistral-tiny", vectordb_dir="vector_db"):
        """Initialize the Knowledge Augmented Generator with Mistral AI client."""
        # Load environment variables
        load_dotenv()
        
        # Initialize Mistral AI client
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in .env file")
        
        # Using the latest unified client
        self.client = Mistral(api_key=api_key)
        
        # Set the model to use
        self.model = model
        
        # Knowledge graph resources
        self.knowledge_graph = None  # NetworkX graph
        self.knowledge_df = None     # Pandas DataFrame
        
        # Vector database settings
        self.vectordb_dir = vectordb_dir
        self.vectorizer = None
        self.tfidf_matrix = None
        self.triplet_texts = []
        
        # Create vector database directory if it doesn't exist
        Path(vectordb_dir).mkdir(parents=True, exist_ok=True)
        
    def load_knowledge_graph(self, graph_path: str = None, df_path: str = None, 
                             graph: nx.Graph = None, df: pd.DataFrame = None,
                             story_id: str = "story"):
        """
        Load the knowledge graph from either files or directly from objects.
        """
        # Try to load from vector database first
        if self._load_from_vectordb(story_id):
            print(f"Successfully loaded knowledge graph and vectors from vector database for {story_id}")
            return
            
        if graph is not None:
            self.knowledge_graph = graph
        elif graph_path is not None:
            self.knowledge_graph = nx.read_gpickle(graph_path)
            
        if df is not None:
            self.knowledge_df = df
        elif df_path is not None:
            self.knowledge_df = pd.read_csv(df_path)
        
        # If we only have the DataFrame, construct the graph
        if self.knowledge_graph is None and self.knowledge_df is not None:
            self._build_graph_from_df()
        
        # If we only have the graph, construct the DataFrame
        elif self.knowledge_df is None and self.knowledge_graph is not None:
            self._build_df_from_graph()
            
        if self.knowledge_graph is None and self.knowledge_df is None:
            raise ValueError("No knowledge graph data provided")
            
        # Build vector representations for the loaded knowledge
        self._build_vector_representations(story_id)
    
    def _build_graph_from_df(self):
        """Build a NetworkX graph from the knowledge DataFrame."""
        G = nx.Graph()
        
        for _, row in self.knowledge_df.iterrows():
            G.add_node(row['node_1'])
            G.add_node(row['node_2'])
            G.add_edge(
                row['node_1'],
                row['node_2'],
                relation=row['relation'],
                weight=row['weight'],
                chunk_id=row['chunk_id']
            )
        
        self.knowledge_graph = G
    
    def _build_df_from_graph(self):
        """Build a DataFrame from the knowledge graph."""
        rows = []
        for u, v, data in self.knowledge_graph.edges(data=True):
            rows.append({
                'node_1': u,
                'node_2': v,
                'relation': data.get('relation', ''),
                'weight': data.get('weight', 1.0),
                'chunk_id': data.get('chunk_id', '')
            })
        
        self.knowledge_df = pd.DataFrame(rows)

    def _build_vector_representations(self, story_id: str):
        """Build TF-IDF vector representations of the knowledge graph triplets."""
        # Create text representation of each knowledge triplet
        self.triplet_texts = []
        
        if self.knowledge_df is None or len(self.knowledge_df) == 0:
            print("Knowledge graph is empty or not loaded")
            return
            
        for _, row in self.knowledge_df.iterrows():
            triplet_text = f"{row['node_1']} {row['relation']} {row['node_2']}"
            self.triplet_texts.append(triplet_text.lower())
        
        # If we have no triplets, return
        if not self.triplet_texts:
            return
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.triplet_texts)
        
        # Save to vector database
        self._save_to_vectordb(story_id)
        
    def _get_vectordb_paths(self, story_id: str):
        """Get paths for vector database files."""
        # Make sure vectordb_dir is a Path object
        base_dir = Path(self.vectordb_dir)
        
        # First check for files directly in the directory (old format)
        vectorizer_path_old = base_dir / f"{story_id}.vectorizer"
        tfidf_path_old = base_dir / f"{story_id}.tfidf"
        triplets_path_old = base_dir / f"{story_id}.triplets"
        graph_path_old = base_dir / f"{story_id}.graph"
        df_path_old = base_dir / f"{story_id}.df"
        
        # Check if old format files exist
        if vectorizer_path_old.exists() and tfidf_path_old.exists() and triplets_path_old.exists():
            print(f"Found vector database files in old format")
            return vectorizer_path_old, tfidf_path_old, triplets_path_old, graph_path_old, df_path_old
        
        # Create a directory for this specific story (new format)
        story_dir = base_dir / story_id
        story_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file paths within that directory
        vectorizer_path = story_dir / "vectorizer.pkl"
        tfidf_path = story_dir / "tfidf.pkl"
        triplets_path = story_dir / "triplets.pkl"
        graph_path = story_dir / "graph.pkl"
        df_path = story_dir / "df.pkl"
        
        return vectorizer_path, tfidf_path, triplets_path, graph_path, df_path
        
    def _save_to_vectordb(self, story_id: str):
        """Save vector representations and knowledge graph to disk."""
        vectorizer_path, tfidf_path, triplets_path, graph_path, df_path = self._get_vectordb_paths(story_id)
        
        try:
            # Create parent directory if it doesn't exist
            Path(vectorizer_path).parent.mkdir(parents=True, exist_ok=True)
            
            print(f"Saving to vector database:")
            print(f"  Vectorizer: {vectorizer_path}")
            print(f"  TFIDF Matrix: {tfidf_path}")
            print(f"  Triplets: {triplets_path}")
            print(f"  Graph: {graph_path}")
            print(f"  DataFrame: {df_path}")
            
            # Save vectorizer
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save TF-IDF matrix
            with open(tfidf_path, 'wb') as f:
                pickle.dump(self.tfidf_matrix, f)
            
            # Save triplet texts
            with open(triplets_path, 'wb') as f:
                pickle.dump(self.triplet_texts, f)
            
            # Save graph - handle different NetworkX versions
            try:
                # Try using nx.write_gpickle if available
                nx.write_gpickle(self.knowledge_graph, graph_path)
            except AttributeError:
                # Fallback to pickle.dump
                with open(graph_path, 'wb') as f:
                    pickle.dump(self.knowledge_graph, f)
            
            # Save dataframe
            self.knowledge_df.to_pickle(df_path)
            
            print(f"Successfully saved vector database for {story_id}")
            return True
        except Exception as e:
            print(f"Error saving vector database: {e}")
            return False
    
    def _load_from_vectordb(self, story_id: str):
        """Load vector representations and knowledge graph from disk."""
        vectorizer_path, tfidf_path, triplets_path, graph_path, df_path = self._get_vectordb_paths(story_id)
        
        # Check which files exist
        vectorizer_exists = vectorizer_path.exists()
        tfidf_exists = tfidf_path.exists()
        triplets_exists = triplets_path.exists()
        graph_exists = graph_path.exists()
        df_exists = df_path.exists()
        
        print(f"Vector database file status:")
        print(f"  Vectorizer: {vectorizer_exists}")
        print(f"  TFIDF: {tfidf_exists}")
        print(f"  Triplets: {triplets_exists}")
        print(f"  Graph: {graph_exists}")
        print(f"  DataFrame: {df_exists}")
        
        # Check if we have at least the vector files (vectorizer, tfidf, triplets)
        if vectorizer_exists and tfidf_exists and triplets_exists:
            try:
                # Load vectorizer
                with open(vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("Loaded vectorizer")
                
                # Load TF-IDF matrix
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_matrix = pickle.load(f)
                print("Loaded TF-IDF matrix")
                
                # Load triplet texts
                with open(triplets_path, 'rb') as f:
                    self.triplet_texts = pickle.load(f)
                print("Loaded triplet texts")
                
                # Try to load graph if it exists
                if graph_exists:
                    try:
                        self.knowledge_graph = nx.read_gpickle(graph_path)
                        print("Loaded graph")
                    except AttributeError:
                        # Try alternate loading method for different nx versions
                        with open(graph_path, 'rb') as f:
                            self.knowledge_graph = pickle.load(f)
                        print("Loaded graph using pickle")
                else:
                    # Since graph is missing, reconstruct it
                    print("Graph file missing, reconstructing knowledge graph...")
                    self._reconstruct_graph_from_triplets()
                    
                # Try to load dataframe if it exists
                if df_exists:
                    self.knowledge_df = pd.read_pickle(df_path)
                    print("Loaded dataframe")
                else:
                    # Since dataframe is missing, reconstruct it
                    print("DataFrame file missing, reconstructing from triplets...")
                    self._reconstruct_df_from_triplets()
                
                print(f"Successfully loaded vector database for {story_id}")
                
                # Save the reconstructed files if needed - using version-safe methods
                try:
                    if not graph_exists and self.knowledge_graph:
                        print("Saving reconstructed graph...")
                        try:
                            # Try using nx.write_gpickle if available
                            nx.write_gpickle(self.knowledge_graph, graph_path)
                        except AttributeError:
                            # Fallback to pickle.dump
                            with open(graph_path, 'wb') as f:
                                pickle.dump(self.knowledge_graph, f)
                        print("Graph saved successfully")
                except Exception as e:
                    print(f"Warning: Could not save graph: {e}")
                    
                try:
                    if not df_exists and not self.knowledge_df.empty:
                        print("Saving reconstructed dataframe...")
                        self.knowledge_df.to_pickle(df_path)
                        print("DataFrame saved successfully")
                except Exception as e:
                    print(f"Warning: Could not save dataframe: {e}")
                    
                return True
            except Exception as e:
                print(f"Error loading vector database: {e}")
                return False
        else:
            print("Missing essential vector files (vectorizer, tfidf, or triplets)")
            return False

    def _reconstruct_graph_from_triplets(self):
        """Reconstruct a knowledge graph from triplet texts."""
        G = nx.Graph()
        
        # Extract nodes and edges from triplet texts
        for i, text in enumerate(self.triplet_texts):
            # Simple parsing of triplet text "node1 relation node2"
            parts = text.split(" ", 2)
            if len(parts) >= 3:
                node_1, relation, node_2 = parts[0], parts[1], parts[2]
                
                G.add_node(node_1)
                G.add_node(node_2)
                G.add_edge(
                    node_1,
                    node_2,
                    relation=relation,
                    weight=1.0,
                    chunk_id=f"chunk_{i}"
                )
        
        self.knowledge_graph = G
        print(f"Reconstructed graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        
    def _reconstruct_df_from_triplets(self):
        """Reconstruct a basic dataframe from triplet texts."""
        rows = []
        for i, text in enumerate(self.triplet_texts):
            # Simple parsing of triplet text "node1 relation node2"
            parts = text.split(" ", 2)
            if len(parts) >= 3:
                node_1, relation, node_2 = parts[0], parts[1], parts[2]
                rows.append({
                    'node_1': node_1,
                    'relation': relation,
                    'node_2': node_2,
                    'weight': 1.0,
                    'chunk_id': f"chunk_{i}"
                })
        
        self.knowledge_df = pd.DataFrame(rows)
        print(f"Reconstructed dataframe with {len(rows)} rows")

    def retrieve_relevant_knowledge(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant knowledge to a query using cosine similarity.
        """
        if self.tfidf_matrix is None or self.vectorizer is None:
            print("Vector representations not built. Falling back to keyword retrieval.")
            return self._keyword_based_retrieval(query, top_n)
        
        try:
            # Transform the query using the same vectorizer
            query_vec = self.vectorizer.transform([query.lower()])
            
            # Calculate cosine similarity between query and each triplet
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get indices of top N similarities
            top_indices = cosine_similarities.argsort()[-top_n:][::-1]
            
            # Only include triplets with non-zero similarity
            relevant_indices = [idx for idx in top_indices if cosine_similarities[idx] > 0]
            
            # Convert to DataFrame indices and get the triplets
            relevant_knowledge = self.knowledge_df.iloc[relevant_indices].to_dict('records')
            
            print(f"Retrieved {len(relevant_knowledge)} relevant knowledge triplets using cosine similarity")
            return relevant_knowledge
        
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            # Fall back to simple keyword matching if vectorization fails
            return self._keyword_based_retrieval(query, top_n)

    def _keyword_based_retrieval(self, query: str, top_n: int = 5) -> List[Dict]:
        """
        Fallback method that uses simple keyword matching when vector similarity fails.
        """
        print("Falling back to keyword-based retrieval")
        query_tokens = set(query.lower().split())
        
        # Score each knowledge triplet based on keyword matching
        scores = []
        for idx, row in self.knowledge_df.iterrows():
            # Combine node texts and relation
            triplet_text = f"{row['node_1']} {row['relation']} {row['node_2']}".lower()
            
            # Calculate how many query tokens appear in the triplet
            matches = sum(1 for token in query_tokens if token in triplet_text)
            
            # Weight by the relationship weight
            score = matches * row['weight']
            scores.append((idx, score))
        
        # Get the indices of the top-scoring triplets
        top_indices = [idx for idx, score in sorted(scores, key=lambda x: x[1], reverse=True)[:top_n] if score > 0]
        
        # Return the top knowledge triplets
        relevant_knowledge = self.knowledge_df.iloc[top_indices].to_dict('records') if top_indices else []
        return relevant_knowledge
    
    def format_knowledge_as_context(self, knowledge_items: List[Dict], max_chars: int = 10000) -> str:
        """
        Format knowledge triplets into a compact readable context for the LLM.
        """
        if not knowledge_items:
            return "No relevant knowledge available."
        
        # Sort items by weight (most important first)
        sorted_items = sorted(knowledge_items, key=lambda x: x.get('weight', 0), reverse=True)
        
        context_items = []
        char_count = 0
        
        for item in sorted_items:
            # Format this item
            relation_text = str(item['relation']).replace(';', ',')
            node_1 = str(item['node_1'])
            node_2 = str(item['node_2'])
            
            # Create a compact representation
            item_text = f"{node_1} {relation_text} {node_2}"
            
            # Check if adding this would exceed our max
            if char_count + len(item_text) + 2 > max_chars:  # +2 for newline
                break
                
            context_items.append(item_text)
            char_count += len(item_text) + 2  # +2 for newline
        
        # Join items with newlines
        if context_items:
            context = "Relevant knowledge:\n- " + "\n- ".join(context_items)
            return context
        else:
            return "No relevant knowledge available."

    def expand_knowledge_context(self, initial_items: List[Dict], depth: int = 1) -> List[Dict]:
        """
        Expand the knowledge context by including neighbors in the knowledge graph.
        """
        if self.knowledge_graph is None or not initial_items:
            return initial_items
        
        # Extract nodes from initial items
        nodes = set()
        for item in initial_items:
            nodes.add(item['node_1'])
            nodes.add(item['node_2'])
        
        # Expand to neighbors
        expanded_nodes = set(nodes)
        for _ in range(depth):
            neighbors = set()
            for node in expanded_nodes:
                if node in self.knowledge_graph:
                    neighbors.update(self.knowledge_graph.neighbors(node))
            expanded_nodes.update(neighbors)
        
        # Get all edges between expanded nodes
        expanded_items = []
        seen_edges = set()
        
        # First add the initial items
        for item in initial_items:
            edge_key = (item['node_1'], item['node_2'])
            seen_edges.add(edge_key)
            seen_edges.add(edge_key[::-1])  # Add reverse too since we have an undirected graph
            expanded_items.append(item)
        
        # Then add new edges from the expanded graph
        for node1 in expanded_nodes:
            if node1 not in self.knowledge_graph:
                continue
                
            for node2 in self.knowledge_graph.neighbors(node1):
                if node2 in expanded_nodes and (node1, node2) not in seen_edges:
                    edge_data = self.knowledge_graph.get_edge_data(node1, node2)
                    item = {
                        'node_1': node1,
                        'node_2': node2,
                        'relation': edge_data.get('relation', ''),
                        'weight': edge_data.get('weight', 1.0),
                        'chunk_id': edge_data.get('chunk_id', '')
                    }
                    expanded_items.append(item)
                    seen_edges.add((node1, node2))
                    seen_edges.add((node2, node1))
        
        return expanded_items
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")  # Using a standard encoding
            tokens = encoding.encode(text)
            return len(tokens)
        except:
            # Fallback to approximate count if tiktoken not available
            return len(text.split()) * 1.3  # Rough approximation
    
    def generate_response(self, query: str, use_knowledge: bool = True, 
                         expand_depth: int = 0, temperature: float = 0.7) -> str:
        """
        Generate a response to a user query, augmented with knowledge graph information.
        """
        if use_knowledge and (self.knowledge_graph is not None or self.knowledge_df is not None):
            # Retrieve relevant knowledge - limit to just 3 items to be safe
            knowledge_items = self.retrieve_relevant_knowledge(query, top_n=10)
            
            # NO EXPANSION - this is likely what's causing the token explosion
            # The expand_depth parameter is now ignored
            
            # Format as context with character limit
            knowledge_context = self.format_knowledge_as_context(knowledge_items, max_chars=5000)
            
            # Create system prompt with knowledge context
            system_content = f"""You are a helpful assistant that answers questions based on the following knowledge:

{knowledge_context}

Answer the user's question concisely using this information and your general knowledge."""
            
            # Create messages using dictionary format to be safe
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ]
        else:
            # Standard system prompt without knowledge augmentation
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
        
        try:
            # Call the Mistral API with the new client format
            chat_response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=500  # Limit response length
            )
            
            # Extract and return the response text
            return chat_response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Mistral API: {e}")
            # Provide a fallback response
            return f"I encountered an error when trying to answer your question: {e}. Please try again later."

    def process_stories_and_build_kg(self, stories_df=None, stories_file=None, story_column='content', 
                                     id_column=None, batch_size=10):
        """
        Process multiple stories from a dataframe or file and build a combined knowledge graph.
        
        Args:
            stories_df: Pandas DataFrame containing stories
            stories_file: Path to CSV file with stories
            story_column: Column name containing the story text
            id_column: Column to use as story ID (if None, will generate hash-based IDs)
            batch_size: Number of stories to process in each batch for memory efficiency
            
        Returns:
            DataFrame and Graph of combined knowledge
        """
        # Load stories from file if dataframe not provided
        if stories_df is None and stories_file is not None:
            print(f"Loading stories from {stories_file}...")
            stories_df = pd.read_csv(stories_file)
            
        if stories_df is None:
            raise ValueError("Either stories_df or stories_file must be provided")
        
        print(f"Processing {len(stories_df)} stories...")
        
        # Create a unique combined ID for all stories
        stories_hash = hashlib.md5(story_column.encode() + str(len(stories_df)).encode()).hexdigest()[:8]
        combined_id = f"combined_{stories_hash}"
        
        # Check if we already have this dataset processed
        vectorizer_path, _, _, graph_path, df_path = self._get_vectordb_paths(combined_id)
        
        # Try to load existing combined knowledge graph
        if vectorizer_path.exists() and graph_path.exists() and df_path.exists():
            print(f"Found existing combined knowledge graph for {combined_id}, loading...")
            success = self._load_from_vectordb(combined_id)
            if success:
                print("Successfully loaded existing combined knowledge graph")
                return self.knowledge_df, self.knowledge_graph
        
        # Process stories in batches to manage memory
        all_dfs = []
        combined_graph = nx.Graph()
        total_nodes = 0
        total_edges = 0
        
        # Process in batches
        for i in tqdm(range(0, len(stories_df), batch_size)):
            batch = stories_df.iloc[i:i+batch_size]
            batch_dfs = []
            
            for _, row in batch.iterrows():
                try:
                    # Get story text
                    story = row[story_column]
                    if not isinstance(story, str) or len(story) < 50:
                        print(f"Skipping invalid story: {story[:30]}...")
                        continue
                    
                    # Generate ID for this story
                    if id_column and id_column in row:
                        story_id = f"story_{row[id_column]}"
                    else:
                        story_id = get_story_id(story)
                    
                    # Process the story
                    from kgg import KnowledgeGraphGenerator
                    kg_generator = KnowledgeGraphGenerator()
                    df, G = kg_generator.process_story(story, story_id)
                    
                    # Combine results
                    if len(df) > 0:
                        batch_dfs.append(df)
                        
                        # Merge the graph
                        combined_graph.add_nodes_from(G.nodes(data=True))
                        combined_graph.add_edges_from(G.edges(data=True))
                        
                        total_nodes += len(G.nodes())
                        total_edges += len(G.edges())
                    
                except Exception as e:
                    print(f"Error processing story: {e}")
            
            # Combine batch dataframes and add to all_dfs
            if batch_dfs:
                batch_combined = pd.concat(batch_dfs, ignore_index=True)
                all_dfs.append(batch_combined)
                print(f"Batch {i//batch_size + 1} complete: {len(batch_combined)} new relationships")
        
        # Combine all the dataframes
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            print(f"Combined knowledge graph contains {len(combined_df)} relationships")
            
            # Remove duplicate relationships
            combined_df = combined_df.drop_duplicates(subset=['node_1', 'relation', 'node_2'])
            print(f"After removing duplicates: {len(combined_df)} unique relationships")
        else:
            print("No valid stories were processed")
            combined_df = pd.DataFrame(columns=['node_1', 'relation', 'node_2', 'weight', 'chunk_id'])
        
        # Set the final knowledge graph and dataframe
        self.knowledge_graph = combined_graph
        self.knowledge_df = combined_df
        
        # Build vector representations and save
        print("Building vector representations...")
        self._build_vector_representations(combined_id)
        
        print(f"Created combined knowledge graph with {total_nodes} nodes and {total_edges} edges")
        print(f"DataFrame contains {len(combined_df)} relationships")
        
        return combined_df, combined_graph

    def process_story_and_build_kg(self, story: str, story_id: str = "story"):
        """
        Process a single story and build a knowledge graph.
        """
        # Create a single-row dataframe
        df = pd.DataFrame([{'content': story}])
        
        # Process using the multi-story method
        return self.process_stories_and_build_kg(df, story_column='content')

def get_story_id(story_text):
    """Generate a unique story ID based on the content hash."""
    return "story_" + hashlib.md5(story_text.encode()).hexdigest()[:8]
