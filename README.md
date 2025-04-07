# STARK-QA

## RAG

Our Retrieval-Augmented Generation (RAG) system is designed to work with short stories. It combines the power of embedding-based retrieval and language model generation to provide contextually relevant responses to user queries. The system is optimized for narrative text and can analyze themes, find similar stories, and generate responses based on retrieved context.

### Features
- **Text Processing**: Cleans and chunks text into manageable pieces for efficient processing.
- **Embedding-Based Retrieval**: Uses Sentence Transformers and FAISS for semantic similarity search.
- **RAG Pipeline**: Combines retrieval with a language model to generate context-aware responses.
- **Theme Analysis**: Identifies literary themes in stories.
- **Story Similarity**: Finds stories similar to a given reference story.

### Components
1. **TextProcessor**: Handles text cleaning, chunking, and embedding generation.
2. **VectorStore**: Manages vector storage and retrieval using FAISS.
3. **RAGPipeline**: Orchestrates the retrieval and generation process.
4. **StoryRAG**: Specialized RAG system optimized for short stories.

### How It Works
1. **Data Ingestion**: The system ingests a dataset of short stories, cleans the text, and chunks it into smaller pieces.
2. **Embedding Generation**: Each chunk is embedded using a Sentence Transformer model.
3. **Vector Storage**: The embeddings are stored in a FAISS index for efficient similarity search.
4. **Querying**: Users can query the system, which retrieves the most relevant chunks based on semantic similarity.
5. **Response Generation**: The retrieved chunks are used as context for a language model to generate a response.
6. **Theme Analysis**: The system can analyze themes in a story by comparing its embedding to predefined theme embeddings.
7. **Story Similarity**: Finds stories similar to a given story based on embedding similarity.

### Dataset
The system works with a dataset of short stories provided as a Pandas DataFrame. Each story should have a unique ID and a text column.

#### Querying
Users can input a query, and the system will retrieve the most relevant story chunks and generate a response.

#### Theme Analysis
The system can analyze themes in a story by providing the story ID.

#### Similar Stories
Users can find stories similar to a given story by providing the story ID.

### Requirements
- Python 3.7+
- Libraries: `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `nltk`, `pandas`, `scikit-learn`

### Installation
1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.

## Future Plans

### Knowledge-Augmented Generation (KAG)
After the mid-evaluation, we plan to extend the project by implementing a Knowledge-Augmented Generation (KAG) system. This system will leverage knowledge graph embeddings to answer questions based solely on structured data extracted from short stories. By utilizing the structured representation of the stories, the KAG system aims to provide precise and contextually relevant answers.

This part of the project will be done by 15th April, 2025. 

### Combining Structured and Unstructured Data
Following the development of the KAG system, we will integrate the context from both textual (unstructured) data and structured data (from the knowledge graph). This combined approach will enable us to build a more robust and comprehensive QA system, capable of leveraging the strengths of both data types to improve accuracy and relevance in responses.

This part of the project will be completed by the date of final submission. 

