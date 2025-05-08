from kag import KnowledgeAugmentedGenerator
import os
import hashlib
import pandas as pd
from pathlib import Path

def get_story_id(story_text):
    """Generate a unique story ID based on the content hash."""
    return "story_" + hashlib.md5(story_text.encode()).hexdigest()[:8]

def load_stories_from_folder(folder_path):
    """Load all stories from text files in the specified folder."""
    stories = []
    
    # Ensure folder path exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []
    
    print(f"Loading stories from folder: {folder_path}")
    
    # Get all .txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"No .txt files found in folder: {folder_path}")
        return []
    
    print(f"Found {len(txt_files)} text files")
    
    # Load each file
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                story_id = filename.replace('.txt', '')
                stories.append({'id': story_id, 'content': content})
                print(f"Loaded story '{story_id}' ({len(content)} characters)")
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
    
    return stories

def main():
    # Check if a stories folder path is provided as an argument
    import sys
    if len(sys.argv) > 1:
        stories_folder = sys.argv[1]
    else:
        # Default stories folder path
        stories_folder = os.path.join(os.path.dirname(__file__), "stories")
    
    print(f"Using stories folder: {stories_folder}")
    
    # Load all stories from the folder
    stories = load_stories_from_folder(stories_folder)
    
    if not stories:
        print("No stories found. Falling back to sample story.")
        use_sample_story()
        return
    
    # Convert stories to DataFrame format
    stories_df = pd.DataFrame(stories)
    print(f"Created DataFrame with {len(stories_df)} stories")
    
    # Configure vector database directory
    vectordb_dir = os.path.join(os.path.dirname(__file__), "vector_db")
    print(f"Vector database directory: {vectordb_dir}")
    
    # Initialize the Knowledge Augmented Generator
    kag = KnowledgeAugmentedGenerator(model="mistral-tiny", vectordb_dir=vectordb_dir)
    
    # Process all stories and build a combined knowledge graph
    print("\nProcessing all stories to build knowledge graph...")
    df, G = kag.process_stories_and_build_kg(
        stories_df=stories_df, 
        story_column='content',
        id_column='id',
        batch_size=5  # Process 5 stories at a time to avoid memory issues
    )
    
    # Example queries related to stories
    queries = [
        "Who awoke on their bed on the second floor of the building?",
        "Who hopped about the deserted sidewalks of the slum ring?",
        "Who took took the vial and folder?"
    ]
    
    # Generate responses for each query
    print("\n--- Knowledge Augmented Responses ---")
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Generate response with knowledge augmentation
        response = kag.generate_response(query, use_knowledge=True, expand_depth=0)
        print(f"Response: {response}")

def use_sample_story():
    """Fall back to using a single sample story."""
    print("Falling back to single story processing...")
    
    # Read the story from file instead of hardcoding it
    story_path = os.path.join(os.path.dirname(__file__), "story.txt")
    try:
        with open(story_path, 'r', encoding='utf-8') as f:
            story = f.read()
        print(f"Successfully loaded story from {story_path}")
        
        # Generate a unique story ID based on content hash
        story_id = get_story_id(story)
        print(f"Story ID: {story_id}")
    except Exception as e:
        print(f"Error reading story file: {e}")
        # Fallback to a simple example story
        story = """
        Mary had a little lamb whose fleece was white as snow.
        And everywhere that Mary went, the lamb was sure to go.
        """
        story_id = "fallback_story"
        print("Using fallback example story instead")
    
    # Initialize the Knowledge Augmented Generator
    vectordb_dir = os.path.join(os.path.dirname(__file__), "vector_db")
    kag = KnowledgeAugmentedGenerator(model="mistral-tiny", vectordb_dir=vectordb_dir)
    
    # Process the story and build knowledge graph
    df, G = kag.process_story_and_build_kg(story, story_id)
    
    # Run example queries
    queries = [
        "Who saved the masked girl from the oncoming car?",
        "What did the car have welded to its fender?",
        "What piece of clothing did the girl's skirt get caught on?"
    ]
    
    print("\n--- Knowledge Augmented Responses ---")
    for query in queries:
        print(f"\nQuery: {query}")
        response = kag.generate_response(query, use_knowledge=True)
        print(f"Response: {response}")

if __name__ == "__main__":
    main()