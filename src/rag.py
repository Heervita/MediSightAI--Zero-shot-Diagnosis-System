# src/rag_engine.py
import chromadb
import os
import uuid # To generate unique IDs for each case

class RAGEngine:
    def __init__(self, db_path="data/vector_store"):
        """
        Initializes the Vector Database (ChromaDB).
        Uses PersistentClient so data is saved to disk.
        """
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or get the collection (like a table in SQL)
        self.collection = self.client.get_or_create_collection(
            name="medical_cases",
            metadata={"description": "Confirmed medical cases for RAG retrieval"}
        )

    def add_case(self, image_vector, description, metadata=None):
        """
        Adds a new verified case to the database.
        
        Args:
            image_vector (list): The embedding vector from BiomedCLIP.
            description (str): The medical report/diagnosis text.
            metadata (dict): Extra info like Patient ID, Date, Hospital Name.
        """
        # ChromaDB expects vectors as lists, not PyTorch tensors
        if hasattr(image_vector, 'tolist'):
            image_vector = image_vector.tolist()[0] # Flatten batch dimension

        self.collection.add(
            ids=[str(uuid.uuid4())], # Generate unique ID
            embeddings=[image_vector],
            documents=[description], # This is what gets returned as "Validation"
            metadatas=[metadata if metadata else {}]
        )

    def search_similar(self, query_vector, n_results=3):
        """
        Finds the top N most similar past cases.
        """
        # Ensure vector is a list
        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()[0]

        # Check if DB is empty
        if self.collection.count() == 0:
            return ["Database is empty. Add trusted cases to data/reference_images first."]

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )
        
        # Extract just the documents (descriptions)
        return results['documents'][0]