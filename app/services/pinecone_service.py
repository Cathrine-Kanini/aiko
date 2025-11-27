import pinecone
import os
from app.core.config import settings
from app.core.logging import logger

class PineconeService:
    def __init__(self):
        self.index = None
        self.initialized = False
        
    def initialize(self):
        """Initialize Pinecone connection"""
        try:
            if not settings.pinecone_api_key:
                logger.warning("Pinecone API key not set")
                return False
                
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Get or create index
            if settings.pinecone_index_name not in pinecone.list_indexes():
                # Create index if it doesn't exist
                pinecone.create_index(
                    name=settings.pinecone_index_name,
                    dimension=384,  # Adjust based on your embeddings
                    metric="cosine"
                )
                logger.info(f"Created new Pinecone index: {settings.pinecone_index_name}")
            
            self.index = pinecone.Index(settings.pinecone_index_name)
            self.initialized = True
            logger.info("✅ Pinecone initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pinecone initialization failed: {e}")
            return False
    
    def search(self, query_embedding, top_k=5, filter_dict=None):
        """Search for similar vectors"""
        if not self.initialized:
            raise Exception("Pinecone not initialized")
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            return results
        except Exception as e:
            logger.error(f"Pinecone search error: {e}")
            return None
    
    def upsert(self, vectors):
        """Add or update vectors"""
        if not self.initialized:
            raise Exception("Pinecone not initialized")
        
        try:
            self.index.upsert(vectors=vectors)
            return True
        except Exception as e:
            logger.error(f"Pinecone upsert error: {e}")
            return False

# Global instance
pinecone_service = PineconeService()