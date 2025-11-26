from __future__ import annotations

import os
import shutil
from typing import List, Dict, Optional
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import google.generativeai as genai
from langchain_chroma import Chroma

from app.core.config import settings
from app.core.logging import logger


class RAGService:
    """Retrieval-Augmented Generation service for CBC curriculum"""
    
    def __init__(self):
        self._embeddings = None
        self._vectorstore = None
        self._llm = None
        self.llm_type = None
        self._initialize_llm()
    
    # ---------------------
    # Lazy-loaded embeddings
    # ---------------------
    @property
    def embeddings(self):
        if self._embeddings is None:
            try:
                logger.info("Loading embedding model...")
                self._embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                logger.info("Embedding model loaded.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embeddings
    
    # ---------------------
    # Lazy-loaded vector store with auto-recovery
    # ---------------------
    @property
    def vectorstore(self):
        if self._vectorstore is None:
            logger.info("Connecting to Chroma vectorstore...")
            
            persist_dir = Path(settings.chroma_persist_dir)
            
            try:
                # Try to connect to existing database
                self._vectorstore = Chroma(
                    persist_directory=str(persist_dir),
                    embedding_function=self.embeddings,
                    collection_name="cbc_curriculum"
                )
                logger.info("Vectorstore connected successfully.")
                
            except Exception as e:
                logger.error(f"Vectorstore connection failed: {e}")
                
                # Check if it's a schema error
                if "no such column" in str(e).lower() or "operational" in str(e).lower():
                    logger.warning("Detected corrupted/incompatible ChromaDB. Attempting recovery...")
                    
                    # Backup old database if it exists
                    if persist_dir.exists():
                        backup_dir = persist_dir.parent / f"{persist_dir.name}_backup"
                        if backup_dir.exists():
                            shutil.rmtree(backup_dir)
                        
                        shutil.move(str(persist_dir), str(backup_dir))
                        logger.info(f"Backed up old database to {backup_dir}")
                    
                    # Create fresh vectorstore
                    try:
                        persist_dir.mkdir(parents=True, exist_ok=True)
                        self._vectorstore = Chroma(
                            persist_directory=str(persist_dir),
                            embedding_function=self.embeddings,
                            collection_name="cbc_curriculum"
                        )
                        logger.info("Created fresh vectorstore successfully.")
                        logger.warning("⚠️ Vector database was reset. You'll need to re-ingest your curriculum data.")
                    except Exception as e2:
                        logger.error(f"Failed to create fresh vectorstore: {e2}")
                        raise
                else:
                    raise
                    
        return self._vectorstore
    
    # ---------------------
    # LLM Initialization
    # ---------------------
    @property
    def llm(self):
        if self._llm is None:
            raise ValueError("LLM not initialized")
        return self._llm

    def _initialize_llm(self):
        """Initialize Gemini or OpenAI LLM with proper model name handling."""
        try:
            provider = settings.llm_provider.lower()
            logger.info(f"Initializing LLM provider={provider}")

            if provider == "google" and settings.google_api_key:
                genai.configure(api_key=settings.google_api_key)

                # Clean model name: ensure it has "models/" prefix for API
                requested = settings.llm_model
                if not requested.startswith("models/"):
                    model_name = f"models/{requested}"
                else:
                    model_name = requested
                
                logger.info(f"Using Gemini model: {model_name}")

                # Validate model exists
                try:
                    available_models = [m.name for m in genai.list_models()]
                    if model_name not in available_models:
                        logger.warning(
                            f"Model '{model_name}' not found in available models. "
                            f"Available models: {available_models[:5]}... "
                            f"Attempting to use anyway."
                        )
                except Exception as e:
                    logger.warning(f"Could not list models: {e}. Proceeding with {model_name}")

                generation_config = genai.GenerationConfig(
                    temperature=settings.llm_temperature,
                    max_output_tokens=settings.max_tokens,
                )

                self._llm = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config
                )
                self.llm_type = "google"
                logger.info("✓ Gemini LLM initialized successfully.")

            elif provider == "openai" and settings.openai_api_key:
                self._llm = ChatOpenAI(
                    temperature=settings.llm_temperature,
                    model=settings.llm_model,
                    max_tokens=settings.max_tokens,
                    openai_api_key=settings.openai_api_key
                )
                self.llm_type = "openai"
                logger.info("✓ OpenAI LLM initialized successfully.")

            else:
                raise ValueError(
                    f"No valid LLM provider or API key provided. "
                    f"Provider: {provider}, "
                    f"Google key: {'set' if settings.google_api_key else 'missing'}, "
                    f"OpenAI key: {'set' if settings.openai_api_key else 'missing'}"
                )

        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            raise

    # ---------------------
    # Context Retrieval
    # ---------------------
    def retrieve_context(
        self, 
        query: str, 
        grade: str, 
        subject: str, 
        k: int = 3
    ) -> List[Document]:
        """Retrieve relevant context from vectorstore."""
        try:
            # Normalize inputs
            subject_normalized = subject.lower().strip()
            grade_normalized = grade.strip()
            
            # Build filter
            filter_dict = {
                "$and": [
                    {"grade": {"$eq": grade_normalized}},
                    {"subject": {"$eq": subject_normalized}}
                ]
            }

            logger.info(f"Searching vectorstore: query='{query[:50]}...', grade={grade_normalized}, subject={subject_normalized}, k={k}")
            
            results = self.vectorstore.similarity_search(
                query, 
                k=k, 
                filter=filter_dict
            )

            logger.info(f"Retrieved {len(results)} context chunks.")
            return results

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []

    # ---------------------
    # Prompt Construction
    # ---------------------
    def build_prompt(
        self, 
        query: str, 
        context: List[Document], 
        grade: str, 
        language: str = "en"
    ) -> str:
        """Build RAG prompt with context."""
        
        # Format context
        if context:
            context_text = "\n\n".join([
                f"[Source {i+1}]:\n{doc.page_content}"
                for i, doc in enumerate(context)
            ])
        else:
            context_text = "No specific curriculum data found for this query."

        # Language instruction
        lang_map = {
            "en": "English", 
            "sw": "Kiswahili",
            "swahili": "Kiswahili"
        }
        lang_instruction = f"Respond in {lang_map.get(language.lower(), 'English')}."

        # Build prompt
        prompt = f"""You are AikoLearn, an AI tutor helping Grade {grade} students in Kenya using the Competency-Based Curriculum (CBC).

{lang_instruction}

CBC-aligned reference content:
{context_text}

Student question:
{query}

Instructions:
- Use CBC-aligned reasoning and terminology
- Be simple, clear, and encouraging
- Guide the student step-by-step
- Use examples relevant to Kenyan context
- Reference CBC strands and learning outcomes where applicable
- If the context doesn't fully answer the question, acknowledge this and provide general guidance

Your response:
"""
        return prompt
    
    # ---------------------
    # RAG Answer Generation
    # ---------------------
    async def generate_answer(
        self, 
        query: str, 
        grade: str, 
        subject: str, 
        language: str = "en",
        k: int = 3
    ) -> Dict:
        """Generate answer using RAG approach."""
        try:
            # Retrieve context
            context = self.retrieve_context(query, grade, subject, k=k)

            # Check if we have context
            has_context = len(context) > 0

            if not has_context:
                logger.warning(f"No context found for query: grade={grade}, subject={subject}")
                return {
                    "answer": (
                        "I don't have specific curriculum content for this question yet. "
                        f"Please ensure the Grade {grade} {subject} curriculum has been loaded into the system, "
                        "or try asking a different question."
                    ),
                    "sources": [],
                    "has_context": False
                }

            # Build prompt
            prompt = self.build_prompt(query, context, grade, language)

            logger.info(f"Generating answer using {self.llm_type}...")

            # Generate response based on LLM type
            if self.llm_type == "google":
                resp = self.llm.generate_content(prompt)
                response_text = resp.text

            elif self.llm_type == "openai":
                response_text = self.llm.invoke(prompt).content
            
            else:
                raise ValueError(f"Unknown LLM type: {self.llm_type}")

            # Format sources
            sources = [
                {
                    "content": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                    "metadata": doc.metadata
                }
                for doc in context
            ]

            logger.info("Answer generated successfully.")
            
            return {
                "answer": response_text,
                "sources": sources,
                "has_context": True
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            raise

    # ---------------------
    # Utility Methods
    # ---------------------
    def check_vectorstore_health(self) -> Dict:
        """Check if vectorstore is accessible and has data."""
        try:
            # Try to get collection info
            collection = self.vectorstore._collection
            count = collection.count()
            
            return {
                "status": "healthy",
                "document_count": count,
                "collection_name": collection.name
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "document_count": 0
            }

    def reset_vectorstore(self) -> bool:
        """Reset the vectorstore (use with caution)."""
        try:
            persist_dir = Path(settings.chroma_persist_dir)
            
            # Close existing connection
            self._vectorstore = None
            
            # Remove directory
            if persist_dir.exists():
                shutil.rmtree(persist_dir)
                logger.info(f"Removed vectorstore at {persist_dir}")
            
            # Recreate
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Force recreation on next access
            _ = self.vectorstore
            
            logger.info("Vectorstore reset successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset vectorstore: {e}")
            return False


# Global instance
rag_service = RAGService()