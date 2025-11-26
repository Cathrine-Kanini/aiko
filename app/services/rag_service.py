from __future__ import annotations

from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import google.generativeai as genai

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
            logger.info("Loading embedding model...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embedding model loaded.")
        return self._embeddings
    
    # ---------------------
    # Lazy-loaded vector store
    # ---------------------
    @property
    def vectorstore(self):
        if self._vectorstore is None:
            logger.info("Connecting to Chroma vectorstore...")
            self._vectorstore = Chroma(
                persist_directory=settings.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="cbc_curriculum"
            )
            logger.info("Vectorstore connected.")
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
            provider = settings.llm_provider
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
                            f"Model '{model_name}' not found in list. "
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
                logger.info("Gemini LLM initialized successfully.")

            elif provider == "openai" and settings.openai_api_key:
                self._llm = ChatOpenAI(
                    temperature=settings.llm_temperature,
                    model=settings.llm_model,
                    max_tokens=settings.max_tokens,
                    openai_api_key=settings.openai_api_key
                )
                self.llm_type = "openai"
                logger.info("OpenAI LLM initialized successfully.")

            else:
                raise ValueError("No valid LLM provider or API key provided.")

        except Exception as e:
            logger.error(f"LLM initialization error: {e}")
            raise

    # ---------------------
    # Context Retrieval
    # ---------------------
    def retrieve_context(self, query: str, grade: str, subject: str, k: int = 3) -> List[Document]:
        try:
            filter_dict = {
                "$and": [
                    {"grade": {"$eq": grade}},
                    {"subject": {"$eq": subject.lower()}}
                ]
            }

            results = self.vectorstore.similarity_search(
                query, k=k, filter=filter_dict
            )

            logger.info(f"Retrieved {len(results)} context chunks.")
            return results

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return []

    # ---------------------
    # Prompt Construction
    # ---------------------
    def build_prompt(self, query, context, grade, language="en"):
        context_text = "\n\n".join([
            f"[Source {i+1}]:\n{doc.page_content}"
            for i, doc in enumerate(context)
        ]) if context else "No curriculum data found."

        lang_map = {"en": "English", "sw": "Kiswahili"}
        lang_instruction = f"Respond in {lang_map.get(language, 'English')}."

        prompt = f"""
You are AikoLearn, an AI tutor helping Grade {grade} students in Kenya.

{lang_instruction}

CBC-aligned reference content:
{context_text}

Student question:
{query}

Instructions:
- Use CBC-aligned reasoning.
- Be simple, clear, and encouraging.
- Guide the student step-by-step.
- Use examples relevant to Kenya.
- Reference CBC strands where possible.

Your response:
"""
        return prompt
    
    # ---------------------
    # RAG Answer Generation
    # ---------------------
    async def generate_answer(self, query, grade, subject, language="en") -> Dict:
        try:
            context = self.retrieve_context(query, grade, subject)

            if not context:
                return {
                    "answer": "I don't have curriculum context for that yet. Try asking another Grade 7 Math or Science question.",
                    "sources": [],
                    "has_context": False
                }

            prompt = self.build_prompt(query, context, grade, language)

            logger.info(f"Generating answer using {self.llm_type}...")

            # Google Gemini
            if self.llm_type == "google":
                resp = self.llm.generate_content(prompt)
                response_text = resp.text

            # OpenAI
            else:
                response_text = self.llm.invoke(prompt).content

            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in context
            ]

            return {
                "answer": response_text,
                "sources": sources,
                "has_context": True
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise


# Global instance
rag_service = RAGService()