from typing import List, Dict, Optional
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from app.core.config import settings
from app.core.logging import logger

class RAGService:
    """Retrieval-Augmented Generation service for CBC curriculum"""
    
    def __init__(self):
        """Initialize RAG components"""
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self._initialize()
    
    def _initialize(self):
        """Initialize all components with error handling"""
        try:
            # Initialize embeddings
            logger.info("Loading embedding model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            logger.info(" Embedding model loaded")
            
            # Initialize vector store
            logger.info("Connecting to vector store...")
            self.vectorstore = Chroma(
                persist_directory=settings.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name="cbc_curriculum"
            )
            logger.info("Vector store connected")
            
            # Initialize LLM
            logger.info("Initializing LLM...")
            self.llm = ChatOpenAI(
                temperature=settings.llm_temperature,
                model=settings.llm_model,
                max_tokens=settings.max_tokens,
                openai_api_key=settings.openai_api_key
            )
            logger.info(" LLM initialized")
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            raise
    
    def retrieve_context(
        self, 
        query: str, 
        grade: str, 
        subject: str, 
        k: int = 3
    ) -> List[Document]:
        """Retrieve relevant curriculum content"""
        try:
            if not self.vectorstore:
                logger.warning("Vector store not available")
                return []
            
            # Build filter
            filter_dict = {
                "grade": grade,
                "subject": subject.lower()
            }
            
            # Search
            results = self.vectorstore.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Retrieved {len(results)} context chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def build_prompt(
        self,
        query: str,
        context: List[Document],
        grade: str,
        language: str = "en"
    ) -> str:
        """Build CBC-aligned prompt"""
        
        # Format context
        context_text = "\n\n".join([
            f"[Source {i+1}]:\n{doc.page_content}" 
            for i, doc in enumerate(context)
        ])
        
        # Language instruction
        lang_map = {
            "en": "English",
            "sw": "Kiswahili"
        }
        lang_instruction = f"Respond in {lang_map.get(language, 'English')}"
        
        # Build prompt
        prompt = f"""You are AikoLearn, an AI tutor for Kenyan students following the Competency-Based Curriculum (CBC).

Your role:
- Provide clear, step-by-step explanations
- Use age-appropriate language for Grade {grade} students
- Align all answers with CBC learning outcomes
- Be encouraging, patient, and supportive
- Guide students to understand concepts rather than just giving answers
- {lang_instruction}

Relevant CBC curriculum context:
{context_text if context else "No specific curriculum content found for this query."}

Student question: {query}

Instructions:
1. If this is a homework question, guide the student through the thinking process
2. Use examples relevant to Kenyan context when possible
3. Reference the CBC strand or learning outcome if applicable
4. Keep explanations clear and concise
5. Encourage the student to try solving similar problems

Your response:"""
        
        return prompt
    
    async def generate_answer(
        self,
        query: str,
        grade: str,
        subject: str,
        language: str = "en"
    ) -> Dict:
        """Generate answer using RAG"""
        
        try:
            # Retrieve context
            context = self.retrieve_context(query, grade, subject)
            
            # Check if context found
            if not context:
                logger.warning(f"No context found for: {query}")
                return {
                    "answer": "I don't have specific curriculum information for that topic yet. Could you try asking about Grade 7 Math or Science topics?",
                    "sources": [],
                    "has_context": False
                }
            
            # Build prompt
            prompt = self.build_prompt(query, context, grade, language)
            
            # Generate response
            logger.info(f"Generating response for: {query[:50]}...")
            response = self.llm.predict(prompt)
            
            # Format sources
            sources = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in context
            ]
            
            logger.info("✅ Response generated successfully")
            
            return {
                "answer": response,
                "sources": sources,
                "has_context": True
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

# Create global instance
rag_service = RAGService()