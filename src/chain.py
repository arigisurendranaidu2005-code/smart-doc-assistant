"""RAG chain construction using LangChain."""
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from src.config import settings

logger = logging.getLogger(__name__)

class RAGChainManager:
    """Manages the Conversational RAG chain."""

    def __init__(self, retriever: Any):
        """Initialize with a configured retriever."""
        self.retriever = retriever
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferWindowMemory(
            k=settings.memory_window_size,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )

    def _initialize_llm(self) -> Any:
        """Initialize the LLM based on configuration."""
        if settings.llm_provider == "ollama":
            logger.info(f"Initializing Ollama LLM with model {settings.llm_model}")
            return ChatOllama(
                model=settings.llm_model,
                temperature=settings.temperature,
            )
        elif settings.llm_provider == "openai":
            logger.info(f"Initializing OpenAI LLM with model {settings.llm_model}")
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables.")
            return ChatOpenAI(
                model_name=settings.llm_model,
                temperature=settings.temperature,
                api_key=settings.openai_api_key,
                streaming=True
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    def build_chain(self):
        """Build the conversational retrieval chain."""
        
        # 1. Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # 2. Q&A Prompt
        qa_system_prompt = (
            "You are a helpful AI assistant tasked with answering questions based ONLY on the provided context.\n"
            "Rules:\n"
            "1. Answer strictly based on the provided context.\n"
            "2. If the answer is not contained in the context, explicitly say 'I don't know based on the provided context.' Do not hallucinate.\n"
            "3. Cite the sources (filename and page/chunk) you used to construct your answer.\n"
            "4. Keep your answer concise and structured.\n"
            "\n"
            "Context:\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        # 3. Final Retrieval Chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain

    def ask(self, question: str) -> dict:
        """Ask a question to the RAG chain."""
        chain = self.build_chain()
        chat_history = self.memory.chat_memory.messages
        
        response = chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        
        # Save context to memory
        self.memory.save_context(
            {"input": question},
            {"answer": response["answer"]}
        )
        
        return {
            "answer": response["answer"],
            "source_documents": response.get("context", [])
        }
        
    def ask_stream(self, question: str):
        """Stream the answer from the RAG chain."""
        chain = self.build_chain()
        chat_history = self.memory.chat_memory.messages
        
        # This yields chunks of the response
        for chunk in chain.stream({
            "input": question,
            "chat_history": chat_history
        }):
            if "answer" in chunk:
                yield chunk["answer"]
