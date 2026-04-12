import sys
from operator import itemgetter
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.utils.pinecone_store import get_pinecone_vectorstore
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer
from pydantic import ValidationError


class ConversationalRAG:
    """
    LCEL-based Conversational RAG using Pinecone as the vector store.

    Pipeline:
        question rewriter → MultiQueryRetriever → ContextualCompression → LLM answer
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id

            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info("ConversationalRAG initialized", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever_from_pinecone(
        self,
        session_id: str,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        embeddings=None,
    ):
        """
        Build a retriever from a Pinecone namespace.

        Args:
            session_id: Used as the Pinecone namespace to isolate per-user docs
            k: Number of documents to return
            search_type: "similarity" or "mmr"
            fetch_k: Docs fetched before MMR re-ranking (MMR only)
            lambda_mult: MMR diversity param (0=max diversity, 1=max relevance)
            embeddings: Optional pre-loaded embeddings (avoids reloading the model)
        """
        try:
            if embeddings is None:
                embeddings = ModelLoader().load_embeddings(task_type="retrieval_query")
            vectorstore = get_pinecone_vectorstore(embeddings, namespace=session_id)

            search_kwargs: dict = {"k": k}
            if search_type == "mmr":
                search_kwargs["fetch_k"] = fetch_k
                search_kwargs["lambda_mult"] = lambda_mult

            self.retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs=search_kwargs,
            )
            self._build_lcel_chain()
            log.info("Pinecone retriever loaded", session_id=session_id, search_type=search_type, k=k)
            return self.retriever
        except Exception as e:
            log.error("Failed to load Pinecone retriever", error=str(e))
            raise DocumentPortalException("Pinecone retriever error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever_from_pinecone() before invoke().", sys
                )
            chat_history = chat_history or []
            answer = self.chain.invoke({"input": user_input, "chat_history": chat_history})
            if not answer:
                log.warning("No answer generated", user_input=user_input, session_id=self.session_id)
                return "no answer generated."
            try:
                answer = ChatAnswer(answer=str(answer)).answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)
            log.info("Chain invoked successfully", session_id=self.session_id, answer_preview=str(answer)[:150])
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite question as standalone (resolves pronouns, chat history context)
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) MultiQueryRetriever — generates 3 query variants, retrieves for each,
            #    deduplicates → better recall on vague or ambiguous questions
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.retriever,
                llm=self.llm,
            )

            # 3) ContextualCompression — strips irrelevant sentences from each chunk
            #    before passing to LLM → less noise, fewer tokens, better answers
            compressor = LLMChainExtractor.from_llm(self.llm)
            compressed_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=multi_query_retriever,
            )

            # 4) Full chain: rewrite → retrieve → compress → answer
            retrieve_docs = question_rewriter | compressed_retriever | self._format_docs

            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info("LCEL chain built successfully", session_id=self.session_id)
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
