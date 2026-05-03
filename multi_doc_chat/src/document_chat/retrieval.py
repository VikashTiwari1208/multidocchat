from __future__ import annotations

import sys
from pathlib import Path
from operator import itemgetter
from typing import Any, List, Optional

import yaml
from pydantic import ValidationError

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from pinecone_text.sparse import BM25Encoder
from pinecone_text.hybrid import hybrid_convex_scale

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.utils.pinecone_store import _pinecone_index
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.model.models import PromptType, ChatAnswer

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "config.yaml"
_cfg = yaml.safe_load(_CONFIG_PATH.read_text())

# Initialise once at import time — BM25Encoder.default() downloads NLTK data
# on first call; re-instantiating it on every query adds unnecessary latency.
_bm25_encoder = BM25Encoder.default()

# FlashrankRerank loads a ~400 MB BERT model (ms-marco-MultiBERT-L-12).
# Lazy-initialised on first chat request and cached — keeps startup fast and
# prevents a model-download failure from crashing the container at boot.
_reranker: "FlashrankRerank | None" = None


def _get_reranker() -> FlashrankRerank:
    global _reranker
    if _reranker is None:
        log.info("Loading FlashrankRerank model into cache (first request)")
        _reranker = FlashrankRerank(
            model=_cfg["reranker"]["model"],
            top_n=_cfg["reranker"]["top_n"],
        )
    return _reranker


# ── Hybrid retriever ──────────────────────────────────────────────────────────

class ParentFetchingHybridRetriever(BaseRetriever):
    """
    Hybrid dense+sparse Pinecone retriever that returns parent-sized chunks.

    For each query:
      1. Encode with BM25 (sparse) + Gemini (dense)
      2. Scale with hybrid_convex_scale(alpha)
      3. Query Pinecone (top_k * 2 to allow deduplication)
      4. Deduplicate by parent_id → return parent_text as Document.page_content
    """

    embeddings: Any
    session_id: str
    k: int = 10
    alpha: float = 0.5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        dense = self.embeddings.embed_query(query)
        sparse = _bm25_encoder.encode_queries(query)
        hdense, hsparse = hybrid_convex_scale(dense, sparse, alpha=self.alpha)

        results = _pinecone_index().query(
            vector=hdense,
            sparse_vector=hsparse,
            top_k=self.k * 2,
            namespace=self.session_id,
            include_metadata=True,
        )

        seen: set = set()
        docs: List[Document] = []
        for m in results.get("matches", []):
            meta = m.get("metadata", {})
            pid = meta.get("parent_id", m["id"])
            if pid not in seen:
                seen.add(pid)
                clean_meta = {key: val for key, val in meta.items() if key != "parent_text"}
                docs.append(Document(
                    page_content=meta.get("parent_text", meta.get("text", "")),
                    metadata=clean_meta,
                ))
            if len(docs) >= self.k:
                break

        log.info(
            "ParentFetchingHybridRetriever",
            query_preview=query[:80],
            returned=len(docs),
            session_id=self.session_id,
        )
        return docs


# ── Conversational RAG ────────────────────────────────────────────────────────

class ConversationalRAG:
    """
    LCEL-based Conversational RAG using Pinecone as the vector store.

    Pipeline:
        question rewriter
          → MultiQueryRetriever (3 query variants)
          → ParentFetchingHybridRetriever (dense + BM25 sparse → parent chunks)
          → ContextualCompressionRetriever (FlashrankRerank, local CPU, no LLM call)
          → LLM answer
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
        embeddings=None,
        **_kwargs,
    ):
        """
        Build a hybrid retriever from a Pinecone namespace.

        Args:
            session_id: Pinecone namespace (isolates per-user docs)
            k:          Number of parent documents to return after deduplication
            embeddings: Pre-loaded embeddings (avoids reloading the model)
        """
        try:
            if embeddings is None:
                embeddings = ModelLoader().load_embeddings(task_type="retrieval_query")

            alpha = _cfg["reranker"]["alpha"]
            self.retriever = ParentFetchingHybridRetriever(
                embeddings=embeddings,
                session_id=session_id,
                k=k,
                alpha=alpha,
            )
            self._build_lcel_chain()
            log.info(
                "Hybrid Pinecone retriever loaded",
                session_id=session_id,
                k=k,
                alpha=alpha,
            )
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
            #    deduplicates → better recall on vague or ambiguous questions.
            #    Each variant is sent through ParentFetchingHybridRetriever.
            multi_query_retriever = MultiQueryRetriever.from_llm(
                retriever=self.retriever,
                llm=self.llm,
            )

            # 3) FlashrankRerank — local CPU re-ranker (ms-marco-MultiBERT-L-12).
            #    Re-ranks the merged candidate parent chunks and keeps top_n.
            #    Replaces LLMChainExtractor: no extra LLM API call, ~3× cheaper.
            compressed_retriever = ContextualCompressionRetriever(
                base_compressor=_get_reranker(),
                base_retriever=multi_query_retriever,
            )

            # 4) Full chain: rewrite → hybrid retrieve → rerank → answer
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
