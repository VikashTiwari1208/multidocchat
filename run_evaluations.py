#!/usr/bin/env python3
"""
LangSmith Evaluation Script for MultiDocChat RAG System.

Expects documents to already be indexed in Pinecone (via the normal
upload → SQS → worker pipeline). Pass the session_id (Pinecone namespace)
of the indexed document as --session-id.

Usage:
    python run_evaluations.py --dataset AgenticAIReportGoldens --session-id session_20260412_123456_abcd1234
    python run_evaluations.py --dataset AgenticAIReportGoldens --session-id <id> --evaluator all
"""

import os
import sys
import argparse
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from langsmith import Client
from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG


# ── RAG function called per evaluation example ────────────────────────────────

def make_rag_fn(session_id: str, k: int = 5):
    """
    Returns a function that answers a question using the RAG pipeline
    against the Pinecone namespace for the given session_id.
    """
    def answer_question(inputs: dict) -> dict:
        question = inputs.get("question", "").strip()
        if not question:
            return {"answer": "No question provided"}
        try:
            rag = ConversationalRAG(session_id=session_id)
            rag.load_retriever_from_pinecone(session_id=session_id, k=k)
            answer = rag.invoke(question, chat_history=[])
            return {"answer": answer}
        except Exception as e:
            return {"answer": f"Error: {str(e)}"}

    return answer_question


# ── LLM-as-a-Judge correctness evaluator ─────────────────────────────────────

def correctness_evaluator(run: Run, example: Example) -> dict:
    """
    Judges whether the actual answer matches the expected answer
    in terms of factual accuracy and semantic coverage.
    Score: 1 = CORRECT, 0 = INCORRECT.
    """
    actual   = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an evaluator judging correctness of RAG-generated answers.\n\n"
            "Correctness means the actual output matches the reference output in terms of "
            "factual accuracy and semantic coverage. Stylistic differences are acceptable.\n\n"
            "Respond strictly in this format:\n"
            "Reasoning: <1-2 sentences>\n"
            "Verdict: CORRECT or INCORRECT"
        )),
        ("human", (
            "Question: {question}\n\n"
            "Expected answer: {expected}\n\n"
            "Actual answer: {actual}"
        )),
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    try:
        response = (eval_prompt | llm).invoke({
            "question": question,
            "expected": expected,
            "actual": actual,
        })
        text = response.content
        reasoning, verdict = "", ""
        for line in text.split("\n"):
            if line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif line.startswith("Verdict:"):
                verdict = line.replace("Verdict:", "").strip()

        return {
            "key": "correctness",
            "score": 1 if "CORRECT" in verdict.upper() else 0,
            "reasoning": reasoning,
            "comment": f"Verdict: {verdict}",
        }
    except Exception as e:
        return {"key": "correctness", "score": 0, "reasoning": f"Evaluator error: {e}"}


# ── Main evaluation runner ────────────────────────────────────────────────────

def run_evaluation(
    dataset_name: str,
    session_id: str,
    evaluator_type: str = "correctness",
    experiment_prefix: Optional[str] = None,
    description: Optional[str] = None,
    k: int = 5,
):
    print(f"\n{'='*70}")
    print(f"Dataset:    {dataset_name}")
    print(f"Session ID: {session_id}  (Pinecone namespace)")
    print(f"Evaluator:  {evaluator_type}")
    print(f"{'='*70}\n")

    if evaluator_type == "correctness":
        evaluators  = [correctness_evaluator]
        exp_prefix  = experiment_prefix or "multidocchat-correctness"
        desc        = description or "LLM-as-a-Judge correctness evaluation"
    elif evaluator_type == "cot_qa":
        evaluators  = [LangChainStringEvaluator("cot_qa")]
        exp_prefix  = experiment_prefix or "multidocchat-cot-qa"
        desc        = description or "Chain-of-Thought QA evaluation"
    elif evaluator_type == "all":
        evaluators  = [correctness_evaluator, LangChainStringEvaluator("cot_qa")]
        exp_prefix  = experiment_prefix or "multidocchat-multi-eval"
        desc        = description or "Correctness + CoT QA evaluation"
    else:
        print(f"Unknown evaluator '{evaluator_type}'. Choose: correctness | cot_qa | all")
        return None

    metadata = {
        "session_id": session_id,
        "vector_store": "pinecone",
        "retrieval": "MultiQueryRetriever + ContextualCompression + MMR",
        "embedding_model": "models/gemini-embedding-001",
        "llm": "groq/openai/gpt-oss-20b",
        "evaluator": evaluator_type,
        "k": k,
    }

    results = evaluate(
        make_rag_fn(session_id=session_id, k=k),
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=exp_prefix,
        description=desc,
        metadata=metadata,
    )

    print("\nEvaluation complete. View results at https://smith.langchain.com/")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run LangSmith evaluations on the MultiDocChat RAG pipeline",
        epilog="""
Examples:
  python run_evaluations.py --dataset AgenticAIReportGoldens --session-id session_20260412_abc123
  python run_evaluations.py --dataset AgenticAIReportGoldens --session-id <id> --evaluator all --k 10
        """,
    )
    parser.add_argument("--dataset",           default="AgenticAIReportGoldens", help="LangSmith dataset name")
    parser.add_argument("--session-id",        required=True,  help="Pinecone namespace (session_id) of indexed document")
    parser.add_argument("--evaluator",         default="correctness", choices=["correctness", "cot_qa", "all"])
    parser.add_argument("--experiment-prefix", default=None,   help="Custom experiment prefix")
    parser.add_argument("--description",       default=None,   help="Experiment description")
    parser.add_argument("--k",                 default=5, type=int, help="Number of docs to retrieve (default: 5)")
    args = parser.parse_args()

    missing = [v for v in ["LANGSMITH_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "PINECONE_API_KEY"] if not os.getenv(v)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    try:
        run_evaluation(
            dataset_name=args.dataset,
            session_id=args.session_id,
            evaluator_type=args.evaluator,
            experiment_prefix=args.experiment_prefix,
            description=args.description,
            k=args.k,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
