import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.generation.generator import llm
from main import agentic_rag_app

def get_pipeline_data(query: str):
    initial_state = {
        "query": query,
        "original_query": query,
        "namespace": "default_namespace",
        "documents": [],
        "final_context_strips": [],
        "needs_web_search": False,
        "answer": "",
        "retries": 0,
        "route": ""
    }
    result = agentic_rag_app.invoke(initial_state)
    answer = result.get("answer", "")
    context = result.get("final_context_strips", [])
    if not context:
        context = ["LLM General Knowledge/No specific context found."]
    return answer, context

def run():
    queries = [
        "Tell me about the cross-modal architecture of RAG-Anything.",
        "What is the capital of France?", # Direct LLM route
        "What is the specific dual-graph construction technique used? Explain step-by-step."
    ]

    data = {"question": [], "answer": [], "contexts": []}
    for q in queries:
        print(f"Evaluating query: {q}")
        ans, ctx = get_pipeline_data(q)
        data["question"].append(q)
        data["answer"].append(ans)
        data["contexts"].append(ctx)

    dataset = Dataset.from_dict(data)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

    # Assign explicitly
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_emb

    print("Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    df = results.to_pandas()
    out_path = PROJECT_ROOT / "ragas_metrics.csv"
    df.to_csv(out_path, index=False)
    print("\nEvaluation Completed! Scores saved to ragas_metrics.csv")
    print(df[["question", "faithfulness", "answer_relevancy"]])

if __name__ == "__main__":
    run()
