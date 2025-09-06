import os
import ast
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

# -----------------------------
# LLM & RAGAS Imports
# -----------------------------
from langchain_ollama import OllamaLLM
# from langchain.chat_models import ChatOpenAI   # Uncomment if using GPT-4
from langchain_core.embeddings import Embeddings
from ragas.run_config import RunConfig

load_dotenv()

# -----------------------------
# Database connection
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    df = pd.read_sql(
        text("""
            SELECT l.id,
                   l.question,
                   l.answer,
                   l.contexts,
                   g.ground_truth
            FROM rag_evaluation_logs l
            JOIN ground_truth g
              ON l.question = g.question
        """),
        conn
    )

# -----------------------------
# Clean contexts
# -----------------------------
def clean_contexts(ctx):
    if isinstance(ctx, str):
        ctx = ast.literal_eval(ctx)
    return [c["chunk"] if isinstance(c, dict) and "chunk" in c else str(c) for c in ctx]

df["contexts"] = df["contexts"].apply(clean_contexts)

# -----------------------------
# Reference in context check
# -----------------------------
df["reference_in_context"] = df.apply(
    lambda row: any(row["ground_truth"].lower() in c.lower() for c in row["contexts"]),
    axis=1
)

# -----------------------------
# SentenceTransformer embeddings wrapper
# -----------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

embeddings = SentenceTransformerEmbeddings()

# -----------------------------
# LLM Selection
# -----------------------------
# Option 1: Ollama local LLaMA 3
ollama_llm = OllamaLLM(model="llama3:latest", timeout=300, keep_alive=300)

# Option 2: GPT-4 via OpenAI
# gpt4_llm = ChatOpenAI(model_name="gpt-4", temperature=0.0, max_retries=2)

# -----------------------------
# Custom Answer Relevancy Metric
# -----------------------------
def answer_relevancy_metric(dataset, embeddings):
    scores = []
    for a, ref in zip(dataset["answer"], dataset["reference"]):
        a_emb = np.array(embeddings.embed_query(a))
        ref_emb = np.array(embeddings.embed_query(ref))
        score = np.dot(a_emb, ref_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(ref_emb))
        scores.append((score + 1) / 2)  # scale [-1,1] → [0,1]
    return scores

# -----------------------------
# Convert DB DataFrame to RAGAS Dataset
# -----------------------------
dataset = Dataset.from_dict({
    "question": df["question"].tolist(),
    "answer": df["answer"].tolist(),
    "contexts": df["contexts"].tolist(),
    "reference": df["ground_truth"].tolist()
})

# -----------------------------
# RunConfig for stable execution, number of workers
# -----------------------------
run_config = RunConfig(
    max_workers=2,      # control parallelism
    timeout=300,        # max seconds per LLM request
    max_retries=2       # retry failed requests
)

# -----------------------------
# Run RAGAS evaluation
# -----------------------------
eval_result = evaluate(
    dataset,
    metrics=[faithfulness, context_precision, context_recall],
    llm=ollama_llm,      # switch to gpt4_llm if using GPT-4
    embeddings=embeddings,
    run_config=run_config,
    raise_exceptions=False
)

# -----------------------------
# Results DataFrame
# -----------------------------
df_results = eval_result.to_pandas()
df_results["answer_relevancy"] = answer_relevancy_metric(dataset, embeddings)

df_results["question"] = dataset["question"]
df_results["ground_truth"] = dataset["reference"]
df_results["model_answer"] = dataset["answer"]
df_results["contexts"] = dataset["contexts"]
df_results["reference_in_context"] = df["reference_in_context"]

# -----------------------------
# Save results
# -----------------------------
df_results.to_csv("ragas_per_question_metrics.csv", index=False)
df_results.to_excel("ragas_per_question_metrics.xlsx", index=False)


print("\n✅ Metrics saved to ragas_per_question_metrics.csv & .xlsx")
