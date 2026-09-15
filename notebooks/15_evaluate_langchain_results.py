# Databricks notebook source
# MAGIC %md
# MAGIC # Lightweight evaluation of the LangChain RAG demo
# MAGIC
# MAGIC This notebook evaluates the generated answers against QASPER annotations with
# MAGIC deterministic exact-match and token-F1 metrics. It makes no additional LLM calls.

# COMMAND ----------

import re
import string
from collections import Counter

import mlflow
from datasets import load_dataset

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("sample_size", "50")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SAMPLE_SIZE = int(dbutils.widgets.get("sample_size"))

RESULTS_TABLE = f"{CATALOG}.{SCHEMA}.langchain_rag_demo_results"
EVALUATION_TABLE = f"{CATALOG}.{SCHEMA}.langchain_rag_demo_evaluation"

# COMMAND ----------

generated_rows = [row.asDict() for row in spark.table(RESULTS_TABLE).collect()]
assert generated_rows, f"No generated answers found in {RESULTS_TABLE}"

data_files = {
    "validation": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/validation/0000.parquet"
}
dataset = load_dataset(
    "parquet",
    data_files=data_files,
    split=f"validation[:{SAMPLE_SIZE}]",
)

# COMMAND ----------

def normalize_text(value):
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction, reference):
    predicted = normalize_text(prediction).split()
    expected = normalize_text(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def answer_texts(value):
    """Extract textual answer candidates from nested QASPER answer records."""
    candidates = []
    if isinstance(value, list):
        for item in value:
            candidates.extend(answer_texts(item))
    elif isinstance(value, dict):
        if value.get("unanswerable"):
            candidates.append("unanswerable")
        free_form = value.get("free_form_answer")
        if free_form:
            candidates.append(str(free_form))
        yes_no = value.get("yes_no")
        if yes_no is not None:
            candidates.append("yes" if bool(yes_no) else "no")
        for span in value.get("extractive_spans") or []:
            if span:
                candidates.append(str(span))
        for key, nested in value.items():
            if key not in {"unanswerable", "free_form_answer", "yes_no", "extractive_spans"}:
                candidates.extend(answer_texts(nested))
    return list(dict.fromkeys(text for text in candidates if normalize_text(text)))


references = {}
for row in dataset:
    qas = row.get("qas") or {}
    for question, answers in zip(qas.get("question") or [], qas.get("answers") or []):
        references[(str(row.get("id")), str(question))] = answer_texts(answers)

# COMMAND ----------

evaluated = []
for row in generated_rows:
    prediction = str(row["answer"])
    candidates = references.get((str(row["paper_id"]), str(row["question"])), [])
    exact = max(
        (float(normalize_text(prediction) == normalize_text(reference)) for reference in candidates),
        default=0.0,
    )
    best_f1 = max((token_f1(prediction, reference) for reference in candidates), default=0.0)
    evaluated.append(
        {
            "paper_id": str(row["paper_id"]),
            "question": str(row["question"]),
            "answer": prediction,
            "reference_count": int(len(candidates)),
            "exact_match": float(exact),
            "token_f1": float(best_f1),
            "answer_word_count": int(len(re.findall(r"\b\w+\b", prediction))),
        }
    )

evaluation_df = spark.createDataFrame(
    evaluated,
    schema=(
        "paper_id STRING, question STRING, answer STRING, reference_count INT, "
        "exact_match DOUBLE, token_f1 DOUBLE, answer_word_count INT"
    ),
)
evaluation_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    EVALUATION_TABLE
)

summary = evaluation_df.selectExpr(
    "count(*) AS evaluated_answers",
    "avg(exact_match) AS mean_exact_match",
    "avg(token_f1) AS mean_token_f1",
    "avg(answer_word_count) AS mean_answer_word_count",
).first()

mlflow.set_experiment("/Shared/thesis-langchain-rag-demo")
with mlflow.start_run(run_name="qasper-deterministic-evaluation"):
    mlflow.log_params(
        {
            "source_table": RESULTS_TABLE,
            "reference_dataset": "allenai/qasper",
            "split": "validation",
            "sample_size": SAMPLE_SIZE,
            "evaluation": "exact_match_and_token_f1",
            "additional_llm_calls": 0,
        }
    )
    mlflow.log_metrics(
        {
            "evaluated_answers": float(summary["evaluated_answers"]),
            "mean_exact_match": float(summary["mean_exact_match"] or 0.0),
            "mean_token_f1": float(summary["mean_token_f1"] or 0.0),
            "mean_answer_word_count": float(summary["mean_answer_word_count"] or 0.0),
        }
    )

display(evaluation_df)
display(spark.table(EVALUATION_TABLE))

