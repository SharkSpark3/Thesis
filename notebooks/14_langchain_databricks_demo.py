# Databricks notebook source
# MAGIC %md
# MAGIC # Thesis portfolio demo: Databricks + LangChain
# MAGIC
# MAGIC This lightweight pipeline downloads a small public QASPER sample, creates
# MAGIC LangChain documents and chunks, builds a FAISS retriever, generates answers,
# MAGIC writes governed outputs, and logs the run to MLflow.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("volume", "thesis_project")
dbutils.widgets.text("sample_size", "50")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
VOLUME = dbutils.widgets.get("volume")
SAMPLE_SIZE = int(dbutils.widgets.get("sample_size"))

VOLUME_ROOT = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
INDEX_PATH = f"{VOLUME_ROOT}/indexes/qasper_langchain_faiss"
OUTPUT_TABLE = f"{CATALOG}.{SCHEMA}.langchain_rag_demo_results"

# COMMAND ----------

# The portfolio workspace already contains this managed Volume. Fail early with a
# clear message when the bundle variables point to a path that does not exist.
assert any(file.name.rstrip("/") == VOLUME for file in dbutils.fs.ls(f"/Volumes/{CATALOG}/{SCHEMA}")), (
    f"Create the Unity Catalog Volume first: {VOLUME_ROOT}"
)

# COMMAND ----------

import os
from operator import itemgetter

import mlflow
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The secret scope/key must be created once in the Databricks workspace.
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(
    scope="thesis", key="openai-api-key"
)

# COMMAND ----------

dataset = load_dataset("allenai/qasper", split=f"validation[:{SAMPLE_SIZE}]")

documents = []
questions = []
for row_number, row in enumerate(dataset):
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    full_text = row.get("full_text", {})
    sections = full_text.get("section_name", [])
    paragraphs = full_text.get("paragraphs", [])
    body = "\n\n".join(
        f"{section}\n{' '.join(section_paragraphs)}"
        for section, section_paragraphs in zip(sections, paragraphs)
    )
    documents.append(
        Document(
            page_content=f"{title}\n\n{abstract}\n\n{body}",
            metadata={"paper_id": row.get("id", str(row_number)), "title": title},
        )
    )
    for question in row.get("qas", {}).get("question", [])[:1]:
        questions.append({"paper_id": row.get("id", str(row_number)), "question": question})

splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
chunks = splitter.split_documents(documents)

# COMMAND ----------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local(INDEX_PATH)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only from the supplied context. If the answer is absent, say that the context is insufficient.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# COMMAND ----------

mlflow.set_experiment("/Shared/thesis-langchain-rag-demo")
results = []
with mlflow.start_run(run_name="qasper-small-sample"):
    mlflow.log_params(
        {
            "dataset": "allenai/qasper",
            "split": "validation",
            "sample_size": SAMPLE_SIZE,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm": "gpt-4.1-mini",
            "retriever_k": 4,
            "chunk_size": 900,
            "chunk_overlap": 120,
        }
    )
    for item in questions[:10]:
        answer = rag_chain.invoke({"question": item["question"]})
        results.append({**item, "answer": answer})
    mlflow.log_metric("documents", len(documents))
    mlflow.log_metric("chunks", len(chunks))
    mlflow.log_metric("questions_answered", len(results))

result_df = spark.createDataFrame(results)
result_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUTPUT_TABLE)
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Portfolio evidence
# MAGIC
# MAGIC After this notebook succeeds, capture:
# MAGIC 1. the Workflow run graph;
# MAGIC 2. this results table;
# MAGIC 3. the MLflow run parameters and metrics;
# MAGIC 4. the Unity Catalog table and Volume paths.
