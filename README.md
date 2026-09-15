# Thesis RAG and Model Evaluation on Databricks

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-Serverless-FF3621?logo=databricks&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![License](https://img.shields.io/badge/License-MIT-green)

This repository presents the code developed for a thesis comparing language-model
approaches across two question-answering datasets:

- **QASPER**, a question-answering dataset based on scientific papers.
- **Government QA**, a question-answering collection derived from government reports.

The project has been reorganized as a public portfolio demonstration of reproducible
research, GitHub version control, and Databricks-based data and machine-learning
workflows.

**Portfolio highlights:** an end-to-end LangChain RAG pipeline, governed storage in
Unity Catalog, reproducible two-stage Databricks Workflow orchestration, MLflow
tracking, and deterministic evaluation against public reference annotations.

> The original research notebooks were developed in Google Colab. This portfolio
> edition removes embedded credentials, clears generated outputs, and uses Databricks
> Unity Catalog Volume paths.

## Thesis

[Read the full thesis (PDF)](paper/GovReport_Thesis_1.1.pdf)

## Research workflow

The experiments compare four model settings:

| Model | Description |
| --- | --- |
| M0 | Base language-model baseline |
| M1 | Retrieval-augmented generation using semantic retrieval and FAISS |
| M2 | LoRA-adapted model without retrieved context |
| M3 | Retrieval-augmented generation combined with LoRA adaptation |

The complete workflow is:

```text
Public datasets
      ↓
Data preparation and exploration
      ↓
M0 baseline ─ M1 RAG ─ M2 LoRA ─ M3 RAG + LoRA
      ↓
Result consolidation
      ↓
RAGAS, ROUGE, retrieval, and model comparison
```

## Databricks + LangChain portfolio pipeline

The original notebooks remain unchanged as the research record. A separate,
lightweight production-style demonstration is provided in
`notebooks/14_langchain_databricks_demo.py`. A companion notebook,
`notebooks/15_evaluate_langchain_results.py`, evaluates the stored answers without
making additional paid LLM calls. The pipeline uses:

- Hugging Face Datasets to load a small QASPER validation sample;
- LangChain `Document` objects and `RecursiveCharacterTextSplitter`;
- `HuggingFaceEmbeddings` and a LangChain FAISS retriever;
- `ChatOpenAI` for context-grounded generation;
- a Unity Catalog Volume for the persisted vector index;
- a Unity Catalog Delta table for generated answers;
- MLflow for parameters and run metrics;
- deterministic exact-match and token-F1 evaluation against QASPER annotations;
- and a Databricks Declarative Automation Bundle for repeatable deployment as a job.

This demonstration intentionally defaults to 50 papers and answers at most 3
questions. It proves the end-to-end platform workflow without repeating the costly
LoRA training runs from the thesis.

### Architecture

```text
Hugging Face QASPER
        │
        ▼
LangChain document loading and chunking
        │
        ▼
MiniLM embeddings ──► FAISS index in a Unity Catalog Volume
        │
        ▼
Retriever + prompt ──► GPT-4.1 mini
        │
        ▼
Generated answers ──► Delta table ──► deterministic evaluation
        │                                  │
        └──────────────────────────────────┴──► MLflow tracking
```

The Databricks Workflow contains two ordered tasks:

```text
langchain_rag_demo ──(on success)──► evaluate_langchain_results
```

No recurring trigger is configured, which prevents unintended compute or API use.

### Verified Databricks run

The portfolio smoke test was successfully executed on **15 September 2026**.

| Item | Verified value |
| --- | ---: |
| QASPER validation papers | 50 |
| Text chunks indexed | 2,304 |
| Questions answered | 3 |
| Generation model | `gpt-4.1-mini` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Retriever `k` | 4 |
| Mean token F1 | 0.1785 |
| Exact match | 0.0000 |
| Mean answer length | 41 words |
| Additional LLM calls during evaluation | 0 |

The generation run and evaluation run both completed successfully and were logged
to the shared MLflow experiment. Generated answers are stored in
`workspace.default.langchain_rag_demo_results`; per-answer evaluation records are
stored in `workspace.default.langchain_rag_demo_evaluation`.

These figures demonstrate that the deployment and tracking path works; they are not
reported as research conclusions because the smoke test contains only three
questions. Exact match is especially strict for generative answers, while token F1
provides partial credit for overlap with QASPER references.

## Repository structure

```text
.
├── notebooks/
│   ├── 01_qasper_data_download.ipynb
│   ├── 02_government_data_processing.ipynb
│   ├── 03_data_understanding.ipynb
│   ├── 04_m0_baseline.ipynb
│   ├── 05_m1_rag.ipynb
│   ├── 06_m2_government_lora.ipynb
│   ├── 07_m2_qasper_lora.ipynb
│   ├── 08_m3_government_rag_lora.ipynb
│   ├── 09_m3_qasper_rag_lora.ipynb
│   ├── 10_qasper_results_cleaning.ipynb
│   ├── 11_government_results_cleaning.ipynb
│   ├── 12_qasper_evaluation.ipynb
│   ├── 13_government_evaluation.ipynb
│   ├── 14_langchain_databricks_demo.py
│   └── 15_evaluate_langchain_results.py
├── resources/
│   └── thesis_rag_job.yml
├── figures/
├── paper/
├── databricks.yml
├── requirements-demo.txt
├── .env.example
├── .gitignore
└── requirements.txt
```

The `paper/` directory is reserved for the thesis PDF. Add it only if the university
and publisher policies permit public distribution. The `figures/` directory can hold
selected, publication-ready results for the GitHub project page.

## Databricks setup

1. Create or select a Databricks workspace.
2. Configure GitHub credentials under **Settings → Linked accounts**.
3. In **Workspace**, select **Create → Git folder** and enter this repository URL.
4. Create a Unity Catalog Volume for the project data.
5. Either create the example path below or update the paths in the notebooks:

   ```text
   /Volumes/workspace/default/thesis_project
   ```

6. Use Serverless Standard environment v6 for the lightweight LangChain demo. The
   original Mistral/LoRA notebooks require suitable GPU compute.
7. Apply `requirements-demo.txt` to the lightweight demo environment. For the
   original research notebooks, install:

   ```python
   %pip install -r requirements.txt
   dbutils.library.restartPython()
   ```

8. Store credentials as Databricks secrets. Do not paste credentials into notebooks.

The portfolio demo uses the governed Unity Catalog secret created in Catalog
Explorer:

```python
import os

os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(
    catalog="workspace",
    schema="default",
    key="openai_api_key",
)
```

For local development, copy `.env.example` to `.env`, add your own credentials, and
keep `.env` untracked.

### Deploy the lightweight demo

Install the current Databricks CLI, authenticate it to your workspace, and run from
the repository root:

```bash
databricks bundle validate
databricks bundle deploy -t dev
databricks bundle run -t dev thesis_rag_demo
```

The included bundle defaults match the portfolio workspace path
`workspace.default.thesis_project`. For another workspace, override the defaults
with a catalog, schema, and volume where you have `USE`, `READ VOLUME`, and `WRITE
VOLUME` privileges. The notebook expects a Unity Catalog secret named
`workspace.default.openai_api_key`; create it in Catalog Explorer and never paste
the value into source code or a notebook cell.

## Data

The research uses two public, CC BY 4.0 datasets:

- [AllenAI QASPER](https://huggingface.co/datasets/allenai/qasper): question
  answering over scientific papers. Notebook `01` reads the official train,
  validation, and test Parquet splits from Hugging Face.
- [LAUNCH GovReport-QS](https://huggingface.co/datasets/launch/gov_report_qs):
  hierarchical question-summary annotations over government reports. Notebook `02`
  converts the source JSONL files into normalized question-answer-evidence tables.

The repository records provenance and preprocessing code without duplicating the
complete datasets. On Databricks, source and processed files are stored under the
example Unity Catalog Volume path:

```text
/Volumes/workspace/default/thesis_project/data/
├── qasper/
└── government_qa/
```

Reusable data, FAISS indexes, model weights, predictions, and evaluation tables are
excluded from Git. For exact reproducibility, pin the Hugging Face dataset revision
used for a final run and record it with the Databricks Runtime, model revision, and
experiment metadata.

## Suggested execution order

1. Run notebooks `01`–`03` to prepare and inspect the datasets.
2. Run notebooks `04`–`09` to reproduce the four model settings.
3. Run notebooks `10` and `11` to consolidate model outputs.
4. Run notebooks `12` and `13` to calculate evaluation metrics.
5. Run notebook `14` for the lightweight Databricks + LangChain demonstration.
6. Run notebook `15` to evaluate its stored answers without additional LLM calls.

Some model notebooks require substantial GPU memory. Compute type, model revision,
random seed, and package versions should be recorded alongside published results.

## Technology demonstrated

- Python and Jupyter notebooks
- Databricks Workspace and Git folders
- Unity Catalog Volumes
- Hugging Face Datasets and Transformers
- PEFT/LoRA fine-tuning
- Retrieval-augmented generation and FAISS
- MLflow experiment tracking
- Databricks Workflows and Declarative Automation Bundles
- RAGAS, ROUGE, exact-match, and token-F1 evaluation
- Pandas, PyArrow, and reproducible experiment organization

## Security

The public notebooks read tokens from `HF_TOKEN` and `OPENAI_API_KEY`. Never commit
real credentials. Credentials used in the original research environment should be
revoked before this repository is published.

## License

The project code is released under the [MIT License](LICENSE). The thesis PDF
remains copyrighted by its author and is included for portfolio and scholarly
review; the MIT License does not apply to the thesis text or third-party datasets,
models, or other materials.
