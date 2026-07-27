Data preparation and exploration
      ↓
M0 baseline ─ M1 RAG ─ M2 LoRA ─ M3 RAG + LoRA
      ↓
Result consolidation
      ↓
RAGAS, ROUGE, retrieval, and model comparison
```

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
│   └── 13_government_evaluation.ipynb
├── figures/
├── paper/
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
   /Volumes/main/default/thesis_project
   ```

6. Attach a GPU-enabled compute resource for the model notebooks.
7. Install dependencies from the repository:

   ```python
   %pip install -r requirements.txt
   dbutils.library.restartPython()
   ```

8. Store credentials as Databricks secrets. Do not paste credentials into notebooks.

Example:

```python
import os

os.environ["HF_TOKEN"] = dbutils.secrets.get(
    scope="thesis",
    key="huggingface-token",
)
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(
    scope="thesis",
    key="openai-api-key",
)
```

For local development, copy `.env.example` to `.env`, add your own credentials, and
keep `.env` untracked.

## Data

The datasets and generated artifacts are deliberately excluded from Git:

- Download public source datasets from their original providers.
- Store reusable data, indexes, predictions, and evaluation tables in a Databricks
  Unity Catalog Volume.
- Record the dataset revision or commit hash used in the final thesis results.
- Do not commit model weights, FAISS indexes, CSV/Parquet outputs, or API credentials.

## Suggested execution order

1. Run notebooks `01`–`03` to prepare and inspect the datasets.
2. Run notebooks `04`–`09` to reproduce the four model settings.
3. Run notebooks `10` and `11` to consolidate model outputs.
4. Run notebooks `12` and `13` to calculate evaluation metrics.

Some model notebooks require substantial GPU memory. Compute type, model revision,
random seed, and package versions should be recorded alongside published results.

## Technology demonstrated

- Python and Jupyter notebooks
- Databricks Workspace and Git folders
- Unity Catalog Volumes
- Hugging Face Datasets and Transformers
- PEFT/LoRA fine-tuning
- Retrieval-augmented generation and FAISS
- RAGAS and ROUGE evaluation
- Pandas, PyArrow, and reproducible experiment organization

## Security

The public notebooks read tokens from `HF_TOKEN` and `OPENAI_API_KEY`. Never commit
real credentials. Credentials used in the original research environment should be
revoked before this repository is published.

## License

The thesis text, code, and figures remain under the rights described in `LICENSE`.
