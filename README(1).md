# RAG Prompt and Retrieval Optimization

A retrieval-augmented generation (RAG) question-answering pipeline combining Pyserini/Lucene retrieval with Llama 3.1 8B Instruct generation. The project evaluates how prompt constraints and retrieval depth affect Exact Match (EM) and token-level F1 on a 200-question HotpotQA development subset.

## Highlights

- Retrieved Wikipedia passages with Pyserini and LuceneSearcher
- Generated answers with `meta-llama/Llama-3.1-8B-Instruct`
- Compared baseline, short-answer, exact-span, and best-guess prompts
- Evaluated retrieval depths `k = 2, 4, 8`
- Measured performance with Exact Match and token-level F1
- Saved predictions and experiment results for reproducibility

## Results

### Prompt comparison (`k = 4`)

| Prompt strategy | EM | F1 |
|---|---:|---:|
| Baseline | 0.0350 | 0.1146 |
| Short answer | 0.0650 | 0.1431 |
| Exact span | 0.0350 | 0.1031 |
| Best guess | 0.0650 | 0.1270 |

### Retrieval-depth comparison

| Retrieval depth | EM | F1 |
|---|---:|---:|
| `k = 2` | 0.2300 | 0.3519 |
| `k = 4` | 0.0650 | 0.1431 |
| `k = 8` | 0.0100 | 0.0262 |

The strongest configuration used the short-answer prompt with `k = 2`, suggesting that additional retrieved context introduced noise for this setup.

## Recommended Repository Structure

```text
.
├── rag_prompt_optimization.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── results/
└── figures/
```

Do not commit the Wikipedia Lucene index, model weights, Hugging Face cache, datasets without permission, or private access tokens.

## Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pyserini also requires a compatible Java installation.

## Authentication

Llama 3.1 is a gated Hugging Face model. Authenticate outside the notebook:

```bash
huggingface-cli login
```

Or set an environment variable:

```bash
export HF_TOKEN="your_token"
```

Never place a token directly in source code or commit it to Git.

## Data and Index Configuration

Place the HotpotQA development split at:

```text
data/dev.jsonl
```

Set the Lucene index path through an environment variable:

```bash
export WIKI_INDEX_PATH="/path/to/wiki_index"
```

Use relative paths or environment variables in code:

```python
import os

DEV_PATH = os.getenv("DEV_PATH", "data/dev.jsonl")
WIKI_INDEX_PATH = os.environ["WIKI_INDEX_PATH"]
```

## Running

```bash
jupyter lab rag_prompt_optimization.ipynb
```

Run all cells from top to bottom. Reduce the subset size for a quick smoke test before running all 200 examples.

## Cleanup Before Publishing

- Remove embedded Hugging Face tokens and revoke exposed tokens
- Replace absolute Purdue paths with environment variables or relative paths
- Delete duplicate definitions of `retrieve_context` and `build_prompt`
- Clear notebook outputs before committing
- Add checks for missing data, indexes, GPUs, and model access
- Save summary results as CSV or JSON
- Document the hardware and software versions used

## Technologies

Python, PyTorch, Hugging Face Transformers, Pyserini, Apache Lucene, Llama 3.1, HotpotQA, Jupyter, Matplotlib

## Limitations

The experiments use a 200-example development subset and one model, so the results are exploratory rather than a general RAG benchmark.

## Author

Kriti Tamirasa
