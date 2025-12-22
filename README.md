# Conversational_Rag

A Retrieval-Augmented Generation (RAG) conversational assistant implemented in Python. This repository contains code and utilities to build, run, and evaluate a conversational RAG system that combines vector-based retrieval over documents with a generative language model to produce context-aware answers.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Prepare Data](#prepare-data)
  - [Build Vector Store](#build-vector-store)
  - [Run the Conversational Agent](#run-the-conversational-agent)
- [Configuration](#configuration)
- [Supported Backends](#supported-backends)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

Conversational_Rag is a Python toolkit for building conversational agents that answer user queries by retrieving relevant documents from a knowledge base and using a generative model to produce fluent, context-aware responses. Typical use-cases include customer support bots, knowledge base assistants, and research helpers.

## Features

- Document ingestion and chunking
- Embedding generation and vector store indexing
- Context retrieval with configurable similarity metrics
- Integration points for popular LLMs (local or API-based)
- Simple conversational loop for multi-turn interactions
- Utilities for evaluation and debugging

## Repository Structure

Note: the exact paths and filenames may differ; update the README if you rename files.

- data/                 — Example documents and dataset loaders
- docs/                 — Documentation and design notes
- src/                  — Main source code (vector store, retriever, agent)
- notebooks/            — Example notebooks for exploration and evaluation
- tests/                — Unit and integration tests
- scripts/              — Convenience scripts (ingest, run, eval)
- requirements.txt      — Python dependencies
- README.md             — This file

## Installation

1. Clone the repository

   git clone https://github.com/bibas06/Conversational_Rag.git
   cd Conversational_Rag

2. Create a virtual environment and install dependencies

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate     # Windows (PowerShell)

   pip install --upgrade pip
   pip install -r requirements.txt

3. Optional: install extras for selected backends (e.g., FAISS, Milvus)

## Usage

High-level steps to run a conversational RAG agent:

1. Prepare documents and put them in `data/` (plain text, markdown, or PDF after extraction).
2. Run the ingestion script to chunk documents and compute embeddings.
3. Build or update the vector store.
4. Start the conversational agent and interact.

### Prepare Data

Place raw documents under `data/raw/`. Use provided ingestion scripts to convert PDFs and other formats into plain text and to chunk long documents into passages.

Example:

   python scripts/ingest.py --input data/raw --output data/chunks --chunk-size 500

### Build Vector Store

Example (FAISS):

   python scripts/build_index.py --chunks data/chunks --index data/faiss_index --embeddings openai

Replace `openai` with your chosen embedding provider.

### Run the Conversational Agent

Start a simple terminal-based chat:

   python scripts/chat.py --index data/faiss_index --model openai-gpt --max-context 5

Options vary depending on available models and backends. Check `--help` for each script.

## Configuration

Configuration is managed via environment variables and/or a config file (examples/config.yaml). Typical settings:

- EMBEDDING_PROVIDER — which embeddings to use (openai, sentence-transformers, etc.)
- LLM_PROVIDER — which model to call for generation
- VECTOR_BACKEND — faiss | milvus | ann | elastic
- INDEX_PATH — where to store the vector index

Keep secrets out of the repo; use environment variables or a secrets manager for API keys.

## Supported Backends

The codebase is backend-agnostic; implementations or adapters may be provided for:

- FAISS
- Milvus
- ElasticSearch / OpenSearch
- SQLite + HNSW

If you add a new backend, please follow the adapter interfaces in `src/retriever`.

## Examples

See `notebooks/` for example workflows that:

- ingest and index a small corpus
- run retrieval-only experiments
- run full RAG responses and compare generations

## Testing

Run tests with pytest:

   pytest -q

Add tests for new features and CI integration.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repo
2. Create a feature branch
3. Add tests and update documentation
4. Open a pull request with a clear description of changes

Please follow the repository's code style and add type hints where appropriate.

## License

This project is provided under the MIT License — see LICENSE for details.

## Acknowledgements

Thanks to the open-source community and the authors of libraries used in this project.

---

If you'd like, I can:
- adapt this README to include concrete examples based on files already in the repository,
- add badges (build, PyPI, license) if you want them, or
- create a contributing guide and CODE_OF_CONDUCT.

Tell me which you'd prefer and I will update the repo accordingly.
