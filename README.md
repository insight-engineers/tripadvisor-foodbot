# TripAdvisor Foodbot

![python](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge) ![package-manager](https://img.shields.io/badge/package_manager-uv-green?style=for-the-badge) ![GitHub last commit](https://img.shields.io/github/last-commit/insight-engineers/tripadvisor-foodbot?style=for-the-badge)

## Overview

Restaurant recommendation system implementing **RAG (Retrieval-Augmented Generation)** architecture with **vector search** and **multi-criteria decision analysis (MCDA - ELECTRE III)**. Built on LlamaIndex for agent orchestration, Qdrant for vector storage, and BigQuery for data operations.

## Core Components
- **Vector Search**: Qdrant with FastEmbed for dense retrieval
- **MCDA Engine**: ELECTRE III implementation for ranking restaurants
- **LLM Integration**: OpenAI API with streaming response and generate final natural response
- **Data Layer**: BigQuery for structured data + Qdrant collections
- **Agent Framework**: LlamaIndex with custom tools and callbacks

## Technical Implementation

### Vector Search Pipeline

- FastEmbed for dense embeddings generation
- Qdrant collections for restaurant vectors
- Hybrid search combining semantic and metadata filtering

### MCDA Implementation

- **ELECTRE III** algorithm for restaurant ranking
- Custom concordance/discordance thresholds
- Multi-criteria evaluation:
  - Food quality (delicious, fresh, etc.)
  - Price sensitivity (affordable, expensive, etc.)
  - Ambience (quiet, cozy, etc.)
  - Service (friendly, fast, polite, etc.)
  - Distance to user location (using GPS)
  - Query matching (using `cosine similarity`)
  - Review sentiment (positive, negative, etc.)

> [!IMPORTANT]
> Because **ELECTRE III** is a **decision analysis** algorithm calculated based on lots of matrix operations, it takes a lots of time to rank the restaurants with `numpy`. To improve the performance for better UX, we use `numba` to speed up the ranking process by compiling the `numpy`-based functions with `njit` (`@njit` - this is an alias for `@jit(nopython=True)`). This will make our `python` numpy-based functions run with near-C speed.

### Repository Structure

```bash
src/
├── bigquery/    # BigQuery operations and data handlers
├── chat/        # LlamaIndex agent implementation
├── helper/      # Utility functions
├── qdrant/      # Vector DB operations
├── ranker/      # ELECTRE III implementation
└── s3/          # S3 client for asset storage (just use for user storage)
```

### Agent Implementation - `chat` directory

- LlamaIndex RAG implementation
- Custom tools for:
  - `candidate_generation_and_ranking`: generate candidate restaurants and rank them using MCDA
  - `enrich_restaurant_recommendations`: enrich the restaurant recommendations with more information, generate final natural response
- Streaming response handlers for tools callbacks
- Context management with chat history

## Development Setup

### Requirements

- `python ~= 3.11`
- `uv` package manager
- GCP account with BigQuery: [GCP Console](https://console.cloud.google.com/)
- OpenAI API key: [OpenAI Console](https://platform.openai.com/)
- Qdrant Cloud instance: [Qdrant Console](https://cloud.qdrant.io/)
- AWS S3 (or alternative storage service using S3 API)

### Quick Start

1. Install dependencies with `uv`

```bash
uv sync
```

2. Place GCP service account key and copy `.env.template` to `.env`

```bash
cp sa.json ./
cp .env.template .env
```


> [!NOTE]
> The `sa.json` is the credential file of the GCP service account. If not have, the BigQuery operations will not work.

3. Fill in the values in `.env`

```bash
OPENAI_API_KEY=your-api-key
QDRANT_API_KEY=your-api-key
QDRANT_API_URL=your-api-url
AWS_REGION_NAME=your-region
AWS_BUCKET_NAME=your-bucket-name
AWS_BUCKET_ENDPOINT_URL=your-bucket-endpoint
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
```

4. Start development server

```bash
make dev
```

### Dependencies

Core: `llama-index>=0.12.30`, `fastembed>=0.4.2`, `qdrant-client>=1.12.1`
Data: `google-cloud-bigquery>=3.30.0`, `pandas>=2.2.3`
UI: `streamlit>=1.45.1`
Dev: `black>=25.1.0`, `isort>=6.0.1`, `ruff>=0.11.5`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
