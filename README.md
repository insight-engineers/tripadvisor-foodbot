# 🍽️ TripAdvisor Foodbot

![python](https://img.shields.io/badge/python-3.11-blue?style=for-the-badge) ![package-manager](https://img.shields.io/badge/package_manager-uv-green?style=for-the-badge) ![GitHub last commit](https://img.shields.io/github/last-commit/insight-engineers/tripadvisor-foodbot?style=for-the-badge)

## 📝 Overview

> [!NOTE]
> 📦 Data for this project is sourced from repositories in the [`include`](include/) directory. It is scraped, processed, and cleaned using multiple modules. The `dm_tripadvisor` and `fs_tripadvisor` datasets are publicly available in BigQuery for authenticated users and power the restaurant recommendation system.

A next-generation restaurant recommendation system implementing **RAG (Retrieval-Augmented Generation)** architecture with **vector search** and **multi-criteria decision analysis (MCDA - ELECTRE III)**. Built on LlamaIndex for agent orchestration, Qdrant for vector storage, and BigQuery for data operations.

---

## 🚀 Core Components

- 🔍 **Vector Search**: Qdrant with FastEmbed for dense retrieval
- 🏆 **MCDA Engine**: ELECTRE III implementation for ranking restaurants
- 🤖 **LLM Integration**: OpenAI API with streaming response to generate final natural responses
- 🗄️ **Data Layer**: BigQuery for structured data + Qdrant collections
- 🧑‍💻 **Agent Framework**: LlamaIndex with custom tools and callbacks

---

## ⚙️ Technical Implementation

### 🧬 Vector Search Pipeline

- ⚡ FastEmbed for dense embeddings generation
- 🗃️ Qdrant collections for restaurant vectors
- 🔗 Hybrid search combining semantic and metadata filtering

### 🏅 MCDA Implementation

- 🧮 **ELECTRE III** algorithm for restaurant ranking
- 🎚️ Custom concordance/discordance thresholds
- 📊 Multi-criteria evaluation:
  - 🍲 Food quality (delicious, fresh, etc.)
  - 💸 Price sensitivity (affordable, expensive, etc.)
  - 🛋️ Ambience (quiet, cozy, etc.)
  - 🧑‍🍳 Service (friendly, fast, polite, etc.)
  - 📍 Distance to user location (using GPS)
  - 🔎 Query matching (using `cosine similarity`)
  - 😊 Review sentiment (positive, negative, etc.)

> [!IMPORTANT]
> ⚡ Because **ELECTRE III** is a **decision analysis** algorithm calculated based on lots of matrix operations, it can take significant time to rank restaurants with `numpy`. To improve performance and user experience, we use `numba` to speed up the ranking process by compiling the `numpy`-based functions with `njit` (`@njit` - alias for `@jit(nopython=True)`). This enables our Python functions to run at near-C speed.

---

### 📁 Repository Structure

```bash
src/
├── bigquery/    # BigQuery operations and data handlers
├── chat/        # LlamaIndex agent implementation
├── helper/      # Utility functions
├── qdrant/      # Vector DB operations
├── ranker/      # ELECTRE III implementation
└── s3/          # S3 client for asset storage (user storage)
```

---

### 🤖 Agent Implementation - `chat` Directory

- LlamaIndex RAG implementation
- Custom tools for:
  - 🥇 `candidate_generation_and_ranking`: generate candidate restaurants and rank them using MCDA
  - 📝 `enrich_restaurant_recommendations`: enrich recommendations with more information and generate the final natural response
- 🔄 Streaming response handlers for tool callbacks
- 🧠 Context management with chat history

---

## 🛠️ Development Setup

### 📋 Requirements

- 🐍 `python ~= 3.11`
- 📦 `uv` package manager
- ☁️ GCP account with BigQuery: [GCP Console](https://console.cloud.google.com/)
- 🔑 OpenAI API key: [OpenAI Console](https://platform.openai.com/)
- 🟣 Qdrant Cloud instance: [Qdrant Console](https://cloud.qdrant.io/)
- 🗄️ AWS S3 (or alternative storage service using S3 API)

---

### ⚡ Quick Start

1. **Install dependencies with `uv`**

    ```bash
    uv sync
    ```

2. **Place GCP service account key and copy `.env.template` to `.env`**

    ```bash
    cp sa.json ./
    cp .env.template .env
    ```

    > [!NOTE]
    > 🗝️ The `sa.json` is the credential file for the GCP service account. Without it, BigQuery operations will not work.

3. **Prepare `.env` file**

    3.1 🔐 Generate a secret key for `chainlit` authentication (Optional, only required if you use `chainlit` UI)

    ```bash
    chainlit create-secret
    ```

    3.2 🛡️ Prepare your GitHub OAuth credentials (Optional, only required if you use `chainlit` UI)

    > [Creating an OAuth app - GitHub Docs](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/creating-an-oauth-app)

    3.3 📝 Fill in the `.env` file with your credentials and configuration

    ```bash
    OPENAI_API_KEY=your-api-key
    QDRANT_API_KEY=your-api-key
    QDRANT_API_URL=your-api-url
    AWS_REGION_NAME=your-region
    AWS_BUCKET_NAME=your-bucket-name
    AWS_BUCKET_ENDPOINT_URL=your-bucket-endpoint
    AWS_ACCESS_KEY_ID=your-access-key-id
    AWS_SECRET_ACCESS_KEY=your-secret-access-key
    OAUTH_GITHUB_CLIENT_ID=your-github-client-id
    OAUTH_GITHUB_CLIENT_SECRET=your-github-client-secret
    CHAINLIT_AUTH_SECRET=generated-secret-key
    ```

4. **Start development server**

    - For `chainlit` UI:
      ```bash
      make cl
      ```
    - For `streamlit` UI:
      ```bash
      make st
      ```

    > [!NOTE] ⚠️ DEPRECATION
    > Streamlit UI is not fully implemented yet, but you can use it to test the agent and see the results.

---

### 📦 Dependencies

This project uses `uv` as the package manager. Dependencies are managed in the `uv` configuration file (`uv.yaml`).

- Core: `llama-index`, `fastembed`, `qdrant-client`
- Data: `google-cloud-bigquery`, `pandas`
- UI: `streamlit`, `chainlit`

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
