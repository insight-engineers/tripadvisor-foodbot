# TripAdvisor Foodbot

## Overview
TripAdvisor Foodbot is an advanced AI-powered chatbot designed to revolutionize the way users discover dining spots in Vietnamese cities like Ho Chi Minh City and Hanoi. By combining state-of-the-art language models, vector search, and comprehensive restaurant data, the chatbot delivers highly personalized restaurant recommendations through an intuitive conversational interface.

## Key Features
- **Intelligent Restaurant Discovery**: Advanced search and recommendation system powered by LlamaIndex and OpenAI
- **Context-Aware Conversations**: Maintains chat history for more relevant and personalized recommendations
- **Real-time Data Processing**: Utilizes Google BigQuery for efficient large-scale data operations
- **Vector-Based Search**: Implements Qdrant for fast and accurate similarity search of restaurant data
- **User-Friendly Interface**: Built with Streamlit for a seamless chat experience
- **Multi-criteria Recommendations**: Considers cuisine, price range, location, and user preferences

## Architecture
The application is built on a modern tech stack:
- **Frontend**: Streamlit for an interactive web interface
- **Language Model**: OpenAI for natural language understanding
- **Vector Database**: Qdrant for efficient similarity search
- **Data Storage**: Google BigQuery for large-scale data management
- **Embedding**: FastEmbed for high-performance text embedding
- **Agent Framework**: LlamaIndex for structured conversation management

## Setup and Development Guide

### Prerequisites
- Python 3.11+ with `uv` package manager
- [Google Cloud Platform](https://cloud.google.com/) account with BigQuery enabled
- [OpenAI API Key](https://openai.com/) for language model access
- [Qdrant Cloud](https://qdrant.tech/) account for vector search capabilities

### Development Environment Setup

1. **Verify Python Environment:**
   ```bash
   python --version  # Should be 3.11 or higher
   uv --version     # Ensure uv is installed
   ```

2. **Install Dependencies:**
   ```bash
   uv sync          # Installs all production dependencies
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_API_URL=your_qdrant_api_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

   Put the service account key file `sa.json` in the root directory of the project.

4. **Launch Development Server:**
   ```bash
   uv run streamlit run app.py
   ```

## Project Dependencies
### Core Dependencies
- `llama-index>=0.12.30`: Agent framework and conversation management
- `fastembed>=0.4.2`: Efficient text embedding generation
- `qdrant-client>=1.12.1`: Vector database client
- `google-cloud-bigquery>=3.30.0`: BigQuery data operations
- `streamlit>=1.45.1`: Web interface framework
- `pandas>=2.2.3`: Data manipulation and analysis
- `scikit-learn>=1.6.1`: Machine learning utilities

### Development Dependencies
- `black>=25.1.0`: Code formatting
- `isort>=6.0.1`: Import sorting
- `ruff>=0.11.5`: Fast Python linter

## Features in Detail
1. **Conversational Interface**
   - Natural language understanding
   - Context-aware responses
   - Streaming response generation
   - Chat history management

2. **Restaurant Recommendations**
   - Personalized suggestions based on user preferences
   - Location-aware recommendations
   - Multi-criteria filtering
   - Real-time data updates

3. **Data Processing**
   - Efficient vector search for similar restaurants
   - Large-scale data querying
   - Real-time data enrichment
   - Contextual information retrieval

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
