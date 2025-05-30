.PHONY: dev start test format load_qdrant_locations load_qdrant_reviews
export STREAMLIT_FILE_PATH := _streamlit.py

# development
start:
	uv run streamlit run $(STREAMLIT_FILE_PATH)

dev:
	uv run streamlit run $(STREAMLIT_FILE_PATH) --server.headless true

# testing and linting
test:
	STREAMLIT_FILE_PATH=$(STREAMLIT_FILE_PATH) uv run pytest -v

format:
	uv run ruff check --fix .
	uv run ruff format .

# qdrant data manipulation
load_qdrant_locations:
	uv run -m cli.qdrant_loader --source 'tripadvisor-recommendations.fs_tripadvisor.fs_location' --collection 'tripadvisor_locations' --embedding_column 'location_text_nlp'

load_qdrant_reviews:	
	uv run -m cli.qdrant_loader --source 'tripadvisor-recommendations.fs_tripadvisor.fs_review' --collection 'tripadvisor_reviews' --embedding_column 'review_text_nlp'