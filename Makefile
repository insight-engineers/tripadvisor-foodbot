.PHONY: dev start test format load_qdrant_locations load_qdrant_reviews load_qdrant_geolocations
export STREAMLIT_FILE_PATH := _streamlit.py
export CHAINLIT_FILE_PATH := _chainlit.py

# development
cl:
	uv run chainlit run $(CHAINLIT_FILE_PATH) -w

st:
	uv run streamlit run $(STREAMLIT_FILE_PATH)

# testing and linting
test-st:
	STREAMLIT_FILE_PATH=$(STREAMLIT_FILE_PATH) uv run pytest -v -W ignore::DeprecationWarning

format:
	uv run ruff check --fix .
	uv run ruff format .

# qdrant data manipulation
load_qdrant_locations:
	uv run -m cli.qdrant_loader --source 'tripadvisor-recommendations.fs_tripadvisor.fs_location' --collection 'tripadvisor_locations' --embedding_column 'location_text_nlp'

load_qdrant_reviews:	
	uv run -m cli.qdrant_loader --source 'tripadvisor-recommendations.fs_tripadvisor.fs_review' --collection 'tripadvisor_reviews' --embedding_column 'review_text_nlp'

load_qdrant_geolocations:
	uv run -m cli.qdrant_loader --source 'tripadvisor-recommendations.fs_tripadvisor.fs_geolocation' --collection 'tripadvisor_geolocations' --embedding_column 'geolocation_text_nlp'
