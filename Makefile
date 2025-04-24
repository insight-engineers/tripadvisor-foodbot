.PHONY: dev format load_qdrant_locations load_qdrant_reviews

dev:
	uv run chainlit run ui/app.py -w --headless

start:
	uv run chainlit run ui/app.py --headless

format:
	uv run black .
	uv run isort . --profile black
	uv run ruff format .

load_qdrant_locations:
	uv run -m cli.qdrant_loader --source 'tripadvisor-recommendations.fs_tripadvisor.fs_location' --collection 'tripadvisor_locations' --embedding_column 'location_text_nlp'
	
load_qdrant_reviews:	
	uv run -m cli.qdrant_loader --source 'tripadvisor-recommendations.fs_tripadvisor.fs_review' --collection 'tripadvisor_reviews' --embedding_column 'review_text_nlp'