.PHONY: run format load_qdrant_locations load_qdrant_geolocations load_qdrant
export CHAINLIT_FILE_PATH := _chainlit.py

# development
run:
	@docker compose up -d
	@if ! grep -q "CHAINLIT_AUTH_SECRET" .env 2>/dev/null; then \
		uv run chainlit create-secret >> .env; \
	fi
	@uv run chainlit run $(CHAINLIT_FILE_PATH) -w

db:
	@docker compose up -d
	@npx prisma migrate deploy
	@npx prisma db push
	@npx prisma studio

format:
	@uv run ruff check --fix .
	@uv run ruff format .

# qdrant include/data manipulation
load_qdrant_locations:
	@uv run -m src.qdrant.cli.loader --source 'include/data/fs_location.parquet' --collection 'tripadvisor_locations' --embedding_column 'location_text_nlp'

load_qdrant_geolocations:
	@uv run -m src.qdrant.cli.loader --source 'include/data/fs_geolocation.parquet' --collection 'tripadvisor_geolocations' --embedding_column 'geolocation_text_nlp'

load_qdrant:
	@make load_qdrant_locations
	@make load_qdrant_geolocations
