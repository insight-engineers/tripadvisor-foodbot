import argparse

from src.qdrant.load import QdrantLoader

parser = argparse.ArgumentParser(description="Simple Qdrant loader CLI.")
parser.add_argument("--source", required=True, help="BigQuery table to load from.")
parser.add_argument("--collection", required=True, help="Qdrant collection name.")
parser.add_argument("--embedding_column", required=True, help="Column to embed.")
args = parser.parse_args()


if __name__ == "__main__":
    loader = QdrantLoader(
        source=args.source,
        collection_name=args.collection,
        embedding_column=args.embedding_column,
    )
    loader.load_data()
