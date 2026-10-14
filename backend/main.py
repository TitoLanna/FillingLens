"""Entry point: python3 main.py"""
from filinglens import config, db, embeddings
from filinglens.server import serve

if __name__ == "__main__":
    settings = config.load()
    embeddings.configure(settings)
    serve(settings, db.connect(settings.db_path))
