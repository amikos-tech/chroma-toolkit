from pathlib import Path


def backup(collections: list = None, output_file: Path = None):
    """Backup the database to a file."""
    if collections:
        return get("/backup", params={"collections": collections})
    else:
        return get("/backup")