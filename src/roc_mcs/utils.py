from pathlib import Path

def resolve_output_folder(output_folder=None):
    return Path.cwd() if output_folder is None else Path(output_folder)

def ensure_folder(path):
    path.mkdir(parents=True, exist_ok=True)
    return path    