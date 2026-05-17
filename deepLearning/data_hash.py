"""MD5 hash of a dataset path for versioning (``train_core`` optional ``data_hash`` field)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def hash_data(data_path: Union[str, Path], chunk_size: int = 65536) -> str:
    if Path(data_path).is_dir():
        hash = _hash_dir(data_path, chunk_size)
    elif Path(data_path).is_file():
        hash = _hash_file(data_path, chunk_size)
    else:
        raise ValueError(f"{data_path} is neither directory nor file.")
    return hash.hexdigest()


def _update_hash_dir(directory: Union[str, Path], hash, chunk_size: int):
    assert Path(directory).is_dir()
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    hash.update(chunk)
        elif path.is_dir():
            hash = _update_hash_dir(path, hash, chunk_size)
    return hash


def _hash_dir(directory: Union[str, Path], chunk_size: int):
    return _update_hash_dir(directory, hashlib.md5(), chunk_size)


def _hash_file(data_file: Union[str, Path], chunk_size: int):
    hash = hashlib.md5()
    with open(data_file, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash.update(chunk)
    return hash
