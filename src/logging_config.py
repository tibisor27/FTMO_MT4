# src/logging_config.py
from pathlib import Path
import logging
import sys

def setup_log(path: Path | None = None) -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)-14s %(message)s"
    formatter = logging.Formatter(fmt)

    # 1. Luăm root logger-ul (părintele absolut al tuturor loggerilor din aplicație)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 2. Curățăm handler-ele vechi ca să nu se suprapună
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3. Adăugăm handler-ul pentru Terminal (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # 4. Adăugăm handler-ul pentru Fișier (Dacă calea este furnizată)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)