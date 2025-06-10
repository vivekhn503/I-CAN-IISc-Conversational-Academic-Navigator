import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))
MODEL_NAME     = os.getenv("MODEL_NAME", "deepseek-r1:latest")
TEMPERATURE    = float(os.getenv("TEMPERATURE", 0.4))

