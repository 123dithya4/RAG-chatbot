import os
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

SECRET_KEY = "your-secret-key-change-in-production"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ALGORITHM = "HS256"
EMBEDDING_DIM = 384
DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")

# Ensure data dir exists immediately
os.makedirs(DATA_DIR, exist_ok=True)