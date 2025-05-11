from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    MAX_SEARCH_RESULTS = 1
    CHUNK_SIZE = 1000