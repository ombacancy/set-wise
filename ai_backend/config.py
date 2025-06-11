from dotenv import load_dotenv

load_dotenv()


class Settings:
    GROQ_API_KEY = "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"
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VDB_PATH = "./chroma_db"
    VDB_COLLECTION = "gym_logs"
    DEBUG = True


settings = Settings()
