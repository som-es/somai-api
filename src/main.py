from fastapi import FastAPI
from api import router as api_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="LLM & Delegate Analysis API",
    description="REST API for LLM requests, summarization, embeddings, and political stance generation.",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api")


# For local development:
# Run with: uvicorn main:app --reload
