import asyncio
import datetime
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import bcrypt
import jwt
import whisper
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError, PyJWTError
from sentence_transformers import SimilarityFunction

from embedding_models import models
from llamacpp_client import LlamaCppAPIClient
from llamacpp_client.llamacpp_client import DEFAULT_OPTIONS, MODEL
from models import (
    CosineResponse,
    EmbedAndCosineRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    GenericStatusResponse,
    LLMRequest,
    LLMResponse,
    LoginRequest,
    LoginResponse,
    PoliticalStanceRequest,
    PoliticalStanceResponse,
    QAItem,
    QAResponse,
    SimilarityResponse,
    SpeechEmbeddingAddRequest,
    SpeechEmbeddingUpdateRequest,
    SummaryFromUrlRequest,
    SummaryRequest,
    SummaryResponse,
    TopicsFromUrlRequest,
    TopicsRequest,
    TopicsResponse,
)
from transcriber import process_video

router = APIRouter()

# === JWT AUTH ===
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = os.getenv("JET_EXPIRE_MINUTES", "120")
API_SECRET = os.getenv("API_SECRET")

if not JWT_SECRET or not API_SECRET:
    print("No Secrets given.")
    quit()

FAKE_USERS_DB = {
    "API_USER": bcrypt.hashpw(
        bytes(API_SECRET, encoding="utf-8"), bcrypt.gensalt()
    ).decode("utf-8"),
}

logger = logging.getLogger("uvicorn.error")

model = None


def is_localhost(ip: str) -> bool:
    return ip in ("127.0.0.1", "::1")


@router.post("/local_login", response_model=LoginResponse)
async def local_login(request: Request):
    client_ip = request.client.host
    if not is_localhost(client_ip):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Local login only allowed from 127.0.0.1, got {client_ip}",
        )

    expire = datetime.utcnow() + timedelta(minutes=float(JWT_EXPIRE_MINUTES))
    payload = {
        "sub": "API_USER",
        "exp": expire,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return LoginResponse(access_token=token)


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    hashed_pw = FAKE_USERS_DB.get(req.username)
    if not hashed_pw:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not bcrypt.checkpw(req.password.encode("utf-8"), hashed_pw.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    expire = datetime.utcnow() + timedelta(minutes=float(JWT_EXPIRE_MINUTES))
    payload = {
        "sub": req.username,
        "exp": expire,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return LoginResponse(access_token=token)


async def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> bool:
    token = credentials.credentials
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing JWT token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(
            timezone.utc
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return True

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (InvalidTokenError, PyJWTError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWT token",
            headers={"WWW-Authenticate": "Bearer"},
        )


SERVER_URL = "http://10.30.0.164:8080"

model = None


async def make_llm_request(req: LLMRequest) -> str:
    options = DEFAULT_OPTIONS
    if req.model:
        options["model"] = req.model
    llama_client = LlamaCppAPIClient(base_url=SERVER_URL, options=options)

    messages = [
        {"role": "system", "content": req.system},
        {"role": "user", "content": req.input},
    ]
    out = ""
    async for chunk in llama_client.stream_completion(
        messages, json_schema=req.json_schema
    ):
        print(chunk)
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        print(delta, end="", flush=True)
        if delta:
            out += delta
    return out


@router.get("/model", response_model=LLMResponse)
async def get_model(_: bool = Depends(verify_jwt)) -> LLMResponse:
    return LLMResponse(result=MODEL)


# === LLM REQUESTS ===
@router.post("/make_llm_request", response_model=LLMResponse)
async def make_llm_request_route(
    req: LLMRequest, _: bool = Depends(verify_jwt)
) -> LLMResponse:
    out = await make_llm_request(req)
    return LLMResponse(result=out)


@router.post("/make_llm_request_streamed")
async def make_llm_request_streamed(
    req: LLMRequest, _: bool = Depends(verify_jwt)
) -> StreamingResponse:
    async def stream() -> AsyncGenerator[str, None]:
        for chunk in ["streaming", "llm", "response"]:
            yield f"{chunk}\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/plain")


# === TOPIC PREDICTION ===
@router.post("/make_topics_predictions", response_model=TopicsResponse)
async def make_topics_predictions(
    req: TopicsRequest, _: bool = Depends(verify_jwt)
) -> TopicsResponse:
    return TopicsResponse(topics=["topic1", "topic2"])


@router.post("/make_topics_predictions_from_url", response_model=TopicsResponse)
async def make_topics_predictions_from_url(
    req: TopicsFromUrlRequest, _: bool = Depends(verify_jwt)
) -> TopicsResponse:
    return TopicsResponse(topics=["topicA", "topicB"])


# === DELEGATE VIDEO QA ===
@router.get("/extract_delegate_preview_video_qa", response_model=QAResponse)
async def extract_delegate_preview_video_qa(
    delegate_id: int, _: bool = Depends(verify_jwt)
) -> QAResponse:
    if model is None:
        model = whisper.load_model(name="medium", device="cpu")

    transcribe_results = None
    try:
        transcribe_results = process_video(delegate_id, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal processing error")

    qa_items = []
    for result in transcribe_results:
        qa_items.append(QAItem(question=result["question"], answer=result["answer"]))

    return QAResponse(qa=qa_items)


# === SPEECH EMBEDDINGS ===
@router.post("/speech_embedding/add", response_model=GenericStatusResponse)
async def speech_embedding_add(
    req: SpeechEmbeddingAddRequest, _: bool = Depends(verify_jwt)
) -> GenericStatusResponse:
    return GenericStatusResponse(status="speech embedding added")


@router.post("/speech_embedding/update_all", response_model=GenericStatusResponse)
async def speech_embedding_update_all(
    req: SpeechEmbeddingUpdateRequest, _: bool = Depends(verify_jwt)
) -> GenericStatusResponse:
    return GenericStatusResponse(status="all embeddings updated")


# === EMBEDDING OPS ===
@router.post("/embedding/embed", response_model=EmbeddingResponse)
async def embedding_embed(
    req: EmbeddingRequest, _: bool = Depends(verify_jwt)
) -> EmbeddingResponse:
    model_id = req.model
    if not req.model:
        model_id = "mixedbread-ai/deepset-mxbai-embed-de-large-v1"
    if model_id not in models:
        raise HTTPException(status_code=400, detail=str("Model not supported"))
    model = models[model_id]

    embedding = model.encode([req.input])[0]

    return EmbeddingResponse(embedding=embedding)


@router.post("/embedding/embed_and_cosine", response_model=CosineResponse)
async def embedding_embed_and_cosine(
    req: EmbedAndCosineRequest, _: bool = Depends(verify_jwt)
) -> CosineResponse:
    model_id = req.model
    if not req.model:
        model_id = "Qwen/Qwen3-Embedding-4B"
    if model_id not in models:
        raise HTTPException(status_code=400, detail=str("Model not supported"))
    model = models[model_id]
    model.similarity_fn_name = SimilarityFunction.COSINE

    lhs_embedding = model.encode([req.lhs])[0]
    rhs_embedding = model.encode([req.rhs])[0]

    similarity = model.similarity(lhs_embedding, rhs_embedding)

    return CosineResponse(cosine_similarity=similarity)


# === POLITICAL STANCE ===
@router.post("/political_stance/generate", response_model=PoliticalStanceResponse)
async def political_stance_generate(
    req: PoliticalStanceRequest, _: bool = Depends(verify_jwt)
) -> PoliticalStanceResponse:
    return PoliticalStanceResponse(stance="stance placeholder")


@router.post("/political_stance/generate_streamed")
async def political_stance_generate_streamed(
    req: PoliticalStanceRequest, _: bool = Depends(verify_jwt)
) -> StreamingResponse:
    async def stream():
        for chunk in ["streaming", "stance"]:
            yield f"{chunk}\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/plain")
