import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import os
from typing import AsyncGenerator
import asyncio

from models import (
    LLMRequest,
    LLMResponse,
    SummaryRequest,
    SummaryFromUrlRequest,
    SummaryResponse,
    TopicsRequest,
    TopicsFromUrlRequest,
    TopicsResponse,
    QAResponse,
    QAItem,
    SimilarityResponse,
    SpeechEmbeddingAddRequest,
    SpeechEmbeddingUpdateRequest,
    GenericStatusResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbedAndCosineRequest,
    CosineResponse,
    PoliticalStanceRequest,
    PoliticalStanceResponse,
)
import jwt
from jwt import PyJWTError, ExpiredSignatureError, InvalidTokenError

router = APIRouter()

# === JWT AUTH ===
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_key")      
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


async def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
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
        if exp and datetime.fromtimestamp(exp, tz=datetime.timezone.utc) < datetime.now(datetime.timezone.utc):
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


# === LLM REQUESTS ===
@router.post("/make_llm_request", response_model=LLMResponse)
async def make_llm_request(
    req: LLMRequest, _: bool = Depends(verify_jwt)
) -> LLMResponse:
    return LLMResponse(result="LLM output placeholder")


@router.post("/make_llm_request_streamed")
async def make_llm_request_streamed(
    req: LLMRequest, _: bool = Depends(verify_jwt)
) -> StreamingResponse:
    async def stream() -> AsyncGenerator[str, None]:
        for chunk in ["streaming", "llm", "response"]:
            yield f"{chunk}\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/plain")


# === SUMMARIZATION ===
@router.post("/summarize_input", response_model=SummaryResponse)
async def summarize_input(
    req: SummaryRequest, _: bool = Depends(verify_jwt)
) -> SummaryResponse:
    return SummaryResponse(summary="summary placeholder")


@router.post("/summarize_input_streamed")
async def summarize_input_streamed(
    req: SummaryRequest, _: bool = Depends(verify_jwt)
) -> StreamingResponse:
    async def stream():
        for chunk in ["streamed", "summary"]:
            yield f"{chunk}\n"
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/plain")


@router.post("/summarize_input_from_url", response_model=SummaryResponse)
async def summarize_input_from_url(
    req: SummaryFromUrlRequest, _: bool = Depends(verify_jwt)
) -> SummaryResponse:
    return SummaryResponse(summary=f"summary for {req.url}")


@router.post("/summarize_input_from_url_streamed")
async def summarize_input_from_url_streamed(
    req: SummaryFromUrlRequest, _: bool = Depends(verify_jwt)
) -> StreamingResponse:
    async def stream():
        for chunk in ["streamed", "url", "summary"]:
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
    return QAResponse(
        qa=[QAItem(question="Q1", answer="A1"), QAItem(question="Q2", answer="A2")]
    )


# === DELEGATE INTEREST SIMILARITY ===
@router.get(
    "/compute_delegate_interests_division_similarity", response_model=SimilarityResponse
)
async def compute_delegate_interests_division_similarity(
    delegate_id: int, _: bool = Depends(verify_jwt)
) -> SimilarityResponse:
    return SimilarityResponse(similarity_score=0.87)


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
    return EmbeddingResponse(embedding=[0.12, 0.34, 0.56])


@router.post("/embedding/embed_and_cosine", response_model=CosineResponse)
async def embedding_embed_and_cosine(
    req: EmbedAndCosineRequest, _: bool = Depends(verify_jwt)
) -> CosineResponse:
    return CosineResponse(cosine_similarity=0.91)


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
