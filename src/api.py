import datetime
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import os
from typing import AsyncGenerator
import asyncio

from llamacpp_client.llamacpp_client import DEFAULT_OPTIONS
from models import (
    LLMRequest,
    LLMResponse,
    LoginRequest,
    LoginResponse,
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
import bcrypt
import jwt
from jwt import PyJWTError, ExpiredSignatureError, InvalidTokenError

from llamacpp_client import LlamaCppAPIClient

router = APIRouter()

# === JWT AUTH ===
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = os.getenv("JET_EXPIRE_MINUTES", "120")
API_SECRET = os.getenv("API_SECRET")

if not JWT_SECRET or not API_SECRET:
    print("No Secrets given.")
    quit();

FAKE_USERS_DB = {
    "API_USER": bcrypt.hashpw(bytes(API_SECRET, encoding="utf-8"), bcrypt.gensalt()).decode("utf-8"),
}

def is_localhost(ip: str) -> bool:
    return ip in ("127.0.0.1", "::1")

@router.post("/local_login", response_model=LoginResponse)
async def local_login(request: Request):
    client_ip = request.client.host
    if not is_localhost(client_ip):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Local login only allowed from 127.0.0.1, got {client_ip}"
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

    expire = datetime.utcnow() + datetime.timedelta(minutes=float(JWT_EXPIRE_MINUTES))
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

async def make_llm_request(req: LLMRequest) -> str:
    options = DEFAULT_OPTIONS
    if req.model:
        options["model"] = req.model
    llama_client = LlamaCppAPIClient(base_url=SERVER_URL, options=options)

    messages = [{"role": "system", "content": req.system}, {"role": "user", "content": req.input}]
    out = ""
    async for chunk in llama_client.stream_completion(messages):
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            out += delta
    return out

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


# === SUMMARIZATION ===
@router.post("/summarize_input", response_model=SummaryResponse)
async def summarize_input(
    req: SummaryRequest, _: bool = Depends(verify_jwt)
) -> SummaryResponse:
    llm_request = LLMRequest(
        input=req.input,
        system="""
        Du bist ein Experte für das Zusammenfassen von Gesetzestexten des österreichischen Nationalrats in Form einer Liste an Schwerpunkten für Laien und politisch Interessierte. 
        Dabei geht es um den vollständigen Kontext, den der Gestzestext behandelt, um einfach nachzuvollziehen, was die Änderungen bedeuten. 
        Fasse den Gesetzestext zusammen. Entferne Informationen über typische Formulierungen mit Paragraphen und Absätze, die in dem Gesetzestext vorkommen können.
        Erhalte Informationen über die wirklich aussagekräftigen wichtigen Inhalte des Gesetzestextes und lass die Informationen weg, die zur Verwirrung von Laien führen.
        Verwendungen von zum Beispiel "§ 18 Abs. 2a", "§ 20 Abs. 10 bis 20 und § 48 Z 3b", ... führt maßgeblich zur Verwirrung und Abschreckung von Laien, weshalb diese nicht in der Zusammenfassung erscheinen dürfen.
        Zahlen und andere wichtigen Informationen sollten in der Zusammenfassung erhalten bleiben. 
        Hebe dabei die wichtigsten Schwerpunkte nur als roh-text (keine Markdown Formatierungen, etc.) Liste hervor.
        Fang direkt ohne Einleitung mit den Schwerpunkten an und verzichte dabei auf sonstige Markdown Formatierungen!
        Schreibe pro Seite Gesetzestext maximal 5 Punkte.
        Der erste Punkt muss eine grobe Zusammenfassung sein, worum es allgemein beim Gesetzestext geht.
        Positives Beispiel, so muss dein Output sein:
        - Punkt 1
        - Punkt 2
        - Punkt 3
        - Punkt 4
        - ...

        Negatives Beispiel, so darf dein Output NICHT sein:
        Im folgenden Teil werden viele Punkte besprochen:
        - **Punkt 1**: § 13 Abs. 6b  
        - **Punkt 2** [...] § 32 Abs. 12 bis 33 und § 44 Z 5c
        - **Punkt 3**: ...
        - **Punkt 4**
        - ...
        """,
        model=req.model,
        use_openai=False,
    )
    out = await make_llm_request(llm_request)
    return SummaryResponse(summary=out)


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
