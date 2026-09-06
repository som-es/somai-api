
from pydantic import BaseModel, HttpUrl
from typing import List, Optional


# === LLM REQUESTS ===
class LLMRequest(BaseModel):
    input: str
    system: str
    model: Optional[str]
    use_openai: bool = False
    json_schema: Optional[dict] = None

class LLMResponse(BaseModel):
    result: str


# === SUMMARIZATION ===
class SummaryRequest(BaseModel):
    input: str
    model: Optional[str] = None
    json_schema: Optional[dict] = None

class SummaryFromUrlRequest(BaseModel):
    url: HttpUrl
    model: str

class SummaryResponse(BaseModel):
    summary: str


# === TOPICS ===
class TopicsRequest(BaseModel):
    input: str

class TopicsFromUrlRequest(BaseModel):
    url: HttpUrl

class TopicsResponse(BaseModel):
    topics: List[str]


# === VIDEO QA ===
class QAItem(BaseModel):
    question: str
    answer: str

class QAResponse(BaseModel):
    qa: List[QAItem]


# === SIMILARITY ===
class SimilarityResponse(BaseModel):
    similarity_score: float


# === SPEECH EMBEDDING ===
class SpeechEmbeddingAddRequest(BaseModel):
    delegate_id: int
    speech: str
    speech_id: Optional[int] = None

class SpeechEmbeddingUpdateRequest(BaseModel):
    delegate_id: int

class GenericStatusResponse(BaseModel):
    status: str


# === EMBEDDING OPS ===
class EmbeddingRequest(BaseModel):
    input: str
    model: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class EmbedAndCosineRequest(BaseModel):
    lhs: str
    rhs: str
    model: Optional[str] = None

class CosineResponse(BaseModel):
    cosine_similarity: float

class RerankingRequest(BaseModel):
    input: str
    to_rerank: List[str]
    model: Optional[str] = None
    top_n: Optional[int] = None

class RerankedDocument(BaseModel):
    index: int
    text: str
    score: float

class RerankingResponse(BaseModel):
    model: str
    results: List[RerankedDocument]


# === POLITICAL STANCE ===
class PoliticalStanceRequest(BaseModel):
    delegate_id: int
    question_or_input: str
    model: Optional[str]

class PoliticalStanceResponse(BaseModel):
    stance: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
