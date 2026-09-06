import math
import threading
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder, SentenceTransformer

model_ids = ["mixedbread-ai/deepset-mxbai-embed-de-large-v1", "Qwen/Qwen3-Embedding-4B"]

models = {}
for model_id in model_ids:
    model = SentenceTransformer(
        model_id,
        trust_remote_code=True,
        device="cpu",
        model_kwargs={"torch_dtype": "float16"},
    )
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    models[model_id] = model

rerank_model_ids = ["mixedbread-ai/mxbai-rerank-base-v2"]

DEFAULT_RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"

RERANK_MAX_LENGTH = 512

cross_encoders = {}
_cross_encoder_lock = threading.Lock()


def get_cross_encoder(model_id: Optional[str] = None) -> CrossEncoder:
    if not model_id:
        model_id = DEFAULT_RERANK_MODEL
    if model_id not in rerank_model_ids:
        raise KeyError(model_id)
    with _cross_encoder_lock:
        if model_id not in cross_encoders:
            cross_encoders[model_id] = CrossEncoder(
                model_id,
                device="cpu",
                max_length=RERANK_MAX_LENGTH,
            )
        return cross_encoders[model_id]


def _sigmoid(score: float) -> float:
    if score >= 0:
        return 1 / (1 + math.exp(-score))
    normalized = math.exp(score)
    return normalized / (1 + normalized)


def rerank(
    query: str,
    documents: List[str],
    model_id: Optional[str] = None,
    top_n: Optional[int] = None,
) -> List[Tuple[int, float]]:
    if not documents:
        return []

    cross_encoder = get_cross_encoder(model_id)
    pairs = [(query, document) for document in documents]
    scores = [(index, _sigmoid(float(score))) for index, score in enumerate(cross_encoder.predict(pairs))]

    ranked = sorted(scores, key=lambda item: item[1], reverse=True)
    if top_n is not None:
        ranked = ranked[: max(top_n, 0)]

    return ranked


if __name__ == "__main__":
    test_text = "This is a test"
    for model_id, model in models.items():
        embedding = model.encode([test_text])
        print(f"Model: {model_id}, Embedding Length: {embedding}")

    test_query = "Who wrote 'To Kill a Mockingbird'?"
    test_documents = [
        "Harper Lee wrote 'To Kill a Mockingbird', published in 1960.",
        "Moby-Dick was written by Herman Melville.",
    ]
    print(f"Rerank: {rerank(test_query, test_documents)}")
