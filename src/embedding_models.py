from sentence_transformers import SentenceTransformer, CrossEncoder

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

rerank_model_ids = ["mixedbread-ai/mxbai-rerank-large-v2"]

cross_encoder_models = {}
for model_id in rerank_model_ids:
    cross_encoder_models[model_id] = CrossEncoder(model_id)

DEFAULT_RERANK_MODEL = rerank_model_ids[0]

if __name__ == "__main__":
    test_text = "This is a test"
    for model_id, model in models.items():
        embedding = model.encode([test_text])
        print(f"Model: {model_id}, Embedding Length: {embedding}")

    rerank_model = cross_encoder_models[DEFAULT_RERANK_MODEL]
    test_query = "Who wrote 'To Kill a Mockingbird'?"
    test_documents = [
        "Harper Lee wrote 'To Kill a Mockingbird', published in 1960.",
        "Moby-Dick was written by Herman Melville.",
    ]
    print(f"Rankings: {rerank_model.rank(test_query, test_documents)}")
