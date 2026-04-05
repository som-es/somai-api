from sentence_transformers import SentenceTransformer

model_ids = ["mixedbread-ai/deepset-mxbai-embed-de-large-v1", "Qwen/Qwen3-Embedding-4B"]

models = {}
for model_id in model_ids:
    model = SentenceTransformer(model_id, trust_remote_code=True, device="cpu")
    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    models[model_id] = model

if __name__ == "__main__":
    test_text = "This is a test"
    for model_id, model in models.items():
        embedding = model.encode([test_text])
        print(f"Model: {model_id}, Embedding Length: {embedding}")
