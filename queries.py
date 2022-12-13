import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
dataset = load_dataset("csv", data_files="embeddings.csv")
dataset_embeddings = torch.from_numpy(dataset["train"].to_pandas().to_numpy()).to(torch.float)


queries = ['full stack engineer with experience in python', 'developer who went to stanford']

for q in queries:
    query_embeddings = model.encode(q)
    hits = semantic_search(query_embeddings, dataset_embeddings, top_k=10)
    print(q)
    print(hits)