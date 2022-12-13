import pandas as pd
from sentence_transformers import SentenceTransformer


# read from the sample data set
df = pd.read_json('engineers_linkedin_chunk_1.ndjson', lines=True)

# picked ABOUT col for this excrcise, you can combine multiple relevant cols I guess.
# I don't have a beefy machine to train the model, thus went with an eaiser case.
sentences = df['about'].apply(lambda x: str(x) if x is not None else '').to_list()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
e = model.encode(sentences)

pd.DataFrame(e).to_csv("embeddings.csv", index=False)


