import pandas as pd
from sentence_transformers import SentenceTransformer, util


pd.set_option('display.max_columns', None)

# read from the sample data set
df = pd.read_json('engineers_linkedin_chunk_1.ndjson', lines=True)

# picked ABOUT col for this excrcise, you can combine multiple relevant cols I guess.
# I don't have a beefy machine to train the model, thus went with an eaiser case.
sentences = df['about'].apply(lambda x: str(x) if x is not None else '').to_list()

# set a matching target
target = 'developer who went to stanford'

sentences_dict = {}
ptr = 0

for s in sentences:
    sentences_dict[ptr] = s
    ptr+=1

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
target_embedding = model.encode(target, convert_to_tensor=True)

scores_dict = {}

# calculate similarity between target and each personal profile
for k, v in sentences_dict.items():
    single_embedding = model.encode(v, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(target_embedding, single_embedding)

    #print(cos_sim.numpy()[0][0])

    scores_dict[k] = cos_sim.numpy()[0][0]

# sort by similarity
sorted_scores_dict = dict(sorted(scores_dict.items(), key=lambda item: item[1]))

TOP_N = 10

for k, v in sorted_scores_dict.items():
    if TOP_N > 0:
        print(f'index: {k}, similarity: {v}')

    TOP_N -= 1

