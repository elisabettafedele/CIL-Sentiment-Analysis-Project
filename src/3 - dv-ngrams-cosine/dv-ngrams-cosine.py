import pandas as pd
import re
import time
from collections import Counter
import numba
from numba import jit
import numpy as np

# train weight matrices
@jit(nopython=True)
def train(docs, len_ngram_ids, doc_vecs, ngram_vecs, nb_weights, alpha, neg_size, lr, nb_A, nb_B):
    doc_vecs = np.asarray(doc_vecs)
    ngram_vecs = np.asarray(ngram_vecs)
    nb_weights = np.asarray(nb_weights)
    
    for index, ngram_id in np.ndenumerate(docs):
        i = index[0]
        if(ngram_id == -1 or np.random.rand() > np.exp(nb_weights[ngram_id] / nb_A) / nb_B):
            continue
                
        w_ids = np.zeros((neg_size + 1), dtype=np.int64)
        w_ids[0] = ngram_id
        for j in range(neg_size):
            w_ids[j + 1] = np.random.randint(0, len_ngram_ids)
                
        d_norm = np.linalg.norm(doc_vecs[i])
        for j, w_id in enumerate(w_ids):
            w_norm = np.linalg.norm(ngram_vecs[w_id])
            cos = np.dot(doc_vecs[i], ngram_vecs[w_id]) / (d_norm * w_norm)
            y = 1.0 / (1.0 + np.exp(-alpha * cos))
            t = 1 if j == 0 else 0
            doc_vecs[i] -= alpha * (y - t) * ((ngram_vecs[w_id] / (d_norm * w_norm)) - (doc_vecs[i] * cos / pow(d_norm, 2))) * lr
            ngram_vecs[w_id] -= alpha * (y - t) * ((doc_vecs[i] / (d_norm * w_norm)) - (ngram_vecs[w_id] * cos / pow(w_norm, 2))) * lr


# parameters
data_path = 'twitter-datasets/'
result_name="dv-ngrams-cosine-embeddings.txt"
dim_emb=500
neg_size=5
alpha=6
epochs=290
lr=1e-3
nb_A=2
nb_B=3
weights_init_min=-1e-3
weights_init_max=1e-3


# compute bigrams and trigrams
n_pos = 0
n_neg = 0
print("compute ngrams")
lines = []
with open(data_path + 'train_pos.txt', "r", encoding='utf-8') as f:
    for line in f:
        lines.append(line)
        n_pos += 1
        
with open(data_path + 'train_neg.txt', "r", encoding='utf-8') as f:
    for line in f:
        lines.append(line)
        n_neg += 1
        
with open(data_path + 'test_data.txt', "r", encoding='utf-8') as f:
    for line in f:
        lines.append(line[line.find(',') + 1:])

docs_raw = []
for i, l in enumerate(lines):
    l = re.sub(r'([\.",\(\)\!\?:;])', r' \1 ', l.lower())
    l = re.sub('<br />|\x85', ' ', l)
    words = l.split() if i < n_pos + n_neg else l.split()[1:]
    ngrams = words.copy()
    ngrams.extend([words[i] + "@$" + words[i+1] for i in range(len(words)-1)])
    ngrams.extend([words[i] + "@$" + words[i+1] + "@$" + words[i+2] for i in range(len(words)-2)])
    docs_raw.append(ngrams)
        
        
# compute vocabulary and documents with ids instead of ngrams
print("create vocabulary")
ngram_ids = {}
counter = Counter()
for ngrams in docs_raw:
    counter.update(ngrams)
    
cur_id = 0
for ngram, count in counter.items():
    if count > 2:
        ngram_ids[ngram] = cur_id
        cur_id += 1

docs = []
for ngrams in docs_raw:
    docs.append([ngram_ids[ngram] for ngram in ngrams if ngram in ngram_ids])
    

# compute weights for NB subsampling
print("compute weights for subsampling")
nb_weights = np.full(len(ngram_ids), 1.0)
pos_counts = np.full(len(ngram_ids), 1.0)
neg_counts = np.full(len(ngram_ids), 1.0)
pos_count = 0
neg_count = 0
for i, doc in enumerate(docs):
    for ngram_id in doc:
        if i < n_pos:
            pos_counts[ngram_id] += 1
            pos_count += 1
        elif i < n_pos + n_neg:
            neg_counts[ngram_id] += 1
            neg_count += 1
logPos = np.log(pos_count + len(ngram_ids))
logNeg = np.log(neg_count + len(ngram_ids))
logPN = logPos - logNeg
nb_weights = np.asarray([np.abs(np.log(pos_counts[ngram_id]) - np.log(neg_counts[ngram_id]) - logPN) for _, ngram_id in ngram_ids.items()])
    
    
# init weight matrices and train
print("init weights and start training")
rng = np.random.default_rng()
doc_vecs = rng.random(size=(len(docs), dim_emb)) * (weights_init_max - weights_init_min) + weights_init_min
ngram_vecs = rng.random(size=(len(ngram_ids), dim_emb)) * (weights_init_max - weights_init_min) + weights_init_min
   
docs_padded = pd.DataFrame(docs).fillna(-1).values.astype(np.int64)
for epoch in range(epochs):
    print("Epoch " + str(epoch))
    t_start = time.perf_counter()
    train(docs_padded, len(ngram_ids.keys()), doc_vecs, ngram_vecs, nb_weights, alpha, neg_size, lr, nb_A, nb_B)
    t_end = time.perf_counter()
    print(f"{t_end - t_start} seconds")


# save doc embeddings
print("save results")
np.savetxt(result_name, doc_vecs)
