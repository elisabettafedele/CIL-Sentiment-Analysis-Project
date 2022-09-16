from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from matplotlib.pyplot import sca
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBRFClassifier, XGBClassifier
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd
import gensim

with open("twitter-datasets/train_pos.txt","r") as infile:
    pos_tweets = infile.readlines()

with open("twitter-datasets/train_neg.txt","r") as infile:
    neg_tweets = infile.readlines()

y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

tweets = np.concatenate((pos_tweets, neg_tweets))

x_train, x_test, y_train, y_test = train_test_split(tweets, y, test_size=0.2)

def read_corpus(tweets, labels):
    for tweet, label in zip(tweets, labels) :
        tokens = gensim.utils.simple_preprocess(tweet)
        
        # For training data, add tags
        yield gensim.models.doc2vec.TaggedDocument(tokens, [label])

train_corpus = list(read_corpus(x_train, y_train))
test_corpus = list(read_corpus(x_test, y_test))

# train_tagged = train.apply(
#     lambda r: TaggedDocument(words=tokenize_text(r['tweets']), tags=[r.y]), axis=1)

size = 400

print("Creating doc2vec")
#instantiate our DM and DBOW models
model_dm = Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, workers=3, epochs=40)
model_dbow = Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, dm=0, workers=3, epochs=40)
print("done")

print("build vocab")
#build vocab over all reviews
model_dm.build_vocab(train_corpus)
model_dbow.build_vocab(train_corpus)
print("done")


print("training doc2vec")
model_dm.train(train_corpus, total_examples=model_dm.corpus_count, epochs=model_dm.epochs, report_delay=1)
model_dbow.train(train_corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs, report_delay=1)
print("done")


def vec_for_learning(model, tagged_docs):
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in tagged_docs])
    return targets, regressors

print("creating vectors")
y_train, X_train = vec_for_learning(model_dbow, train_corpus)
y_test, X_test = vec_for_learning(model_dbow, test_corpus)
y_train_2, X_train_2 = vec_for_learning(model_dm, train_corpus)
y_test_2, X_test_2 = vec_for_learning(model_dm, test_corpus)

X_train = np.hstack((X_train, X_train_2))
X_test = np.hstack((X_test, X_test_2))
with open(f"x_train_{size}_doc2vec.npy", "wb") as f:
    np.save(f, X_train)
with open(f"y_train_{size}_doc2vec.npy", "wb") as f:
    np.save(f, y_train)
with open(f"x_test_{size}_doc2vec.npy", "wb") as f:
    np.save(f, X_test)
with open(f"y_test_{size}_doc2vec.npy", "wb") as f:
    np.save(f, y_test)
print("done")



# # print("generating train_vecs")
# # train_vecs = np.concatenate([buildWordVector(z, size) for z in x_train])
# # train_vecs = scale(train_vecs)
# # with open(f"x_train_{size}.npy", "wb") as f:
# #     np.save(f, train_vecs)
# # with open(f"y_train_{size}.npy", "wb") as f:
#     np.save(f, y_train)
# print("done")

# print("generating test_vecs")
# test_vecs = np.concatenate([buildWordVector(z, size) for z in x_test])
# test_vecs = scale(test_vecs)
# with open(f"x_test_{size}.npy", "wb") as f:
#     np.save(f, test_vecs)
# with open(f"y_test_{size}.npy", "wb") as f:
#     np.save(f, y_test)
# print("done")

# def get_ensemble():
#     classifiers = list()

#     #classifiers.append(("gb", GradientBoostingClassifier()))
#     classifiers.append(("rf", RandomForestClassifier(n_jobs=-1)))
#     classifiers.append(("svc", SGDClassifier(loss="log", penalty="l1")))
#     classifiers.append(("xgbrf", XGBRFClassifier(n_estimators = 1000, subsample = 0.9, tree_method = "gpu_hist", objective='binary:logistic', use_label_encoder=False)))
#     classifiers.append(("xgbc", XGBClassifier(n_estimators = 1000,  subsample = 0.9, tree_method = "gpu_hist", objective='binary:logistic', use_label_encoder=False)))
#     ensemble = VotingClassifier(estimators=classifiers, voting="soft", n_jobs=-1)
#     return ensemble

# def get_models():
#     classifiers = dict()

#     #classifiers["gb"] = GradientBoostingClassifier()
#     classifiers["rf"] =  RandomForestClassifier(n_jobs=-1)
#     classifiers["svc"] = SGDClassifier(loss="log", penalty="l1")
#     classifiers["xgbrf"] =  XGBRFClassifier(n_estimators = 1000, subsample = 0.9, tree_method = "gpu_hist", objective='binary:logistic', use_label_encoder=False)
#     classifiers["xgbc"] =  XGBClassifier(n_estimators = 1000,  subsample = 0.9, tree_method = "gpu_hist", objective='binary:logistic', use_label_encoder=False)
#     classifiers["ensemble"] = get_ensemble()
#     return classifiers

# from tqdm import tqdm
# models = get_models()
# preds, names = list(), list()
# for name, model in tqdm(models.items()):
#     print(name)
#     model.fit(train_vecs, y_train)
#     preds.append(model.predict(test_vecs))
#     names.append(name)

# fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(14,6))
# for y_pred, name, ax in zip(preds, names, axes.flatten()):
    
#     ConfusionMatrixDisplay.from_predictions(y_test, 
#                                             y_pred,
#                                             labels=[0,1],
#                                             cmap='Blues',
#                                             colorbar=False,
#                                             ax=ax)
#     ax.title.set_text(name + " --> score: " + "{:.4f}".format(accuracy_score(y_test, y_pred)))
# plt.tight_layout()  
# plt.savefig(f"ConfusionMatrix_{time.time()}.png")

# # results = []
# # models = get_models()
# # for name, model in tqdm(models.items()):
# #     scores = cross_val_score(model, np.concatenate((train_vecs, test_vecs)), np.concatenate((y_train, y_test)), scoring='accuracy', cv=5, n_jobs=-1)
# #     results.append(scores)
# #     print(name, ': %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
# # plt.boxplot(results, labels=names, showmeans=True)
# # plt.savefig("Boxplots.png")