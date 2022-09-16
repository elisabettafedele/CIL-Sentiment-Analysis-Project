from gensim.models.word2vec import Word2Vec
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

with open("twitter-datasets/train_pos.txt","r") as infile:
    pos_tweets = infile.readlines()

with open("twitter-datasets/train_neg.txt","r") as infile:
    neg_tweets = infile.readlines()

y = np.concatenate((np.ones(len(pos_tweets)), np.zeros(len(neg_tweets))))

tweets = np.concatenate((pos_tweets, neg_tweets))

x_train, x_test, y_train, y_test = train_test_split(tweets, y, test_size=0.2)

def cleanText(corpus):
    corpus = [z.lower().replace("\n", "").split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)

size = 1000
window = 3
min_count = 1
workers = 16
sg = 1

print("creating w2v")
twitter_w2v = Word2Vec(sentences=x_train, vector_size=size, min_count=min_count, window=window, workers=workers, sg=sg)
print("done")

word2vec_model_file = f"word2vec_{str(size)}.model"
print("saving model to {}".format(word2vec_model_file))
twitter_w2v.save(word2vec_model_file)

def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in text:
        try:
            vec += twitter_w2v.wv[word].reshape((1, size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

print("generating train_vecs")
train_vecs = np.concatenate([buildWordVector(z, size) for z in x_train])
train_vecs = scale(train_vecs)
with open(f"x_train_{size}.npy", "wb") as f:
    np.save(f, train_vecs)
with open(f"y_train_{size}.npy", "wb") as f:
    np.save(f, y_train)
print("done")

print("generating test_vecs")
test_vecs = np.concatenate([buildWordVector(z, size) for z in x_test])
test_vecs = scale(test_vecs)
with open(f"x_test_{size}.npy", "wb") as f:
    np.save(f, test_vecs)
with open(f"y_test_{size}.npy", "wb") as f:
    np.save(f, y_test)
print("done")

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