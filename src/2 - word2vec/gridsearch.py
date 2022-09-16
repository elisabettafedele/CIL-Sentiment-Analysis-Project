from gensim.models.word2vec import Word2Vec
from matplotlib.pyplot import sca
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.preprocessing import scale
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBRFClassifier, XGBClassifier
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import GridSearchCV

size = 1000

with open(f"x_train_{size}.npy", "rb") as f:
    x_train = np.load(f)

with open(f"y_train_{size}.npy", "rb") as f:
    y_train = np.load(f)

with open(f"x_test_{size}.npy", "rb") as f:
    x_test = np.load(f)

with open(f"y_test_{size}.npy", "rb") as f:
    y_test = np.load(f)


classifier = XGBRFClassifier()

print(classifier.get_params().keys())

# param_grid = {"n_estimators" : [10, 100, 200], "max_depth" : [10, 100, None], "min_samples_split" : [2, 5], "min_samples_leaf": [1, 2, 3], "n_jobs" : [8]}

#next max_depth, colsample_bylevel, lamda, alpha 
param_grid = {"n_estimators" : [100], "max_depth" : [10], "learning_rate" : [0.1], "min_child_weight" : [1], "gamma" : [0], "subsample" : [0.7], "colsample_bytree" : [0.6], "reg_alpha" : [1e-5]}

grid = GridSearchCV(XGBRFClassifier(tree_method = "gpu_hist", objective = 'binary:logistic', use_label_encoder =  False, n_jobs = 12), param_grid=param_grid, cv=5, n_jobs=1, verbose=2, scoring="accuracy")

grid.fit(np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)))

print(grid.best_params_)

cvres = grid.cv_results_

print
for score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(score, params)

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
#     model.fit(x_train, y_train)
#     preds.append(model.predict(x_test))
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