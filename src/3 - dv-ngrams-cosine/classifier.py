import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

embeddings_file = 'dv-ngrams-cosine-embeddings.txt'
n_pos = 100000
n_neg = 100000


# load embedddings

embeddings = np.loadtxt('dv-ngrams-cosine-embeddings.txt')
X_train = embeddings[0:n_pos + n_neg]
X_test = embeddings[n_pos + n_neg:]
y_train = np.asarray([1 if i < n_pos else -1 for i in range(n_pos + n_neg)]).transpose()


# preprocessing

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0, random_state=1)


# model fitting

clf = LogisticRegression(max_iter = 1000, verbose=4, C=1e5).fit(X_train,y_train)


# predict and save test labels

res = clf.predict(X_test)

with open('submission.csv', 'w') as txtoutfile:
    txtoutfile.write('Id,Prediction\n')
    for i, pred in np.ndenumerate(res):
        txtoutfile.write(str(i+1) + ',' + str(pred) + '\n')
