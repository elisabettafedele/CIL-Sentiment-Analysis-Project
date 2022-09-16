import numpy as np
import re
import fasttext

data_path = 'twitter-datasets/'
output_path = ''

# load tweets

train_tweets = []
train_labels = []

def load_tweets(filename, label):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            train_tweets.append(line.rstrip())
            train_labels.append(label)
    
load_tweets(data_path + 'train_neg_full.txt', 0)
load_tweets(data_path + 'train_pos_full.txt', 1)

# Convert to NumPy array to facilitate indexing
train_tweets = np.array(train_tweets)
train_labels = np.array(train_labels)

print(f'{len(train_tweets)} tweets loaded')


# preprocessing

for i, tweet in enumerate(train_tweets):
    tweet = ' '.join(re.sub("[\.\(\)\,\!\?\:\;\-\=]", " ", tweet).split())
    tweet = tweet.lower()
    train_tweets[i] = '__label__' + str(train_labels[i]) + ' ' + tweet
 
 
# prepare training file

np.random.seed(1)
np.random.shuffle(train_tweets)
train_input_file = 'train_input_file'
with open(output_path + train_input_file, 'w') as txtoutfile:
    for row in train_tweets:
        txtoutfile.write(row + '\n' )
        

# train model

hyper_params = {"lr": 0.01,
                "epoch": 20,
                "wordNgrams": 2,
                "dim": 50,
                "bucket": 200000}

model = fasttext.train_supervised(input=output_path + train_input_file, **hyper_params)
print("Model trained with the hyperparameter \n {}".format(hyper_params))


# testing

predictions = []
with open(data_path + 'test_data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        line = re.sub("[\.\(\)\,\!\?\:\;\-\=]", " ", line).split()
        line = line[1:]
        line = ' '.join(line).lower()
        predictions.append(-1 if '0' in model.predict(line)[0][0] else 1)
        

# save test predictions

test_output_file = 'submission.csv'
with open(output_path + test_output_file, 'w') as txtoutfile:
    txtoutfile.write('Id,Prediction\n')
    for i, pred in enumerate(predictions):
        txtoutfile.write(str(i+1) + ',' + str(pred) + '\n')