import numpy as np

def load_tweets(filename):
    tweets = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            tweets.append(line.rstrip())
    print(f'{len(tweets)} tweets loaded')
    return np.array(tweets)

def write_processed_tweets(tweets, output_tweets_path):
    with open(output_tweets_path, 'w') as f:
        for tweet in tweets:
            f.write(tweet+'\n')