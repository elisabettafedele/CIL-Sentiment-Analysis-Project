import numpy as np

def remove_idx_test(tweets):
    procs = []
    for tweet in tweets:
        procs.append(",".join(tweet.split(",")[1:]))
    return np.array(procs)
