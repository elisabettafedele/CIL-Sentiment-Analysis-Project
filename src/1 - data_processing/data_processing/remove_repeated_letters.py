import re

def replace(string):
    string = re.sub("([a-z])\\1{3,}", "\\1\\1\\1", string)
    return string

def remove_repeated_letters(tweets):
  processed_tweets = []
  for tweet in tweets:
    processed_tweets.append(replace(tweet))
  return processed_tweets