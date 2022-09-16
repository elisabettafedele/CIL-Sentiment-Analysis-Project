from file_handler import *
from remove_duplicates import remove_duplicates
from hashtag_segment import *
from remove_repeated_letters import remove_repeated_letters
from remove_idx_test import remove_idx_test

# Choose your input/output paths for positive and negative tweets here
input_path_neg = 'path/to/neg/train.txt'
input_path_pos = 'path/to/pos/train.txt'
input_path_test = 'path/to/test.txt'

output_path_neg = 'path/to/output/neg.txt'
output_path_pos = 'path/to/output/pos.txt'
output_path_test = 'path/to/output/test.txt'

print("1. Processing training set...")
# Tweets loading
neg = load_tweets(input_path_neg)
pos = load_tweets(input_path_pos)

tweets = np.concatenate((neg, pos))
labels = np.concatenate((np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)))

# Step 1: remove duplicates
print("Step 1/3: removing duplicates...")
pos, neg = remove_duplicates(tweets, labels)
# Step 2: hashtags segmenting
print("Step 2/3: segmenting hashtags...")
neg = process_hashtags(neg)
pos = process_hashtags(pos)

# Step 3: remove repeated letters
print("Step 3/3: removing repeated letters...")
neg = remove_repeated_letters(neg)
pos = remove_repeated_letters(pos)

# Write the processed tweets
write_processed_tweets(neg, output_path_neg)
write_processed_tweets(pos, output_path_pos)
print("Training tweets preprocessing ended, you can find:"
      "\n- negative tweets in " + output_path_neg +
      "\n- positive tweets in " + output_path_pos)

print("\n2. Processing test set...")
test = load_tweets(input_path_test)

# Step 1: remove indices
print("Step 1/3: removing indices...")
test = remove_idx_test(test)

# Step 2: hashtags segmenting
print("Step 2/3: segmenting hashtags...")
test = process_hashtags(test)

# Step 3: remove repeated letters
print("Step 3/3: removing repeated letters...")
test = remove_repeated_letters(test)

write_processed_tweets(test, output_path_test)
print("Test tweets preprocessing ended, you can find the processed test tweets in " + output_path_test)
