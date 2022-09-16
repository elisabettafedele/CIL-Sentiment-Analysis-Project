import pandas as pd
import numpy as np

def remove_duplicates(tweets, labels):
    train = pd.DataFrame({'tweet':tweets, 'label':labels})

    duplicated = train.groupby(['tweet', 'label']).size().reset_index(name = 'counts')

    #Drop tweets that appear the same number of time in pos and in neg
    no_diff_label = duplicated.drop_duplicates(['tweet','counts'], keep = False)

    #Keep only tweet with the highest value of counts
    no_dup_diff_label = no_diff_label.drop_duplicates(['tweet'], keep = 'first')
    neg_pd = no_dup_diff_label[no_dup_diff_label['label'] == 0]
    pos_pd = no_dup_diff_label[no_dup_diff_label['label'] == 1]

    neg_np = np.array(neg_pd['tweet'])
    pos_np = np.array(pos_pd['tweet'])

    return neg_np, pos_np

