# Text Sentiment Classification
## Computational Intelligence Lab 2022 - ETHZ

Implementation of the sentiment analysis project for the course Computational Intelligence Lab @ ETH Zurich, Spring 2022. Please find the detailed description in the [Project Report](https://github.com/notchla/CIL/blob/submission/CIL_2022_Report.pdf)

## Instructions to run the code

All the code can be found in the corresponding folder in the **src** folder. To install the dependencies run:

```console
$ pip install requirements.txt
```

### 1 - Data Preprocessing

Change the data_path variable in the code to the path of the folder where the dataset files are. Then running
```console
$ python pre_process.py
```
to obtain the full pre processing.
After these steps, you will find three .txt files in the desired output directory:
- neg_processed.txt: processed negative tweets
- pos_processed.txt: processed positive tweets
- test_processed.txt: processed test tweets

### 2 - Word2Vec

Change the paths at the start of the file **word2vec_model.py**, then run:

```console
$ python word2vec_model.py
```

To generate the embeddings. Then run:

```console
$ python classifier.py
```

To load the generated the embeddings and perform classification.


### 3 - DV-ngram-cosine

How to run:
Change the data_path variable in the code to the path of the folder where the dataset files are. Then running
```console
$ python dv-ngrams-cosine.py
```
will generate the embeddings. To get the test predictions, run:
```console
$ python classifier.py
```

### 4 - TFIDF

Change the paths in the second cell of the notebook (model path refers to the directory in which the classification model will be saved). Then run the notebook.

### 5 - LSTM 

In the LSTM folder you can find a notebook in which you can choose the desired data paths and the type of LSTM (vanilla, bidirectional, stacked)
you wish to implement.
After that, you can simply run all the cells and, once the training is finished, the prediction will be saved in the desired path.
Also, there are 2 figures provided after each training, one for the training and validation loss over epochs and one for the training and validation accuracy.

### 6 - FastText

Change the data_path variable in the code to the path of the folder where the dataset files are. Then running
```console
$ python fasttext.py
```
will generate the test predictions.

### 7 - Bert

This notebook needs a specific data directory structure to run. First create a directory as follows:

```console
├── dataset_directory
│   ├── train
│   │   ├── neg
│   │   │   ├── train_neg.txt
│   │   ├── pos
│   │   │   └── train_pos.txt
└───└── test
        └── text.txt
```

Once this directory structure is created, run the following commands:

```console
$ cd dataset_directory/train/neg
$ split train_neg.txt -l 1 --verbose --additional-suffix=.txt
$ rm train_neg.txt
```

```console
$ cd dataset_directory/train/pos
$ split train_pos.txt -l 1 --verbose --additional-suffix=.txt
$ rm train_pos.txt
```

```console
$ cd dataset_directory/test
$ split test.txt -l 1 --verbose --additional-suffix=.txt
$ rm test.txt
```

Now you are ready to run the notebook: change the paths in the second cell of the notebook and run it.

### 8 - BERTweet

To train a model using **bert.py** first generate the directory as described in the previous step. Then run:

```console
$ python bert.py -args args
```

Run 

```console
$ python bert.py --help
```

to understand the arguments to pass to the program. To use the BertTweet model please use the flag `-m vinai/bertweet-base`. This program will generate the .csv with the predictions for the test data and will save the trained model.

To save the prediction probabilites of a model run:
 
 ```console
$ python load_checkpoint.py
```

(change the paths for the checkpoint and the name of the prediction file to save). You can then combine multiple predictions by runnning

 ```console
$ python make_enseble.py
```

(set the names of the precitions to use in the **predictions** list)

## Instructions to obtain the best submission 
Train two BERTweet models:
- model 1: use the instructions given in section 8 with not pre-processed data
- model 2: use the instructions given in section 8 with pre-processed data, obtained using the instructions of section 1

Follow the instructions above to generate the predictions of the ensemble of these two models.
