
from spacy.lang.en import English
import re
import demoji
import json
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from sklearn.metrics import f1_score
import torch

demoji.download_codes()
tok = English()
stemmer = SnowballStemmer(language="english")


def epoch_time(start_time, end_time):
    """
    Returns the difference between two given time 
    in minutes
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def clean_tweet(tweet):
    
    # Remove usernames, "RT" and Hash
    tweet = re.sub(r"(RT|[@*])(\w*)", " ", tweet)
    # Hashtags are very useful. It gives context to the tweet.
    # Remove links in tweets
    tweet = re.sub(r"http\S+", " ", tweet)
    # We remove "#" and keep the tags
    tweet = re.sub(
        r"(\\n)|(\#)|(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])",
        "",
        tweet,
    )
    tweet = re.sub(r"(<br\s*/><br\s*/>)|(\-)|(\/)", " ", tweet)
    # convert to lower case
    tweet = re.sub(r"[^a-zA-Z0-9]", " ", tweet.lower())  # Convert to lower case
    # Tweets are usually full of emojis. We need to remove them.
    tweet = demoji.replace(tweet, repl="")
    # Stop words don't meaning to tweets. They can be removed
    tweet_words = tok(tweet)
    clean_tweets = []
    for word in tweet_words:
        if word.is_stop == False and len(word) > 1:
            clean_tweets.append(word.text.strip())

    tweet = " ".join(clean_tweets)

    return tweet

def stem_tweet(tweet):
    tokenized_tweets = []
    doc = tweet.split()  # Tokenize tweet
    for word in doc:
        word = stemmer.stem(word)  # Stem word
        tokenized_tweets.append(word)
    return tokenized_tweets

def save_dict(filename, data):
    json.dump(data, open(filename, "w"))

def encode_sentence(text, vocab2index, N=75):
  tokenized = stem_tweet(text)
  encoded = np.zeros(N, dtype=int)
  enc = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
  length = min(N, len(enc))
  encoded[:length] = enc[:length]
  return encoded, length


def index2word(tensor, vocab2index):
  """
    It takes a tensor argument and contructs back to a text
  """
  index2word = {index:word for word,index in vocab2index.items()}
  data = tensor.numpy()
  words_list = []
  for row in data:
    for cell in row:
      word = index2word[cell]
      words_list.append(word)

  word = " ".join(words_list)
  return word

def count_trainable_params(model):
  """
   Print a string of trainable parameters in a model
  Args:
      model (object): The model object
  """
  n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  print(f'The model has {n_params} trainable paramters')

  def cal_accuracy(preds, y):
    """
  Returns accuracy per batch. i.e 6 out of 10 means 0.6
  """
    preds = torch.round(preds)  # Round prediction up i.e 0.0 and 1.0
    correct = preds == y
    correct = correct.float()
    # print(f'correct: {correct.sum()}, wrong {len(correct)}')
    acc = correct.sum() / len(correct)
    return acc


def convert_pred(y_preds, y_true):
    """
    Round up predicted values to 0.0 or 1.0
    """
    y_preds = torch.round(y_preds)  # Round prediction up i.e 0.0 and 1.0

    return y_preds, y_true


def cal_f1_score(y_pred, y_true):
    """
    Returns an f1 score 
  """
    y_pred, y_true = convert_pred(y_pred, y_true)
    f1 = f1_score(y_pred, y_true)

    return f1