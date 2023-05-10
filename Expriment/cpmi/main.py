import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
import swifter
import re
import string
import pandas as pd

stop_words = list(set(stopwords.words('english')))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[{}0-9]'.format(string.punctuation), ' ', text)
    text=re.sub(r'[^A-Za-z0-9 ]+', ' ', text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stop_words]
    text = [WordNetLemmatizer().lemmatize(word) for word in text]
    text = ' '.join(text)
    return text

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
df=pd.DataFrame({"content":newsgroups["data"]})

df["content"]=df["content"].swifter.apply(lambda x: preprocess_text(x))
df['content_length'] = df['content'].str.len()

df = df[df['content_length'] > 100]
df = df[df['content_length'] < 2000]

df=df[["content"]].reset_index(drop=True).reset_index().rename(columns={"index":"id"})
documents_20newsgroup=df.content.to_list()


l_chunck=15
sentences=[sentence.strip().lower() for sentence in documents_20newsgroup]
sentences  = ["".join([char for char in text if char not in string.punctuation]) for text in sentences]
sentences = [sentence.split(' ') for sentence in sentences]
sentences=[[token for token in sentence if len(token)>0] for sentence in sentences]
sentences=[sentence for sentence in sentences if len(sentence) > 5]
sentences = [sentence[i:i+l_chunck] for sentence in sentences for i in range(0, len(sentence), l_chunck) ]
sentences=[sentence for sentence in sentences if len(sentence) > 5]

import pickle
with open("sentences_20", "wb") as fp:
  pickle.dump(sentences, fp)

import torch
import txt_to_pmi
DEVICE = torch.device('mps')
MODEL = txt_to_pmi.languagemodel.BERT(DEVICE, 'bert-base-cased', 128)
outs = txt_to_pmi.get_cpmi(MODEL, sentences, verbose=False)


with open("outs_20", "wb") as fp:   #Pickling
  pickle.dump(outs, fp)
