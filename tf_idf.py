import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

import re
import string

import nltk
nltk.download('stopwords')
from sklearn.utils import shuffle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words('english')

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

with open('Assets/Data/test.json', 'r') as file:
    data = json.load(file)

f_data = {}
for k, v in data.items():
    f_data[k] = v[:95]

processed_data = {
    'text': [], 'target': []
}
for k, v in f_data.items():
    processed_data['text'] += v
    processed_data['target'] += [k] * len(v)

df = pd.DataFrame.from_dict(processed_data)

# X_train, X_test, y_train, y_test = train_test_split(df['text'].values, df['target'].values, test_size=0,
#                                                    random_state=123, stratify=df['target'].values)

X_train = df['text'].values
y_train = df['target'].values
X_train, y_train = shuffle(X_train, y_train)
with open('Assets/Outputs/test_shuffle.txt', 'w') as file:
    for item in X_train.tolist():
        file.write(item + "\n")

with open('Assets/Outputs/y_test_shuffle.txt', 'w') as file:
    for item in y_train.tolist():
        file.write(item + "\n")
y_test = y_train
X_test = X_train
tfidf_vectorizer = TfidfVectorizer()

tfidf_train_vectors = tfidf_vectorizer.fit_transform(X_train)

tfidf_test_vectors = tfidf_vectorizer.transform(X_test)

classifier = RandomForestClassifier()

classifier.fit(tfidf_train_vectors, y_train)

y_pred = classifier.predict(tfidf_test_vectors)


with open('Assets/Outputs/y_pred_shuffle.txt', 'w') as file:
    for item in y_pred.tolist():
        file.write(item + "\n")

print(classification_report(y_test, y_pred))
