import numpy as np
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline

def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]


def model(query):
    response = Pipe.predict([query])[0]
    return(response)

df = pd.read_csv('./dataset2.txt',sep='\t')
a = pd.Series(df.columns)
a = a.rename({0: df.columns[0],1: df.columns[1]})
b = {'Questions':'Hi','Answers':'hello'}
c = {'Questions':'Hello','Answers':'hi'}
d= {'Questions':'how are you','Answers':"i'm fine. how about yourself?"}
e= {'Questions':'how are you doing','Answers':"i'm fine. how about yourself?"}


df = df.append(a,ignore_index=True)
df.columns=['Questions','Answers']
df = df.append([b,c,d,e],ignore_index=True)
df = df.append(b,ignore_index=True)
df = df.append(c,ignore_index=True)
df = df.append(d,ignore_index=True)
df = df.append(e,ignore_index=True)
  
Pipe = Pipeline([
    ('bow',CountVectorizer(analyzer=cleaner)),
    ('tfidf',TfidfTransformer()),
    ('classifier',DecisionTreeClassifier())
])


Pipe.fit(df['Questions'],df['Answers'])

model()