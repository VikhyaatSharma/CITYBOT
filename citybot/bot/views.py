from ast import Not
from multiprocessing import context
from django.shortcuts import render, redirect

import numpy as np
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.pipeline import Pipeline
from django.contrib.staticfiles import finders
from django.http import JsonResponse
from django.core import serializers
from django.http import HttpResponse

def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]


def model(query):
    response = Pipe.predict([query])[0]
    return(response)

df = pd.read_csv('static/dialogs.txt',sep='\t')
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

context={}

def index(request):
    if(request.method == 'POST'):
        if(request.POST.get('email') == 'test@test.com' and request.POST.get('pass') == '123'):
            email = (request.POST.get('email'))
            city = request.POST.get('city')
            global context
            context = { 'email': email, 'city':city}
            return redirect('home')
    return render(request, "index.html")

def home(request):
    if(request.method == 'POST'):
        mess=request.POST.get('usermess')
        res=(bot(mess))
        return HttpResponse(res)
    return render(request, "home.html", context)

def bot(param):
    res = model(param)
    return res