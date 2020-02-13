#!/usr/bin/env python
# coding: utf-8

# In[34]:


## without NLTK
import re
from collections import defaultdict
def wordCount(text):
    count = defaultdict(lambda : 0)
    text = text.translate({ord(';'):None,ord(','):None,ord('.'):None,ord("'"):None,ord('"'):None,ord('\\'):None,ord('/'):None})
    text = text.split(" ")
    for word in text:
        count[word.lower()] = count[word.lower()]+1
    return (count)
        
   
    
def inputText():
    with open('WordCountInput.txt','r') as w:
        text = w.read()
        count = wordCount(text)
        w.close()
    return count
    
        
x = inputText()
for k,v in x.items():
    print(k+"   :   "+str(v))
    


# In[41]:


get_ipython().system('pip install nltk')
nltk.download('punkt')


# In[46]:


#with NLTK
from nltk import word_tokenize
from nltk import FreqDist

def wordCount(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    fdist = FreqDist(words)
    return fdist.most_common()
    

def inputText():
    with open('WordCountInput.txt','r') as w:
        text = w.read()
        count = wordCount(text)
        w.close()
    return count

inputText()

