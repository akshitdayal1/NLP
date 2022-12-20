#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip3 install torch torchvision torchaudio')



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re

import pandas as pd
import numpy as np

# # Instantiate Model




tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

tokens = tokenizer.encode('i am probably gonna study today', return_tensors='pt')
tokens


#return_tensors



tokens[0]

tokens.size()

tokenizer.decode(tokens[0])

result=model(tokens)

result

result.logits



#this will generate a sentiment score b/w 1-10


int(torch.argmax(result.logits))+1


# # COLLECT REVIEWS: Scraping yelp to extract reviews



r =  requests.get('https://www.yelp.com/biz/biang-biang-noodles-seattle-2?osq=Restaurants')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment*.')

regex

#find all the para and filter with class of type regex prev defined
results= soup.find_all('p', {'class':regex})

results[0].text


# scrape text output in an array reviews
reviews = [rev.text for rev in results]



reviews


# # Load reviews into a DF and score




# In[51]:


df = pd.DataFrame(np.array(reviews), columns=['Reviews'])


# In[52]:


df['Reviews'].iloc[0]



# method to calculate sentiment score

def sentiment_score(review):
    tokens= tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1
    

#limited to 512 tokens
df['Sentiment']= df['Reviews'].apply(lambda x: sentiment_score(x[:512]))


df


df.Reviews[10]




