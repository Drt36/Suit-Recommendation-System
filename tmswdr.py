#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


# In[2]:


datasetsuit = pd.read_csv("tmswdrsuit.csv")
datasetsuit.head()


# In[3]:


datasethalfsuit = pd.read_csv("tmswdrhalfsuit.csv")
datasethalfsuit.head()


# In[4]:


dataset3piecesuit = pd.read_csv("tmswdr3piecesuit.csv")
dataset3piecesuit.head()


# In[5]:


datasetsuit.drop(["Design_url2"],axis=1,inplace=True)


# In[6]:


datasetsuit.head()


# In[7]:


datasethalfsuit.drop(["Design_url2"],axis=1,inplace=True)
datasethalfsuit.head()


# In[8]:


dataset3piecesuit.drop(["Design_url2"],axis=1,inplace=True)
dataset3piecesuit.head()


# In[9]:


datasetsuit.shape


# In[10]:


datasethalfsuit.shape


# In[11]:


dataset3piecesuit.shape


# In[12]:


datasetsuit['all_content'] =datasetsuit['Design_title'] + datasetsuit['Design_Color'] + datasetsuit['Design_Style']


# In[13]:


datasethalfsuit['all_content'] =datasethalfsuit['Design_title'] + datasethalfsuit['Design_Color'] + datasethalfsuit['Design_Style']


# In[14]:


dataset3piecesuit['all_content'] =dataset3piecesuit['Design_title'] + dataset3piecesuit['Design_Color'] + dataset3piecesuit['Design_Style']


# In[15]:


vectorizer = TfidfVectorizer(analyzer='word')


# In[16]:


tfidf_all_contentsuit = vectorizer.fit_transform(datasetsuit['all_content'])


# In[17]:


tfidf_all_contenthalfsuit = vectorizer.fit_transform(datasethalfsuit['all_content'])


# In[18]:


tfidf_all_content3piecesuit = vectorizer.fit_transform(dataset3piecesuit['all_content'])


# In[19]:


tfidf_all_contentsuit.shape


# In[20]:


tfidf_all_contenthalfsuit.shape


# In[21]:


tfidf_all_content3piecesuit.shape


# In[22]:


cosine_similarity_all_contentsuit = linear_kernel(tfidf_all_contentsuit, tfidf_all_contentsuit)


# In[23]:


cosine_similarity_all_contenthalfsuit = linear_kernel(tfidf_all_contenthalfsuit, tfidf_all_contenthalfsuit)


# In[24]:


cosine_similarity_all_content3piecesuit = linear_kernel(tfidf_all_content3piecesuit, tfidf_all_content3piecesuit)


# In[25]:


indices_nsuit = pd.Series(datasetsuit['Design_code'])

indices_nhalfsuit = pd.Series(datasethalfsuit['Design_code'])

indices_n3piecesuit = pd.Series(dataset3piecesuit['Design_code'])


# In[26]:


inddictsuit = indices_nsuit.to_dict()

inddicthalfsuit = indices_nhalfsuit.to_dict()

inddict3piecesuit = indices_n3piecesuit.to_dict()


# In[27]:


inddictsuit = dict((v,k) for k,v in inddictsuit.items())

inddicthalfsuit = dict((v,k) for k,v in inddicthalfsuit.items())

inddict3piecesuit = dict((v,k) for k,v in inddict3piecesuit.items())


# In[28]:


def recommend_suit(Design_codein):
    Design_codein=int(Design_codein)
    id = inddictsuit[Design_codein]
    
    # Get the pairwise similarity scores of all designs compared that design,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_similarity_all_contentsuit[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    
    #Get the designs index
    design_index = [i[0] for i in similarity_scores]
    print(similarity_scores)
    
    #Return the top 5 most similar designs
    return datasetsuit.iloc[design_index].to_dict('records')


# In[29]:


def recommend_halfsuit(Design_codein):
    Design_codein=int(Design_codein)
    id = inddicthalfsuit[Design_codein]
    
    # Get the pairwise similarity scores of all designs compared that design,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_similarity_all_contenthalfsuit[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:9]
    
    #Get the designs index
    designs_index = [i[0] for i in similarity_scores]
    
    #Return the top 5 most similar designs
    return datasethalfsuit.iloc[designs_index].to_dict('records')


# In[30]:


def recommend_3piecesuit(Design_codein):
    Design_codein=int(Design_codein)
    id = inddict3piecesuit[Design_codein]
    
    # Get the pairwise similarity scores of all designs compared that design,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_similarity_all_content3piecesuit[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:9]
    
    #Get the designs index
    designs_index = [i[0] for i in similarity_scores]
    #Return the top 5 most similar designs
    return dataset3piecesuit.iloc[designs_index].to_dict('records')


# In[31]:


recommend_suit(3)


# In[32]:


recommend_halfsuit(1)


# In[33]:


recommend_3piecesuit(1)


# In[34]:


def sample_suit():
    dataset_frame=pd.DataFrame(datasetsuit)
    return dataset_frame.sample(n=4).to_dict('records')


# In[35]:


sample_suit()


# In[36]:


def sample_halfsuit():
    dataset_frame=pd.DataFrame(datasethalfsuit)
    return dataset_frame.sample(n=4).to_dict('records')


# In[37]:


sample_halfsuit()


# In[38]:


def sample_3piecesuit():
    dataset_frame=pd.DataFrame(dataset3piecesuit)
    return dataset_frame.sample(n=4).to_dict('records')


# In[39]:


sample_3piecesuit()

