#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


# In[18]:


dataset = pd.read_csv("tmswdr.csv")
dataset.head()


# In[19]:


datasethalfsuit = pd.read_csv("tmswdrhalfsuit.csv")
datasethalfsuit.head()


# In[20]:


dataset3piecesuit = pd.read_csv("tmswdr3piecesuit.csv")
dataset3piecesuit.head()


# In[3]:


dataset.drop(["Design_url2"],axis=1,inplace=True)


# In[4]:


dataset.head()


# In[21]:


datasethalfsuit.drop(["Design_url2"],axis=1,inplace=True)
datasethalfsuit.head()


# In[22]:


dataset3piecesuit.drop(["Design_url2"],axis=1,inplace=True)
dataset3piecesuit.head()


# In[5]:


dataset.shape


# In[23]:


datasethalfsuit.shape


# In[24]:


dataset3piecesuit.shape


# In[6]:


dataset['all_content'] =dataset['Design_title'] + dataset['Design_Color'] + dataset['Design_Style']


# In[25]:


datasethalfsuit['all_content'] =datasethalfsuit['Design_title'] + datasethalfsuit['Design_Color'] + datasethalfsuit['Design_Style']


# In[26]:


dataset3piecesuit['all_content'] =dataset3piecesuit['Design_title'] + dataset3piecesuit['Design_Color'] + dataset3piecesuit['Design_Style']


# In[27]:


vectorizer = TfidfVectorizer(analyzer='word')


# In[8]:


tfidf_all_content = vectorizer.fit_transform(dataset['all_content'])


# In[28]:


tfidf_all_contenthalfsuit = vectorizer.fit_transform(datasethalfsuit['all_content'])


# In[29]:


tfidf_all_content3piecesuit = vectorizer.fit_transform(dataset3piecesuit['all_content'])


# In[9]:


tfidf_all_content.shape


# In[30]:


tfidf_all_contenthalfsuit.shape


# In[31]:


tfidf_all_content3piecesuit.shape


# In[10]:


cosine_similarity_all_content = linear_kernel(tfidf_all_content, tfidf_all_content)


# In[32]:


cosine_similarity_all_contenthalfsuit = linear_kernel(tfidf_all_contenthalfsuit, tfidf_all_contenthalfsuit)


# In[33]:


cosine_similarity_all_content3piecesuit = linear_kernel(tfidf_all_content3piecesuit, tfidf_all_content3piecesuit)


# In[34]:


indices_n = pd.Series(dataset['Design_code'])

indices_nhalfsuit = pd.Series(datasethalfsuit['Design_code'])

indices_n3piecesuit = pd.Series(dataset3piecesuit['Design_code'])


# In[36]:


inddict = indices_n.to_dict()

inddicthalfsuit = indices_nhalfsuit.to_dict()

inddict3piecesuit = indices_n3piecesuit.to_dict()


# In[39]:


inddict = dict((v,k) for k,v in inddict.items())

inddicthalfsuit = dict((v,k) for k,v in inddicthalfsuit.items())

inddict3piecesuit = dict((v,k) for k,v in inddict3piecesuit.items())


# In[14]:


def recommend_cosine(Design_code):
    Design_code=int(Design_code)
    
    id = inddict[Design_code]
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_similarity_all_content[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    
    #Get the books index
    books_index = [i[0] for i in similarity_scores]
    print(similarity_scores)
    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return dataset.iloc[books_index].to_dict('records')
    #return dataset.to_dict('books_index')


# In[40]:


def recommend_halfsuit(Design_codein):
    Design_codein=int(Design_codein)
    id = inddicthalfsuit[Design_codein]
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 
    similarity_scores = list(enumerate(cosine_similarity_all_contenthalfsuit[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:9]
    
    #Get the designs index
    designs_index = [i[0] for i in similarity_scores]
    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return datasethalfsuit.iloc[designs_index].to_dict('records')


# In[41]:


def recommend_3piecesuit(Design_codein):
    Design_codein=int(Design_codein)
    id = inddict3piecesuit[Design_codein]
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 8
    similarity_scores = list(enumerate(cosine_similarity_all_content3piecesuit[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:9]
    
    #Get the designs index
    designs_index = [i[0] for i in similarity_scores]
    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return dataset3piecesuit.iloc[designs_index].to_dict('records')


# In[51]:


def sample_row():
    dataset_frame=pd.DataFrame(dataset)
    return dataset_frame.sample(n=4).to_dict('records')




# In[47]:


def sample_halfsuit():
    dataset_frame=pd.DataFrame(datasethalfsuit)
    return dataset_frame.sample(n=4).to_dict('records')



# In[49]:


def sample_3piecesuit():
    dataset_frame=pd.DataFrame(dataset3piecesuit)
    return dataset_frame.sample(n=4).to_dict('records')
   
