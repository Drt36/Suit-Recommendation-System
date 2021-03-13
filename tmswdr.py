
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


dataset = pd.read_csv("tmswdr.csv")
dataset.head()

datasethalfsuit = pd.read_csv("tmswdrhalfsuit.csv")
datasethalfsuit.head()

dataset3piecesuit = pd.read_csv("tmswdr3piecesuit.csv")
dataset3piecesuit.head()


dataset.drop(["Design_url2"],axis=1,inplace=True)
dataset.head()

datasethalfsuit.drop(["Design_url2"],axis=1,inplace=True)
datasethalfsuit.head()

dataset3piecesuit.drop(["Design_url2"],axis=1,inplace=True)
dataset3piecesuit.head()


dataset.shape
datasethalfsuit.shape
dataset3piecesuit.shape


dataset['all_content'] =dataset['Design_title'] + dataset['Design_Color'] + dataset['Design_Style']

datasethalfsuit['all_content'] =datasethalfsuit['Design_title'] + datasethalfsuit['Design_Color'] + datasethalfsuit['Design_Style']

dataset3piecesuit['all_content'] =dataset3piecesuit['Design_title'] + dataset3piecesuit['Design_Color'] + dataset3piecesuit['Design_Style']


vectorizer = TfidfVectorizer(analyzer='word')



tfidf_all_content = vectorizer.fit_transform(dataset['all_content'])

tfidf_all_contenthalfsuit = vectorizer.fit_transform(datasethalfsuit['all_content'])

tfidf_all_content3piecesuit = vectorizer.fit_transform(dataset3piecesuit['all_content'])




tfidf_all_content.shape
tfidf_all_contenthalfsuit.shape
tfidf_all_content3piecesuit.shape



cosine_similarity_all_content = linear_kernel(tfidf_all_content, tfidf_all_content)

cosine_similarity_all_contenthalfsuit = linear_kernel(tfidf_all_contenthalfsuit, tfidf_all_contenthalfsuit)

cosine_similarity_all_content3piecesuit = linear_kernel(tfidf_all_content3piecesuit, tfidf_all_content3piecesuit)



indices_n = pd.Series(dataset['Design_code'])

indices_nhalfsuit = pd.Series(datasethalfsuit['Design_code'])

indices_n3piecesuit = pd.Series(dataset3piecesuit['Design_code'])


inddict = indices_n.to_dict()

inddicthalfsuit = indices_nhalfsuit.to_dict()

inddict3piecesuit = indices_n3piecesuit.to_dict()


inddict = dict((v,k) for k,v in inddict.items())

inddicthalfsuit = dict((v,k) for k,v in inddicthalfsuit.items())

inddict3piecesuit = dict((v,k) for k,v in inddict3piecesuit.items())



def recommend_cosine(Design_codein):
    Design_codein=int(Design_codein)
    id = inddict[Design_codein]
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 
    similarity_scores = list(enumerate(cosine_similarity_all_content[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:9]
    
    #Get the designs index
    designs_index = [i[0] for i in similarity_scores]
    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return dataset.iloc[designs_index].to_dict('records')

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

def sample_row():
    dataset_frame=pd.DataFrame(dataset)
    return dataset_frame.sample(n=4).to_dict('records')

def sample_halfsuit():
    dataset_frame=pd.DataFrame(datasethalfsuit)
    return dataset_frame.sample(n=4).to_dict('records')

def sample_3piecesuit():
    dataset_frame=pd.DataFrame(dataset3piecesuit)
    return dataset_frame.sample(n=4).to_dict('records')
