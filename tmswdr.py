

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


dataset = pd.read_csv("tmswdr.csv")
dataset.head()

dataset.drop(["Design_url2"],axis=1,inplace=True)

dataset.head()


dataset.shape


dataset['all_content'] =dataset['Design_title'] + dataset['Design_Color'] + dataset['Design_Style']


vectorizer = TfidfVectorizer(analyzer='word')



tfidf_all_content = vectorizer.fit_transform(dataset['all_content'])




tfidf_all_content.shape



cosine_similarity_all_content = linear_kernel(tfidf_all_content, tfidf_all_content)



indices_n = pd.Series(dataset['Design_code'])


inddict = indices_n.to_dict()


inddict = dict((v,k) for k,v in inddict.items())


def recommend_cosine(Design_codein):
    Design_codein=int(Design_codein)
    id = inddict[Design_codein]
    # Get the pairwise similarity scores of all books compared that book,
    # sorting them and getting top 5
    similarity_scores = list(enumerate(cosine_similarity_all_content[id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:6]
    
    #Get the books index
    books_index = [i[0] for i in similarity_scores]
    #Return the top 5 most similar books using integar-location based indexing (iloc)
    return dataset.iloc[books_index].to_dict('records')
 
def sample_row():
    dataset_frame=pd.DataFrame(dataset)
    return dataset_frame.sample(n=3).to_dict('records')

