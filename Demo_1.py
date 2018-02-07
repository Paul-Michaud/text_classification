!rm jamesbond.txt
!rm starwars.txt

!wget https://gist.githubusercontent.com/Paul-Michaud/efc51212e7119218134875c4027fb2d9/raw/6717bee0d587ed49f7ac3a37ecbb039e2bf06a1d/jamesbond.txt
!wget https://gist.githubusercontent.com/Paul-Michaud/d115f2e18b23dbc492e7743c1ea8d28e/raw/2a24c8612f8755dff7572bb83807c3ed35079678/starwars.txt

from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nltk.stem


class StemmedTfidfVectorizer(TfidfVectorizer): # hérite de TfidVectorizer
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))

      
synopses_jamesbond = open('jamesbond.txt').read().split('\nNEXT')
synopse_starwars = open('starwars.txt').read().split('\nNEXT')

synopses_jamesbond_clean = []
for text in synopses_jamesbond:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_jamesbond_clean.append(text)
    
synopses_starwars_clean = []
for text in synopse_starwars:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_starwars_clean.append(text)
    

synopses = synopses_starwars_clean + synopses_jamesbond_clean
stemmer=nltk.stem.SnowballStemmer('english')

vectorizer = StemmedTfidfVectorizer(max_df=0.8, min_df=0.2,stop_words='english')
X = vectorizer.fit_transform(synopses)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
cluster_names=[]
for i in range(true_k):
    print("Cluster %d:" % i),
    names=[]
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind]),
        names.append(terms[ind])
    else:
        print
    cluster_names.append(names)
    

print("\n")
print("Prediction")
 
Y = vectorizer.transform(["""About 30 years after the destruction of the Death 
                          Star II, Luke Skywalker has vanished following the 
                          demise of the new Jedi Order he was attempting to 
                          build."""])
                          
prediction = model.predict(Y)
print(prediction)
 
Y = vectorizer.transform(["""After an operation in Istanbul ends in disaster, 
                          Bond is missing and presumed to be dead. In the 
                          aftermath, questions are raised over M's ability to 
                          run the Secret Service, and she becomes the subject 
                          of a government review over her handling of the 
                          situation."""])

prediction = model.predict(Y)
print(prediction)
