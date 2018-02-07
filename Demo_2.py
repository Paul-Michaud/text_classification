!rm synopses_list_imdb.txt
!wget https://raw.githubusercontent.com/brandomr/document_cluster/master/synopses_list_imdb.txt

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import nltk
import re
import nltk.stem

class StemmedTfidfVectorizer(TfidfVectorizer): # hérite de TfidVectorizer
    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))
     
    
    
stemmer=nltk.stem.SnowballStemmer('english')
  
synopses = open('synopses_list_imdb.txt').read().split('\n BREAKS HERE')


synopses_clean = []
for text in synopses:
    text = BeautifulSoup(text, 'html.parser').getText()
    #strips html formatting and converts to unicode
    synopses_clean.append(text)
    
stemmer=nltk.stem.SnowballStemmer('english')

vectorizer = StemmedTfidfVectorizer(max_df=0.7, min_df=0.3,  ngram_range=(1,3), stop_words='english')
X = vectorizer.fit_transform(synopses)
 
true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=20, n_init=1)
model.fit(X)



plt.hist(model.labels_, bins=true_k)
plt.show()
cluster_names=[]

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    names=[]
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind]),
        names.append(terms[ind])
    cluster_names.append(names)
    print
    

