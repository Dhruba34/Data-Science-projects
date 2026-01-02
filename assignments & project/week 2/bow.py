from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class FeatureExtractor:
    def __init__(self, max_features=10000, ngram_range=(1, 2)):
        """
        Initialize vectorizers.
        
        Parameters:
        - max_features: Keep top 5000 most frequent words
        - ngram_range: (1,2) means unigrams + bigrams
          Example: "profit increase" → 
            unigrams: ['profit', 'increase']
            bigrams: ['profit increase']
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.bow_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,           # Ignore words in <2 docs
            max_df=0.8,         # Ignore words in >80% docs
            stop_words='english'
        )
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
            stop_words='english'
        )
    
    def fit_transform_bow(self, texts):
        sp_matrix=self.bow_vectorizer.fit_transform(texts)
        return (sp_matrix,self.bow_vectorizer.get_feature_names_out())
    
    def fit_transform_tfidf(self, texts):
        sp_matrix=self.tfidf_vectorizer.fit_transform(texts)
        return (sp_matrix,self.tfidf_vectorizer.get_feature_names_out())
    
    def get_top_features(self, vectorizer, X, n=20):
        tfidf=X.mean(axis=0).A1
        names=vectorizer.get_feature_names_out()
        order=np.argsort(tfidf)[-n::]
        return [(names[i],tfidf[i]) for i in order]
    
    def visualize_top_features(self, vectorizer, X, n=20):
        tops=self.get_top_features(vectorizer,X,n)
        features=[k[0] for k in tops]
        scores=[k[1] for k in tops]
        plt.barh(features,scores)

'''obj=FeatureExtractor()
f=np.array(pd.read_csv('all-data.csv',names=['sentiment','feedback'],encoding_errors='ignore')['feedback'])

#text=['I eat apple','You love me','I love you','She eat me','Love everyone']
X1=obj.fit_transform_bow(f)
print(X1[0].shape)
X=obj.fit_transform_tfidf(f)[0]
print(X.shape)
print(len(obj.tfidf_vectorizer.vocabulary_))
tops=obj.get_top_features(obj.tfidf_vectorizer,X)[::-1]
for t in tops:
    print(str(t[0])+'\t\t\tavg_tfidf_score = '+str(t[1]))
obj.visualize_top_features(obj.tfidf_vectorizer,X,50)
plt.show()'''
