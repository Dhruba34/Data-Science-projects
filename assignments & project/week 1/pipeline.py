import re
import string
import nltk
import contractions as cont
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as easy
from nltk.stem import WordNetLemmatizer as easy2
from nltk.tokenize import word_tokenize

class TextPreprocessor:
        def expand_contractions(self,text):
            return cont.fix(text)
        def remove_special_chars(self,text):
            return re.sub(r"[^a-zA-Z0-9\.\s]|\.(?!\d)|(?<!\d)\.",'',text)
        def tokenize(self,text):
            return word_tokenize(text)
        def remove_stopwords(self,tokens):
            words=set(stopwords.words("english"))
            return [word for word in tokens if word not in words]
        def stem(self,tokens):
            o=easy()
            return [o.stem(word) for word in tokens]
        def lemmatize(self,tokens):
            o=easy2()
            return [o.lemmatize(word) for word in tokens]
        def preprocess(self,text, use_lemmatization=False):
            o=TextPreprocessor()
            text=text.lower()
            text=o.expand_contractions(text)
            text=o.remove_special_chars(text)
            tokens=o.tokenize(text)
            tokens=o.remove_stopwords(tokens)
            if use_lemmatization:
                tokens=o.lemmatize(tokens)
            else:
                tokens=o.stem(tokens)
            return tokens
