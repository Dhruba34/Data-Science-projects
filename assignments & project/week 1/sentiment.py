from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from pipeline import TextPreprocessor as tp
from collections import Counter
import numpy as np
class LexiconSentimentAnalyzer:
    def vader_sentiment(self,text):
        vader=SentimentIntensityAnalyzer()
        score=vader.polarity_scores(text)['compound']
        label=''
        if score>=0.05:
            label='Positive'
        elif score<=-0.05:
            label='Negative'
        else:
            label='Neutral'
        return {'compound':score,'label':label}
    def textblob_sentiment(self,text):
        blob=TextBlob(text)
        pol=blob.sentiment.polarity
        #print(blob.sentiment.polarity,blob.sentiment.subjectivity)
        label=''
        if pol>=0.05:
            label='Positive'
        elif pol<=-0.05:
            label='Negative'
        else:
            label='Neutral'
        return {'polarity':(pol+1)/2,'label':label}
    def custom_lexicon_sentiment(self,text):
        lexicon = {
    "excel": 3,
    "earn": 3,
    "bullish": 3,
    "ralli": 3,
    "breakout": 3,
    "surg": 3,
    "strong": 2,
    "growth": 2,
    "profit": 2,
    "outperform": 2,
    "upsid": 2,
    "beat": 2,
    "recoveri": 2,
    "momentum": 2,
    "expans": 2,
    "upgrad": 2,
    "dividend": 1,
    "bearish": -3,
    "crash": -3,
    "collaps": -3,
    "plung": -3,
    "weak": -2,
    "loss": -2,
    "declin": -2,
    "downsid": -2,
    "miss": -2,
    "downgrad": -2,
    "volatil": -1,
    "risk": -1,
    "slowdown": -2,
    "recess": -3,
    "inflat": -1
}
        obj=tp()
        tokens=obj.preprocess(text)
        #print(tokens)
        counts=Counter(tokens)
        score=0
        maxi=0
        for word,count in counts.items():
            score+=count*lexicon.get(word,0)
            maxi+=count
        if maxi==0:
            return 0
        score=score/(maxi*3)
        return score
    def analyze(self,text,w1=1,w2=1,w3=1):
        s1,s2,s3=self.vader_sentiment(text)['compound'],self.textblob_sentiment(text)['polarity']*2-1,self.custom_lexicon_sentiment(text)
        mag=(s1*w1+s2*w2+s3*w3)/(w1+w2+w3)
        agree=1/(1+np.var([s1,s2,s3]))
        arr=[t>=0 for t in [s1,s2,s3]]
        direc=1.0 if all(arr) or not any(arr) else 0.66
        return {'vader score':s1,'textblob score':(s2+1)/2,'custom lexicon score':s3,'ensemble score':mag,'confidence':mag*agree*direc}
