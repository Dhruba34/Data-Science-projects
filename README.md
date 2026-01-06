# Data-Science-projects

This repository lists down some assignments completed during `WiDs 5.0 2025 projects` organised by the **Analytics club of IIT Bombay**. The main topic of the project is: **"Sentiment Analysis for Financial News with NLP"**.

- ## skills covered
  - Important libraries of Python like matplotlib, pandas, numpy, etc.
  - some good algorithms like K - means clustering
  - use of APIs of pretrained models for NLP
  - Data cleaning and preprocessing
  - Data visualization using Seaborn, Matplotlib

- ## week-wise assignment topics
  - Lisan_Al_Gaib problem statement (incorporating K-means clustering)
  - Within week 1 much hectic work was there:
    - a full data cleaning and preprocessing pipeline is made which tokenizes and stems data along with removing unnecessary things like stop words and punctuation
    - incorporating 3 different sentiment scoring systems using vader, textblob and custom lexicon specially for financial news sentiment analysis and giving a simplistic preview of scores from all different methods along with an ***ensemble score*** and a measure of ***confidence***
    - retrieving almost 4400 data of sentiment labels of different financial texts and training a linear model to choose different weights for the scores of 3 different approaches and deciding proper upper and lower thresholds for negative, neutral and positive sentiments.
    - no usage of any deep learning or ML algorithms. Just pure lexicon approach with a bit of optimization to reach an accuracy of 61% and F1 score of 0.52 even with a biased data taking care of recalls of minority classes which are the negative and positive sentiments
    - many representations of the final outcome is shown along with a visualization of the confusion matrix
  - **Week 2 :**
    - Training 3 models for sentiment analysis namely, Logistic Regression, KNN and Naive Bayes which are giving 75 - 80 % accuracy quite flawlessly and they beat basic lexicon based approaches.
    - Evaluation and comparision of all the 3 models and it is seen that Logistic Regression stands out with 77% accuracy
  - **Week 3 :**
    - Loading and exploring a pretrained transformer model called FinBERT designed for this task and learning about transformer model architecture and algorithm. Worth mentioning is Google's mind blowing research paper which read **"Attention is all you need"**.
    - Fine tuning and training FinBERT on my dataset and improving by 2% accuracy (from 89 to 91)
    - Comparing all 5 models made in weeks 1,2 and 3 namely, vader, logistic regression, knn, naive bayes and finbert
