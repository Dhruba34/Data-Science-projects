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
