# Sentiment Analysis
The task is to identify sentiment analysis of amazon reviews and twitter tweets.

Amazon reviews: Implementation of Naive Bayes classifier after pre-processing the reviews to clean and label them.
Twitter Tweets: Implementation of Logistic Regression, Naive Bayes and Decision Trees classifiers after pre-processing the tweets. Also, used GloVe for word embeddings and ELMO to fetch contextual embeddings.

## Dataset
I used open source data set of amazon reviews for this purpose.

Link: https://www.cs.jhu.edu/mdredze/datasets/sentiment/unprocessed.tar.gz

For tweets, scraped a small dataset of tweets off the website.

## Requirements
1. Libraries required for running the python script:
- sklearn
- pandas
- numpy
- matplotlib

## Run the script
To execute Amazon reviews sentiment analysis -  `python naive_bayes_reviews.py`
To execute Tweets sentiment analysis-  `python log_res.py`, `python log_res_word_embed.py` and `python better_models.py`

## General notes
1. Correct the data folder location to where the reviews are present.

2. To test a particular review, use this line after executing the complete script: `classifier.classify(get_features('returned tshirt .', all_words, 500))`

## Contact
Please contact parul100495@gmail.com in case of any queries.
