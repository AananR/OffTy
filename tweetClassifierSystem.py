# Import the libraries that we will use to help create the train and test sets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Test of a naive bayes algorithm, the "fit" is the training
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


def classify(account):
    # Import the dataset, need to use the ISO-8859-1 encoding due to some invalid UTF-8 characters
    df = pd.read_csv("../Data_Twitter.csv")

    # Randomly select 14000 normal examples from the dataframe
    dfNormal = df[df["label"] == "normal"].sample(n=24000 )
    # Randomly select 14000 offensive examples from the dataframe
    dfOffensive = df[df["label"] == "offensive"].sample(n=14000)
    # Combine the results to make a small random subset of reviews to use
    dfPartial = dfNormal.append(dfOffensive)


    # We use 100% of our data to train, since the testing will be done on the user queried account
    train_tweets, test_tweets, train_tags, test_tags = train_test_split(dfPartial["text"],
                                                                          dfPartial["label"])
    train_tags = train_tags.to_numpy()
    train_tweets = train_tweets.to_numpy()

    # The tweets to be classified

    df2 = pd.read_csv("../searchedIDS/" + account)

    df2Tweets = df2["text"]


    test_tweets = df2Tweets.to_numpy()

    df3 = df2[['id', 'text']]

    df3 =df3.to_numpy()






    # The CountVectorizer builds a dictionary of all words (count_vect.vocabulary_),
    # and generates a matrix (train_counts), to represent each sentence
    # as a set of indices into the dictionary. The words in the dictionary are the words found in train_tweets.

    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(train_tweets)







    #############################################################################################################################################################################################
    # Training the model with an SVM using a linear kernel (less computationally intensive)
    #clf_svm = svm.SVC(kernel="linear", max_iter=5000).fit(train_counts, train_tags)
    clf = MultinomialNB().fit(train_counts, train_tags)







    # Pre-process test set tweets

    tweets_counts = count_vect.transform(test_tweets)
    # Predict the results
    test_predicted_svm = clf.predict(tweets_counts)


    # This will hold all the tweets that are deemed offensive
    offensive_Tweets = {}
    for doc, category in zip(df3,
                             test_predicted_svm):  # zip allows to go through two lists simultaneously

        if (category == 'offensive'):
            id = doc[0]
            text = doc[1]
            offensive_Tweets[id] = text



    return offensive_Tweets

