# Import the libraries that we will use to help create the train and test sets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Test of a naive bayes algorithm, the "fit" is the training
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


# Import the dataset, need to use the ISO-8859-1 encoding due to some invalid UTF-8 characters
df = pd.read_csv("../Data_Twitter.csv")

# Randomly select 14000 normal examples from the dataframe
dfNormal = df[df["label"] == "normal"].sample(n=24000 )
# Randomly select 14000 offensive examples from the dataframe
dfOffensive = df[df["label"] == "offensive"].sample(n=14000)
# Combine the results to make a small random subset of reviews to use
dfPartial = dfNormal.append(dfOffensive)


# Split the data such that 90% is used for training and 10% is used for testing
# We use the parameter stratify to split the training and testing data equally to create
# a balanced dataset
train_tweets, test_tweets, train_tags, test_tags = train_test_split(dfPartial["text"],
                                                                      dfPartial["label"],
                                                                      test_size=0.1,
                                                                      stratify=dfPartial["label"])
train_tags = train_tags.to_numpy()
train_tweets = train_tweets.to_numpy()

# Testing set (what we will use to test the trained model)
test_tags = test_tags.to_numpy()
test_tweets = test_tweets.to_numpy()





# The CountVectorizer builds a dictionary of all words (count_vect.vocabulary_),
# and generates a matrix (train_counts), to represent each sentence
# as a set of indices into the dictionary. The words in the dictionary are the words found in train_tweets.

count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(train_tweets)







#############################################################################################################################################################################################
# Training the model with an SVM using a linear kernel (less computationally intensive)
clf_svm = svm.SVC(kernel="linear", max_iter=5000).fit(train_counts, train_tags)


# # Testing on training set
# predicted_svm = clf_svm.predict(train_counts)
# # Print the first ten predictions
# for doc, category in zip(train_tweets[:15], predicted_svm[:15]):   # zip allows to go through two lists simultaneously
#     print('%r => %s\n' % (doc, category))
# correct = 0
# for tag, pred in zip(train_tags, predicted_svm):   # zip allows to go through two lists simultaneously
#     if (tag == pred):
#         correct += 1
# print("Correctly classified %s total training examples out of %s examples" %(correct, train_tags.size))
#
#
#
#
#

# Pre-process test set test_tweets

test_tweets_counts = count_vect.transform(test_tweets)
# Predict the results
test_predicted_svm = clf_svm.predict(test_tweets_counts)
# Print the first ten predictions
for doc, category in zip(test_tweets[:10], test_predicted_svm[:10]):   # zip allows to go through two lists simultaneously
    print('%r => %s\n' % (doc, category))
# Print the total correctly classified instances out of the total instances
correct = 0
for tag, pred in zip(test_tags, test_predicted_svm):   # zip allows to go through two lists simultaneously
    if (tag == pred):
        correct += 1
print("Correctly classified %s total training examples out of %s examples" %(correct, test_tags.size))