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

# Randomly select 10000 fresh examples from the dataframe
dfNormal = df[df["label"] == "normal"].sample(n=24000)
# Randomly select 10000 rotten examples from the dataframe
dfOffensive = df[df["label"] == "offensive"].sample(n=14000)
# Combine the results to make a small random subset of reviews to use
dfPartial = dfNormal.append(dfOffensive)


# Split the data such that 90% is used for training and 10% is used for testing
# We use the parameter stratify to split the training and testing data equally to create
# a balanced dataset
train_tweets, test_tweets, train_tags, test_tags = train_test_split(dfPartial["text"],
                                                                      dfPartial["label"],
                                                                      stratify=dfPartial["label"])
train_tags = train_tags.to_numpy()
train_tweets = train_tweets.to_numpy()

# Testing set (what we will use to test the trained model)
test_tags = test_tags.to_numpy()
test_tweets = test_tweets.to_numpy()

df2 = pd.read_csv("OHonours_Testing_041120.csv")
control_test_tags = df2["label"]
control_test_tweets = df2["text"]


# Read the negative words
# to fix encoding problems, you might need to replace the line below
# with open("negative-words.txt", encoding = "ISO-8859-1") as f:

with open("bWord.txt") as f:
    negWords = f.readlines()
negWords = [p[0:len(p)-1] for p in negWords if p[0].isalpha()]

#print(negWords[:50])

# First let's define methods to count negative words

def countWord(text):
    count = 0
    for t in text.split():

        count += 1
    return count

def countNeg(text):
    count = 0
    for t in text.split():
        if t in negWords:
            count += 1
    return count

# Simple counting algorithm as baseline approach to polarity detection
def baselinePolarity(review):
    numW = countWord(review)
    numNeg = countNeg(review)

    perc = (numNeg / numW)*100
    perc = int(perc)
    if perc > 5:
        return "offensive"
    else:
        return "normal"
# Test the baseline method
# print("Testing baselinePolarity with the review:", train_tweets[0])
# print("baselinePriority result:", baselinePolarity(train_tweets[0]))
# print("Actual result:", train_tags[0])
# print(" ")
# print("Testing baselinePolarity with the review:", train_tweets[1])
# print("baselinePriority result:", baselinePolarity(train_tweets[1]))
# print("Actual result:", train_tags[1])



# Function takes a one dimensional array of reviews and a one dimensional array of
# tags as input and prints the number of incorrect assignments when running the baseline approach
# on the reviews.
# Let's establish the polarity for each review
# def incorrectReviews(reviews, tags):
#     nbWrong = 0
#     count = 0
#     for i in range(len(reviews)):
#         polarity = baselinePolarity(reviews[i])
#         if (count < 50):
#             print(reviews[i] + " -- Prediction: " + polarity + ". Actually: " + tags[i] + " \n")
#             count += 1
#         if (polarity != tags[i]):
#             nbWrong += 1
#
#     print('There are %s wrong predictions out of %s total predictions' %(nbWrong, len(tags)))
#
#
# incorrectReviews(test_tweets, test_tags)



# The CountVectorizer builds a dictionary of all words (count_vect.vocabulary_),
# and generates a matrix (train_counts), to represent each sentence
# as a set of indices into the dictionary. The words in the dictionary are the words found in train_tweets.

count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(train_tweets)

# Training the model
#clf = MultinomialNB().fit(train_counts, train_tags)


# Testing on training set
#predicted = clf.predict(train_counts)
# # Print the first ten predictions
# for doc, category in zip(train_tweets[:10], predicted[:10]):   # zip allows to go through two lists simultaneously
#     print('%r => %s\n' % (doc, category))
# correct = 0
# for tag, pred in zip(train_tags, predicted):   # zip allows to go through two lists simultaneously
#     if (tag == pred):
#         correct += 1
# print("Correctly classified %s total training examples out of %s examples" %(correct, train_tags.size))


# # Pre-process test set test_tweets
# # Note, we use transform and NOT fit_transform since this we do not want to re-fit the vecotrizer
# # that we used to train the model
# test_tweets_counts = count_vect.transform(test_tweets)
# # Predict the results
# predicted = clf.predict(test_tweets_counts)
# # Print the first ten predictions
# for doc, category in zip(test_tweets[:10], predicted[:10]):   # zip allows to go through two lists simultaneously
#     print('%r => %s\n' % (doc, category))
# correct = 0
# for tag, pred in zip(test_tags, predicted):   # zip allows to go through two lists simultaneously
#     if (tag == pred):
#         correct += 1
# # Print the total correctly classified instances out of the total instances
# print("Correctly classified %s total test examples out of %s examples" %(correct, test_tags.size))



#############################################################################################################################################################################################

###This is testing our control account


# test_tweets_counts = count_vect.transform(control_test_tweets)
# # Predict the results
# predicted = clf.predict(test_tweets_counts)
# # Print the first ten predictions
# for doc, category in zip(control_test_tweets[:10], predicted[:10]):   # zip allows to go through two lists simultaneously
#     print('%r => %s\n' % (doc, category))
# correct = 0
# for tag, pred in zip(control_test_tags, predicted):   # zip allows to go through two lists simultaneously
#     if (tag == pred):
#         correct += 1
# # Print the total correctly classified instances out of the total instances
# print("Correctly classified %s total test examples out of %s examples" %(correct, control_test_tags.size))




#############################################################################################################################################################################################
# Training the model with an SVM using a linear kernel (less computationally intensive)
# For the purpose of this Notebook we will stick to this simple model and stop after 3500
# iterations to save time (would take much longer otherwise, feel free to duplicate the notebook
# and test different parameters for yourself to see how much better it does!)
# This will take several minutes to run on less powerful machines, so be patient!
clf_svm = svm.SVC(kernel="linear", max_iter=5000).fit(train_counts, train_tags)


# # May take a few minutes to run on weaker machines
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






# # Pre-process test set test_tweets
# # Note, we use transform and NOT fit_transform since this we do not want to re-fit the vecotrizer
# # that we used to train the model
# test_tweets_counts = count_vect.transform(test_tweets)
# # Predict the results
# test_predicted_svm = clf_svm.predict(test_tweets_counts)
# # Print the first ten predictions
# for doc, category in zip(test_tweets[:10], test_predicted_svm[:10]):   # zip allows to go through two lists simultaneously
#     print('%r => %s\n' % (doc, category))
# # Print the total correctly classified instances out of the total instances
# correct = 0
# for tag, pred in zip(test_tags, test_predicted_svm):   # zip allows to go through two lists simultaneously
#     if (tag == pred):
#         correct += 1
# print("Correctly classified %s total training examples out of %s examples" %(correct, test_tags.size))
#
#


###This is testing our control account


test_tweets_counts = count_vect.transform(control_test_tweets)
# Predict the results
predicted = clf_svm.predict(test_tweets_counts)
# Print the first ten predictions
for doc, category in zip(control_test_tweets[:10], predicted[:10]):   # zip allows to go through two lists simultaneously
    print('%r => %s\n' % (doc, category))
correct = 0
for tag, pred in zip(control_test_tags, predicted):   # zip allows to go through two lists simultaneously
    if (tag == pred):
        correct += 1
# Print the total correctly classified instances out of the total instances
print("Correctly classified %s total test examples out of %s examples" %(correct, control_test_tags.size))