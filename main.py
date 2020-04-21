import tweet_dumper as td
import tweetClassifierSystem as tc
import jsonFunctions as js


##this will be our main function

def main(account):


    try:
        nameOfTweetFile = td.get_all_tweets(account)


        return(tc.classify(nameOfTweetFile))

    except:
        return("Sorry account doesn't exist.")



# if __name__ == '__main__':
#      print(main("OHonours"))
