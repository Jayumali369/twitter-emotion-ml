import snscrape.modules.twitter as sntwitter
import pandas as pd

def collect_tweets(topic, limit=200):

    tweets = []

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(topic).get_items()):

        if i > limit:
            break

        tweets.append(tweet.content)

    df = pd.DataFrame(tweets, columns=["tweet"])

    return df