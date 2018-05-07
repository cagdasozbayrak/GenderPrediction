import os
from pprint import pprint
import re
import nltk
from pathlib import Path
import string
import pickle
from sklearn import preprocessing
nltk.data.path.append(Path("./nltk_data"))


# Parses and stores the dataset and provides utility functions to use/manage the data
class TweetPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.tweets = {}
        self.author_genders = {}
        self.vectorized_features = []
        self.targets = []
        self.build_tweet_dictionary(dataset_path)
        self.extract_author_genders(dataset_path, "truth.txt")
        self.vectorize_features()
        self.extract_targets()

    # Some utility functions
    def get_gender(self, author_id):
        return self.author_genders.get(author_id)

    def get_tweets_of(self, author_id):
        return self.tweets.get(author_id)

    def print_all_tweets(self):
        pprint(self.tweets)

    # Reads and parses truth.txt to be used as a target to train a model
    # Loads the author dictionary from .pickle file if it exists
    # If the file does not exist, it will read, parse and save the dictionary to a .pickle file
    def extract_author_genders(self, dataset_path, filename):
        author_genders = load_author_genders()
        if not author_genders:
            author_genders = {}
            file = open(os.path.join(dataset_path, filename), "r")
            text = file.readlines()

            for line in text:
                line_parts = line.strip().split(":::")
                author = line_parts[0]
                gender = line_parts[1]
                author_genders[author] = gender

            file.close()
            save_author_genders(author_genders)
        self.author_genders = author_genders

    # Reads the dataset from the given path, parses ands tags the tweets
    # Loads the tagged tweets from .pickle file if it exists
    # If the file does not exist, it will parse, tag and save the tagged tweets to a .pickle file
    # The keys of the tweets dictionary are author ids, the values are the list of pairs of (word, pos_tag)
    def build_tweet_dictionary(self, dataset_path):
        tweets = load_tweets_tagged()
        if not tweets:
            tweets = {}
            for document in dataset_path.iterdir():
                if str(document).endswith(".xml"):
                    xml_str = open(document, encoding='utf-8').read()
                    tweet_list = parse_and_tag_tweets(xml_str)
                    tweets[str(document)[3:-4]] = tweet_list
            save_tweets_tagged(tweets)
        self.tweets = tweets

    # Vectorizes and scales the features so that the text data can be used to train a chosen model
    # Loads the vectorized features from .pickle file if it exists
    # If the file does not exist, it will vectorize the features and save the result to a .pickle file
    def vectorize_features(self):
        vectorized_features = load_vectorized_features()
        if type(vectorized_features) != None:  # If vectorized features can not be loaded from the pickle file
            vectorized_features = []
            for author, gender in self.author_genders.items():
                tagged_words = self.get_tweets_of(author)
                n_determiner = 0.0001
                n_preposition = 0.0001
                n_pronoun = 0.0001
                for _, tag in tagged_words:
                    if tag in ["DT", "PDT", "WDT"]:
                        n_determiner += 1
                    elif tag in ["IN", "TO"]:
                        n_preposition += 1
                    elif tag in ["PRP", "PRP$", "WP", "WP$"]:
                        n_pronoun += 1

                n_words = len(tagged_words)
                determiner_percentage = n_determiner / n_words
                preposition_percentage = n_preposition / n_words
                pronoun_percentage = n_pronoun / n_words
                vectorized_features.append([determiner_percentage, preposition_percentage, pronoun_percentage])
            vectorized_features = preprocessing.scale(vectorized_features)
            save_vectorized_features(vectorized_features)
        self.vectorized_features = vectorized_features

    # Builds a list of targets from the author gender dictionary generated
    def extract_targets(self):
        for _, gender in self.author_genders.items():
            self.targets.append(gender)


# Functions to save/load preprocessed data as .pickle file
def load_author_genders():
    path = Path("./author_genders.pickle")
    if path.is_file():
        file = path.open("rb")
        author_genders = pickle.load(file)
        file.close()
        return author_genders
    return None


def load_tweets_tagged():
    path = Path("./tweets_tagged.pickle")
    if path.is_file():
        file = path.open("rb")
        tweets_tagged = pickle.load(file)
        file.close()
        return tweets_tagged
    return None


def load_vectorized_features():
    path = Path("./vectorized_features.pickle")
    if path.is_file():
        file = path.open("rb")
        vectorized_features = pickle.load(file)
        file.close()
        return vectorized_features
    return None


def save_author_genders(author_genders):
    file = open(Path("./author_genders.pickle"), "wb")
    pickle.dump(author_genders, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_tweets_tagged(tweets):
    with open(Path("./tweets_tagged.pickle"), "wb") as handle:
        pickle.dump(tweets, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_vectorized_features(vectorized_features):
    with open(Path("./vectorized_features.pickle"), "wb") as handle:
        pickle.dump(vectorized_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Cleans the dataset as preparation for tagging operation
def parse_and_tag_tweets(xml_str):
    url_reg = r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*'
    twitter_handle_reg = r'([@?])(\w+)\b'
    re_emoji = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)  # taken from alex-just in github
    re_non_alphanumeric = re.compile('[^a-zA-Z0-9 :]')
    translator = str.maketrans('', '', string.punctuation)
    xml_str = xml_str.replace("&lt;", "<")
    xml_str = xml_str.replace("&gt;", ">")
    xml_str = xml_str.replace("&amp;", "&")
    xml_str = re_emoji.sub('', xml_str)  # removes emojis from the text
    xml_str = re.sub(url_reg, '', xml_str)  # removes the urls in the text
    xml_str = str(re.sub(twitter_handle_reg, '', xml_str))  # removes usernames like @username in text
    xml = xml_str.strip().split("<author lang=\"en\">\n\t<documents>")[1].split("</document>")
    tweet_list = []
    for element in xml:
        tweets = element.split("\n\t\t<document><![CDATA[")
        for tweet in tweets[1:]:
            tweet = re_non_alphanumeric.sub('', tweet)  # removes non-alphanumerical characters
            tweet = tweet.strip()  # removes the trailing new-line character
            tweet = tweet.translate(translator)  # removes punctuation
            tweet_with_tags = tag_words(tweet)  # returns a list of (word, pos_tag) pairs
            tweet_list.extend(tweet_with_tags)
    return tweet_list[:-1]


# tags all the words for given tweet returns a list of (word, pos_tag) pairs
def tag_words(tweet):
    words_to_tag = nltk.word_tokenize(tweet)
    fixed_words = []
    for word in words_to_tag:
        fixed_words.append(fix_repeated_letters(word))
    words_with_tags = nltk.pos_tag(fixed_words)
    return words_with_tags


# Fixes repeated letters. For example: for given string "amazzzzzzzziing" returns "amazing"
def fix_repeated_letters(word):
    word = word.lower()
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", word)