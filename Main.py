from TweetPreprocessor import TweetPreprocessor
from LogisticRegression import LogisticRegression
from pathlib import Path

if __name__ == '__main__':
    dataset_path = Path("en")
    tweet_preprocessor = TweetPreprocessor(dataset_path)

    model = LogisticRegression(tweet_preprocessor.vectorized_features, tweet_preprocessor.targets)
    train_scores, test_scores = model.k_fold_cross_validate()  # As default it performs 10-fold cross validation

    print("Train score: " + str(train_scores.mean()))
    print("Test score : " + str(test_scores.mean()))
