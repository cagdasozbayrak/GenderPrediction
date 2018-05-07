# Gender Prediction with Logistic Regression
Assignment for CENG463 - Introduction to Machine Learning class.
The assignment was to use given tweet dataset to predict the gender of an author based on syntatic constructions of their tweets.

Tweets of each user is tagged using NLTK's pos_tagger and determiners, prepositions and pronouns are selected as features.
The frequency of each selected feature is fed into the model (logistic regression) to train.
10-fold cross validation is used to evaluate the accuracy of the model.
