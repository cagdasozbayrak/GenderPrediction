from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


# Wrapper class for sklearn.linear_model.LogisticRegression
class LogisticRegression:
    def __init__(self, X, y):
        self.model = LR()
        self.X = X  # Feature set
        self.y = y  # Targets

    def train(self):
        self.model.fit(self.X, self.y)

    def k_fold_cross_validate(self, k=10, shuffle=True):
        cv = KFold(n_splits=k, shuffle=shuffle)
        cv_results = cross_validate(self.model, self.X, self.y, cv=cv, return_train_score=True)
        train_score = cv_results["train_score"]
        test_score = cv_results["test_score"]
        return train_score, test_score
