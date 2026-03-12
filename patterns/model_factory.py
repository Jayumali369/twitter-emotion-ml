from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

class ModelFactory:

    @staticmethod
    def get_model(model_type):

        if model_type == "naive_bayes":
            return MultinomialNB()

        elif model_type == "svm":
            return SVC()

        else:
            raise ValueError("Unknown model")