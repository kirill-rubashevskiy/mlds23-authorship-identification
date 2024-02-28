import nltk
import pandas as pd
import simplemma
from sklearn.base import BaseEstimator, TransformerMixin


class TextTransformer(BaseEstimator, TransformerMixin):
    """
    Removes punctuation and stopwords and lemmatizes text.
    """

    def __init__(
        self,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ):
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.tokenizer = None
        if remove_punctuation:
            self.tokenizer = nltk.RegexpTokenizer(r"[А-яЁё]+")
        else:
            self.tokenizer = nltk.NLTKWordTokenizer()
        if remove_stopwords:
            self.stopwords = nltk.corpus.stopwords.words("russian")
        if lemmatize:
            self.lemmatizer = simplemma

    def _preprocess(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token, lang="ru") for token in tokens]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.apply(self._preprocess)
        else:
            X = pd.Series(X, name="text").apply(self._preprocess)
        return X
