import re

import nltk
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts sentence and/or token statistics from corpus of texts.
    """

    def __init__(self, sent_stats: bool = True, tkn_stats=True):
        """
        :param sent_stats: whether to calculate sentence stats
        :param tkn_stats: whether to calculate token stats
        """
        self.punct = ".,—?«:!;"
        self.tokenizer = nltk.RegexpTokenizer(r"[А-яЁё]+")
        self.sent_stats = sent_stats
        self.tkn_stats = tkn_stats
        try:
            assert (
                self.sent_stats or self.tkn_stats
            ), "At least one variable should be True"
        except Exception as e:
            print(f"AssertionError: {e}")

    def _extract_sent_stats(self, text: str) -> dict:
        sent_stats = dict.fromkeys([s for s in self.punct], 0)
        punctuation_pattern = re.compile(f"[{self.punct}]")
        punctuation_list = re.findall(punctuation_pattern, text)
        for s in punctuation_list:
            sent_stats[s] += 1
        sent_stats["avg_snt_len"] = len(text) / (
            sent_stats["."] + sent_stats["!"] + sent_stats["?"]
        )
        return sent_stats

    def _extract_tkn_stats(self, text: str) -> dict:
        word_stats = dict()
        tokens = self.tokenizer.tokenize(text.lower())
        tokens_total_length = sum([len(token) for token in tokens])
        word_stats["avg_tkn_len"] = tokens_total_length / len(tokens)
        word_stats["ttr"] = len(set(tokens)) / len(tokens)
        return word_stats

    def _extract_text_stats(self, text: str) -> dict:
        text_stats = dict()
        if self.sent_stats:
            text_stats.update(**self._extract_sent_stats(text))
        if self.tkn_stats:
            text_stats.update(**self._extract_tkn_stats(text))
        return text_stats

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = pd.json_normalize(X.apply(self._extract_text_stats))
        return X_transformed
