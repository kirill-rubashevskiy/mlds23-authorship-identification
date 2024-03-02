from functools import partial

import nltk
import pandas as pd
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin


class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts sentence and/or token and/or pos statistics from corpus of texts.
    """

    def __init__(
        self, sent_stats: bool = True, tkn_stats: bool = True, pos_stats: bool = True
    ):
        """
        :param sent_stats: whether to calculate sentence statistics
        :param tkn_stats: whether to calculate token statistics
        :param pos_stats: whether to calculate pos statistics
        """
        self.sent_stats = sent_stats
        self.tkn_stats = tkn_stats
        self.pos_stats = pos_stats
        try:
            assert (
                self.sent_stats or self.tkn_stats or self.pos_stats
            ), "At least one variable should be True"
        except Exception as e:
            print(f"AssertionError: {e}")
        if sent_stats:
            self.sent_stats_list = [
                "...",
                ".",
                ",",
                "—",
                "?",
                "«",
                ":",
                "!",
                ";",
                "avg_snt_len",
            ]
            self.punct_tokenizer = nltk.RegexpTokenizer(r"(?:\.{3})|[.,—?«:!;]")
        if self.tkn_stats or self.pos_stats:
            self.word_tokenizer = nltk.RegexpTokenizer(r"[А-яЁё]+")
        if self.tkn_stats:
            self.tkn_stats_list = ["avg_tkn_len", "ttr"]
        if self.pos_stats:
            self.pos_tagger = partial(pos_tag, tagset="universal", lang="rus")
            self.pos_tags_list = ["VERB", "NOUN", "ADJ", "ADV", "ADP", "CONJ", "PRON"]
            self.pos_stats_list = [
                "noun2verb",
                "adj2noun",
                "adv2verb",
                "adp2noun",
                "conj2noun",
                "pron2noun",
                "verb2word",
            ]

    def _tokenize(self, text: str, tokenizer: nltk.RegexpTokenizer) -> list[str]:
        tokens = tokenizer.tokenize(text.lower())
        return tokens

    def _extract_sent_stats(self, text: str, punct_tokens: list[str]) -> dict:
        sent_stats = dict.fromkeys(self.sent_stats_list, 0)
        for token in punct_tokens:
            sent_stats[token] += 1
        if sent_stats["."] + sent_stats["!"] + sent_stats["?"] > 0:
            sent_stats["avg_snt_len"] = len(text) / (
                sent_stats["."] + sent_stats["!"] + sent_stats["?"]
            )
        else:
            sent_stats["avg_snt_len"] = len(text)
        return sent_stats

    def _extract_tkn_stats(self, word_tokens: list[str]) -> dict:
        tkn_stats = dict.fromkeys(self.tkn_stats_list, 0)
        tokens_total_length = sum([len(token) for token in word_tokens])
        if word_tokens:
            tkn_stats["avg_tkn_len"] = tokens_total_length / len(word_tokens)
            tkn_stats["ttr"] = len(set(word_tokens)) / len(word_tokens)
        return tkn_stats

    def _extract_pos_stats(self, word_tokens: list[str]) -> dict:
        pos_stats = dict().fromkeys(self.pos_stats_list, 0)
        pos_count = dict.fromkeys(self.pos_tags_list, 0)
        _, tags = zip(*self.pos_tagger(word_tokens), strict=True)
        for tag in tags:
            if tag in pos_count:
                pos_count[tag] += 1
        if pos_count["NOUN"] and pos_count["VERB"]:
            pos_stats["noun2verb"] = pos_count["NOUN"] / pos_count["VERB"]
        if pos_count["ADJ"] and pos_count["NOUN"]:
            pos_stats["adj2noun"] = pos_count["ADJ"] / pos_count["NOUN"]
        if pos_count["ADV"] and pos_count["VERB"]:
            pos_stats["adv2verb"] = pos_count["ADV"] / pos_count["VERB"]
        if pos_count["ADP"] and pos_count["NOUN"]:
            pos_stats["adp2noun"] = pos_count["ADP"] / pos_count["NOUN"]
        if pos_count["CONJ"] and pos_count["NOUN"]:
            pos_stats["conj2noun"] = pos_count["CONJ"] / pos_count["NOUN"]
        if pos_count["PRON"] and pos_count["NOUN"]:
            pos_stats["pron2noun"] = pos_count["PRON"] / pos_count["NOUN"]
        if pos_count["VERB"]:
            pos_stats["verb2word"] = pos_count["VERB"] / len(word_tokens)
        return pos_stats

    def _extract_text_stats(self, text: str) -> dict:
        punct_tokens = None
        word_tokens = None
        text_stats = dict()
        if self.sent_stats:
            punct_tokens = self._tokenize(text, self.punct_tokenizer)
            text_stats.update(**self._extract_sent_stats(text, punct_tokens))
        if self.tkn_stats or self.pos_stats:
            word_tokens = self._tokenize(text, self.word_tokenizer)
        if self.tkn_stats:
            text_stats.update(**self._extract_tkn_stats(word_tokens))
        if self.pos_stats:
            text_stats.update(**self._extract_pos_stats(word_tokens))
        if punct_tokens and word_tokens:
            text_stats["punct2word"] = len(punct_tokens) / len(word_tokens)
        return text_stats

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = pd.json_normalize(X.apply(self._extract_text_stats))
        return X_transformed
