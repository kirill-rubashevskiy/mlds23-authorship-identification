import nltk
import pandas as pd

from mlds23_authorship_identification.extractors import TextStatsExtractor


nltk.download("averaged_perceptron_tagger_ru")


def test_initialization():
    extractor = TextStatsExtractor(sent_stats=False, tkn_stats=False, pos_stats=True)
    assert not extractor.sent_stats
    assert not extractor.tkn_stats
    assert extractor.pos_stats


def test_tokenize():
    extractor = TextStatsExtractor()
    tokenizer = nltk.RegexpTokenizer(r"[А-яЁё]+")
    tokens = extractor._tokenize("Привет, мир! Это тест.", tokenizer)
    assert tokens == ["привет", "мир", "это", "тест"]


def test_extract_sent_stats():
    extractor = TextStatsExtractor()
    extractor.sent_stats_list = [
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
    text = "Привет... Как дела? Хорошо! Отлично."
    punct_tokens = ["...", "?", "!", "."]
    expected = {
        "...": 1,
        ".": 1,
        ",": 0,
        "—": 0,
        "?": 1,
        "«": 0,
        ":": 0,
        "!": 1,
        ";": 0,
        "avg_snt_len": 9,
    }
    assert extractor._extract_sent_stats(text, punct_tokens) == expected


def test_extract_tkn_stats():
    extractor = TextStatsExtractor()
    extractor.tkn_stats_list = ["avg_tkn_len", "ttr"]
    word_tokens = ["привет", "мир", "мир", "привет"]
    expected = {"avg_tkn_len": 4.5, "ttr": 0.5}
    assert extractor._extract_tkn_stats(word_tokens) == expected


def test_extract_pos_stats():
    extractor = TextStatsExtractor()
    extractor.pos_tags_list = ["V", "S", "A", "ADV", "PR", "CONJ", "S-PRO", "A-PRO"]
    extractor.pos_stats_list = [
        "noun2verb",
        "adj2noun",
        "adv2verb",
        "pr2noun",
        "conj2noun",
        "pron2noun",
        "verb2word",
    ]
    # пусть что pos_tagger возвращает следующее (упрощение):
    extractor.pos_tagger = lambda tokens: [
        (token, "S") if token == "мир" else (token, "V") for token in tokens
    ]
    word_tokens = ["мир", "привет"]
    expected = {
        "noun2verb": 1.0,
        "adj2noun": 0,
        "adv2verb": 0,
        "pr2noun": 0,
        "conj2noun": 0,
        "pron2noun": 0,
        "verb2word": 0.5,
    }
    assert extractor._extract_pos_stats(word_tokens) == expected


def test_fit_and_transform_integration():
    extractor = TextStatsExtractor()
    df = pd.DataFrame(
        {"text": ["Привет... Как дела? Хорошо! Отлично.", "Тест. Тест, тест."]}
    )
    # fit и transform
    extractor.fit(df["text"])
    transformed = extractor.transform(df["text"])
    # проверка, что в результате получаем датафрейм с нужными колонками
    expected_columns = set(
        [
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
            "avg_tkn_len",
            "ttr",
            "noun2verb",
            "adj2noun",
            "adv2verb",
            "pr2noun",
            "conj2noun",
            "pron2noun",
            "verb2word",
            "punct2word",
        ]
    )
    # проверка, что значения в какой-либо колонке соответствуют ожидаемым
    assert set(transformed.columns) == expected_columns
