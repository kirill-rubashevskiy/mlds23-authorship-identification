import re
from string import punctuation

import compress_fasttext
import nltk
import numpy as np
import pandas as pd
import simplemma
from nltk.tokenize import RegexpTokenizer, word_tokenize
from pymystem3 import Mystem
from sklearn.base import BaseEstimator, TransformerMixin


nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger_ru")


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

    def _preprocess(self, text):
        tokens = self.tokenizer.tokenize(text.lower())
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        if self.lemmatize:
            tokens = [simplemma.lemmatize(token, lang="ru") for token in tokens]
        return " ".join(tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.apply(self._preprocess)
        else:
            X = pd.Series(X, name="text").apply(self._preprocess)
        return X


class FastTextTransformer(BaseEstimator, TransformerMixin):
    """Convert texts into their mean fastText vectors"""

    def __init__(self):
        self.model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
            "https://github.com/avidale/compress-fasttext/releases/download/gensim-4-draft/geowac_tokens_sg_300_5_2020-100K-20K-100.bin"
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.stack([np.mean([self.model[w] for w in text.split()], 0) for text in X])


my_stopwords = []  # заглушка если придумаем стопслова
russian_stopwords_new = nltk.corpus.stopwords.words("russian")
russian_stopwords_new.extend(my_stopwords)
not_stopwords = {"не", "ни"}
russian_stopwords = [word for word in russian_stopwords_new if word not in not_stopwords]


def text_splitter(text, number=1000):
    # Исходная строка - отделяем 12 символов - это id автора, остальное - сам текст произведения
    author = text[:12]
    text = text[12:]

    # Токенизация строки
    tokens = nltk.word_tokenize(text)

    # Максимальная длина подстроки
    max_length = number

    # Список подстрок
    substrings = []

    # Текущая подстрока
    current_substring = ""

    # Обработка токенов
    for token in tokens:
        # Добавление токена в текущую подстроку
        current_substring += token + " "

        # Если длина текущей подстроки достигла максимальной длины
        if len(current_substring) >= max_length:
            # Находим последний знак препинания в текущей подстроке
            last_punctuation = max(
                current_substring.rfind("."),
                current_substring.rfind(","),
                current_substring.rfind("!"),
                current_substring.rfind("?"),
            )

            # Отрезаем все символы после последнего знака препинания
            if last_punctuation != -1:
                current_substring = current_substring[: last_punctuation + 1]

            # Добавляем текущую подстроку в список подстрок
            substrings.append(current_substring.strip())

            # Начинаем собирать новую подстроку
            current_substring = ""

    # Добавление последней подстроки в список подстрок
    if current_substring != "":
        substrings.append(current_substring.strip())

    # сохраняем список в датафрейм - каждая подстрока как отдельное наблюдение, id автора - таргет
    df = pd.DataFrame(columns=["target", "text"])
    df["text"] = pd.Series(substrings)
    df["target"] = author

    return df


def convert_to_lower(text):
    return text.lower()


def remove_numbers(text):
    text = re.sub(r"\d+", "", text)
    return text


def remove_http(text):
    text = re.sub(r"\s*https?:\/\/t.co\/[A-Za-zа-яА-Я0-9]*\s*", " ", text).strip()
    return text


def remove_short_words(text):
    # Удаляем слова из одной или двух букв и заменяем их на один пробел
    text = re.sub(r"\b\w{1,2}\b", " ", text)
    # Удаляем лишние пробелы, возникающие после удаления коротких слов
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def remove_punctuation(text):
    punctuations = """!()[]{};«№»:'",`<>./?@=#$%^&*~"""
    # Добавляем пробел вокруг символов пунктуации, чтобы избежать слипания слов после их удаления
    for punct in punctuations:
        text = text.replace(punct, " ")
    # Удаляем лишние пробелы, возникающие после замены
    text = re.sub(r"\s+", " ", text).strip()
    return text


mystem = Mystem()
tokenizer = RegexpTokenizer(r"[A-Za-zА-яЁё]+")


# удаляем цифры, http, приводим к нижнему регистру, убираем пробелы, короткие слова и пунктуацию
def preprocess_text1(text, tokenize=True, tostr=True):
    text = remove_numbers(text)
    text = remove_http(text)
    text = convert_to_lower(text)
    text = remove_short_words(text)  # это убирает не/ни кстати
    text = remove_punctuation(
        text
    )  # cлепливает слова, там где знаки препинания без пробела

    text = str(text)
    if tokenize:
        tokens = mystem.lemmatize(text.lower())
        tokens = [
            token
            for token in tokens
            if token not in russian_stopwords
            and token != " "
            and len(token) >= 3
            and token.strip() not in punctuation
            and token.isdigit() is False
        ]
    if tostr:
        tokens = " ".join(tokens)  # чтобы сделать не список, а строку
    return tokens


# удаляем цифры, http, приводим к нижнему регистру, убираем пробелы, короткие слова. ОСТАВЛЯЕМ пунктуацию
def preprocess_text2(text, tokenize=True, tostr=True):
    text = remove_numbers(text)
    text = remove_http(text)
    text = convert_to_lower(text)
    text = remove_short_words(text)  # это убирает не/ни кстати
    # text = remove_punctuation(text) #cлепляет слова, там где знаки препинания без пробела

    text = str(text)
    if tokenize:
        tokens = mystem.lemmatize(text.lower())
        tokens = [
            token
            for token in tokens
            if token not in russian_stopwords
            and token != " "
            and len(token) >= 3  # and token.strip() not in punctuation \
            and token.isdigit() is False
        ]
    if tostr:
        tokens = " ".join(tokens)  # чтобы сделать не список, а строку
    return tokens


# удаляем цифры, http, приводим к нижнему регистру, убираем пробелы, пунктуацию ОСТАВЛЯЕМ короткие слова.
def preprocess_text3(text, tokenize=True, tostr=True):
    text = remove_numbers(text)
    text = remove_http(text)
    text = convert_to_lower(text)
    # text = remove_short_words(text) #это убирает не/ни кстати
    text = remove_punctuation(
        text
    )  # cлепляет слова, там где знаки препинания без пробела

    text = str(text)
    if tokenize:
        tokens = mystem.lemmatize(text.lower())
        tokens = [
            token
            for token in tokens
            if token not in russian_stopwords
            and token != " "  # and len(token)>=3 \
            and token.strip() not in punctuation
            and token.isdigit() is False
        ]
    if tostr:
        tokens = " ".join(tokens)  # чтобы сделать не список, а строку
    return tokens


# удаляем цифры, http, приводим к нижнему регистру, убираем пробелы, пунктуацию ОСТАВЛЯЕМ короткие слова и стопслова
def preprocess_text4(text, tokenize=True, tostr=True):
    text = remove_numbers(text)
    text = remove_http(text)
    text = convert_to_lower(text)
    # text = remove_short_words(text) #это убирает не/ни кстати
    text = remove_punctuation(
        text
    )  # cлепляет слова, там где знаки препинания без пробела

    text = str(text)
    if tokenize:
        tokens = mystem.lemmatize(text.lower())
        tokens = [
            token
            for token in tokens
            if  # token not in russian_stopwords\
            # and
            token != " "  # and len(token)>=3 \
            and token.strip() not in punctuation
            and token.isdigit() is False
        ]
    if tostr:
        tokens = " ".join(tokens)  # чтобы сделать не список, а строку
    return tokens


def tokenizing(text):
    tokens = word_tokenize(text)
    # Remove Stopwords from tokens
    result = [i for i in tokens if i not in russian_stopwords]
    return result


# расшифровка меток
label2name = {
    0: "А. Пушкин",
    1: "Д. Мамин-Сибиряк",
    2: "И. Тургенев",
    3: "А. Чехов",
    4: "Н. Гоголь",
    5: "И. Бунин",
    6: "А. Куприн",
    7: "А. Платонов",
    8: "В. Гаршин",
    9: "Ф. Достоевский",
}


def clean_sentence(sentence):
    """
    заменяем все некириллические символы и не знаки препинания на пробелы

    >>> clean_sentence("как-то - 'рано' 'Marta'? пела: лЕСОМ, * &нифига(она) не ела")
    "как-то - 'рано' ''? пела: лЕСОМ,  нифига(она) не ела"
    """
    sentence = re.sub(r"[^а-яА-ЯёЁ \-\"!'(),.:;?]", "", sentence)
    return sentence


def make_punkt(sentence):
    """
    заменяем знаки препинания на их кодовые обозначения

    >>> make_punkt(', двадцать-два, - номер, помереть: не сейчас!')
    ' CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL '
    """
    repl = [
        ("...", " MP "),
        ("..", " MP "),
        (".", " PNT "),
        (",", " CM "),
        ("?", " QST "),
        ("!", " EXCL "),
        (":", " CL "),
        (";", " SMC "),
    ]
    for p, r in repl:
        sentence = sentence.replace(p, r)
    sentence = re.sub(
        r"\s?-\s|\s-\s?", " DSH ", sentence
    )  # не трогать тире в слове (как-то)

    return sentence


def make_grams(sentence):
    """
    заменяет слова в тексте на соответствующие им лексические кодировщики (часть речи, падеж и тп)

    >>> make_grams(' CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL ')
    'CM NUM=(вин|им) NUM=(вин,муж,неод|им,муж|вин,сред|им,сред) CM DSH S,муж,неод=(вин,ед|им,ед) CM V,нп=инф,сов CL PART= ADV= EXCL'
    """

    mystem_analyzer = Mystem()
    morph = mystem_analyzer.analyze(sentence)

    ret = []
    for lex in morph:
        if lex["text"] in ["MP", "PNT", "CM", "QST", "EXCL", "CL", "SMC", "DSH"]:
            ret.append(lex["text"])
            continue

        try:
            if "analysis" in lex.keys() and "gr" in lex["analysis"][0].keys():
                ret.append(lex["analysis"][0]["gr"])
        except Exception as e:
            # встретил что-то непотребное в стиле ру-ру-ру
            print(e)
            pass
    return " ".join(ret)


def make_grams_brief(sentence):
    """
    заменяет слова в тексте на соответствующие им лексические кодировщики
    но уже в сокращенном варианте

    >>> make_grams_brief(' CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL ')
    'CM NUM NUM CM DSH S,муж,неод CM V,нп CL PART ADV EXCL'
    """

    mystem_analyzer = Mystem()
    morph = mystem_analyzer.analyze(sentence)

    ret = []
    for lex in morph:
        if lex["text"] in ["MP", "PNT", "CM", "QST", "EXCL", "CL", "SMC", "DSH"]:
            ret.append(lex["text"])
            continue

        try:
            if "analysis" in lex.keys() and "gr" in lex["analysis"][0].keys():
                ret.append(lex["analysis"][0]["gr"].split("=")[0])
        except Exception as e:
            # встретил что-то непотребное в стиле ру-ру-ру
            print(e)
            pass
    return " ".join(ret)


def prepare_text(Text_corp, full=True):
    """
    итоговая предобработка для наших моделей
    >>> prepare_text(["Мама. Мыла раму папе"], full=True)
    ['S,жен,од=им,ед PNT V,несов,пе=прош,ед,изъяв,жен S,жен,неод=вин,ед S,муж,од=(пр,ед|дат,ед)']

    >>> prepare_text(["Мама. Мыла раму папе"], full=False)
    ['S,жен,од PNT V,несов,пе S,жен,неод S,муж,од']
    """

    res = []
    for text in Text_corp:
        text = clean_sentence(text)
        text = make_punkt(text)
        if full:
            text = make_grams(text)
        else:
            text = make_grams_brief(text)
        res.append(text)
    return res
