import pandas as pd
import nltk
from nltk.corpus import stopwords
my_stopwords = [] #заглушка если придумаем стопслова
nltk.download('stopwords')
nltk.download('wordnet')
russian_stopwords_new = stopwords.words("russian")
russian_stopwords_new.extend(my_stopwords)
not_stopwords = {'не', 'ни'}
russian_stopwords = [word for word in russian_stopwords_new if word not in not_stopwords]
from pymystem3 import Mystem
from string import punctuation

def text_splitter(text):
    # Исходная строка - отделяем 12 символов - это id автора, остальное - сам текст произведения
    author = text[:12]
    text = text[12:]

    # Токенизация строки
    tokens = nltk.word_tokenize(text)

    # Максимальная длина подстроки
    max_length = 550

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
            last_punctuation = max(current_substring.rfind("."),
                                   current_substring.rfind(","),
                                   current_substring.rfind("!"),
                                   current_substring.rfind("?"))

            # Отрезаем все символы после последнего знака препинания
            if last_punctuation != -1:
                current_substring = current_substring[:last_punctuation + 1]

            # Добавляем текущую подстроку в список подстрок
            substrings.append(current_substring.strip())

            # Начинаем собирать новую подстроку
            current_substring = ""

    # Добавление последней подстроки в список подстрок
    if current_substring != "":
        substrings.append(current_substring.strip())

    # сохраняем список в датафрейм - каждая подстрока как отдельное наблюдение, id автора - таргет
    df = pd.DataFrame(columns=["target", "text"])
    df['text'] = pd.Series(substrings)
    df['target'] = author

    return df

def convert_to_lower(text):
    return text.lower()

def remove_numbers(text):
    text = re.sub(r'\d+' , '', text)
    return text

def remove_http(text):
    text = re.sub("https?:\/\/t.co\/[A-Za-zа-яА-Я0-9]*", ' ', text)
    return text

def remove_short_words(text):
    text = re.sub(r'\b\w{1,2}\b', '', text)
    return text

def remove_punctuation(text):
    punctuations = '''!()[]{};«№»:'"\,`<>./?@=#$-(%^)+&[*_]~'''
    # no_punct = ""
    no_punct = " "
    for char in text:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def remove_white_space(text):
    text = text.strip()
    return text

mystem = Mystem()

#удаляем цифры, http, приводим к нижнему регистру, убираем пробелы, короткие слова и пунктуацию
def preprocess_text1(text):
    text = remove_numbers(text)
    text = remove_http(text)
    text = convert_to_lower(text)
    text = remove_white_space(text)
    text = remove_short_words(text) #это убирает не/ни кстати
    text = remove_punctuation(text) #cлепливает слова, там где знаки препинания без пробела

    text = str(text)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and len(token)>=3 \
              and token.strip() not in punctuation \
              and token.isdigit()==False]
#      tokens = " ".join(tokens) #чтобы сделать не список, а строку
    return tokens

#удаляем цифры, http, приводим к нижнему регистру, убираем пробелы, короткие слова. ОСТАВЛЯЕМ пунктуацию
def preprocess_text2(text):
    text = remove_numbers(text)
    text = remove_http(text)
    text = convert_to_lower(text)
    text = remove_white_space(text)
    text = remove_short_words(text) #это убирает не/ни кстати
    # text = remove_punctuation(text) #cлепливает слова, там где знаки препинания без пробела

    text = str(text)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              and len(token)>=3 \
              # and token.strip() not in punctuation \
              and token.isdigit()==False]
#      tokens = " ".join(tokens) #чтобы сделать не список, а строку
    return tokens

#удаляем цифры, http, приводим к нижнему регистру, убираем пробелы, пунктуацию ОСТАВЛЯЕМ короткие слова.
def preprocess_text3(text):
    text = remove_numbers(text)
    text = remove_http(text)
    text = convert_to_lower(text)
    text = remove_white_space(text)
    # text = remove_short_words(text) #это убирает не/ни кстати
    text = remove_punctuation(text) #cлепливает слова, там где знаки препинания без пробела

    text = str(text)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token != " " \
              # and len(token)>=3 \
              and token.strip() not in punctuation \
              and token.isdigit()==False]
#      tokens = " ".join(tokens) #чтобы сделать не список, а строку
    return tokens

#удаляем цифры, http, приводим к нижнему регистру, убираем пробелы, пунктуацию ОСТАВЛЯЕМ короткие слова и стопслова
def preprocess_text4(text):
    text = remove_numbers(text)
    text = remove_http(text)
    text = convert_to_lower(text)
    text = remove_white_space(text)
    # text = remove_short_words(text) #это убирает не/ни кстати
    text = remove_punctuation(text) #cлепливает слова, там где знаки препинания без пробела

    text = str(text)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if #token not in russian_stopwords\
               # and
              token != " " \
              # and len(token)>=3 \
              and token.strip() not in punctuation \
              and token.isdigit()==False]
#      tokens = " ".join(tokens) #чтобы сделать не список, а строку
    return tokens




