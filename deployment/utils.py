import re
from pymystem3 import Mystem
from nltk.tokenize import RegexpTokenizer

mystem = Mystem()
tokenizer = RegexpTokenizer(r'[A-Za-zА-яЁё]+')


def remove_http(text):
    text = re.sub("https?:\/\/t.co\/[A-Za-zа-яА-Я0-9]*", ' ', text)
    return text


def convert_to_lower(text):
    return text.lower()


def preprocess_text5(text: str, lemmatize: bool = False) -> str:

    text = remove_http(text)
    text = convert_to_lower(text)

    text = ' '.join(tokenizer.tokenize(text))  # убирает цифры и пунктуацию

    text = text.replace('\n', '').replace('\r', '')

    if lemmatize:
        text = ' '.join(mystem.lemmatize(text))

    return text


label2name = {
    0: 'А. Пушкин',
    1: 'Д. Мамин-Сибиряк',
    2: 'И. Тургенев',
    3: 'А. Чехов',
    4: 'Н. Гоголь',
    5: 'И. Бунин',
    6: 'А. Куприн',
    7: 'А. Платонов',
    8: 'В. Гаршин',
    9: 'Ф. Достоевский'
}
