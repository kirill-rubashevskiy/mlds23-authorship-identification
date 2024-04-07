from unittest.mock import patch

import nltk
import pytest

from mlds23_authorship_identification.preprocessing import (
    clean_sentence,
    convert_to_lower,
    make_grams,
    make_grams_brief,
    make_punkt,
    remove_http,
    remove_numbers,
    remove_punctuation,
    remove_short_words,
    text_splitter,
    tokenizing,
)


nltk.download("punkt")


@pytest.fixture
def text_data():
    return {
        "input": "123456789012Это пример текста, который будет использоваться в тесте. "
        "Текст содержит более 1000 символов, чтобы можно было проверить, "
        "правильно ли функция разбивает текст на части.",
        "expected_author": "123456789012",
        "expected_text_start": "Это пример текста",
        "expected_rows": 2,
    }


def test_text_splitter(text_data):
    # Разделяем текст на подстроки
    result_df = text_splitter(text_data["input"], 100)
    # Проверяем, что ID автора правильно извлекается
    assert (
        result_df["target"].iloc[0] == text_data["expected_author"]
    ), "Автор не совпадает"
    # Проверяем, что текст начинается с ожидаемого фрагмента
    assert (
        result_df["text"].iloc[0].startswith(text_data["expected_text_start"])
    ), "Текст не начинается с ожидаемого фрагмента"
    # Проверяем количество строк в DataFrame
    assert len(result_df) == text_data["expected_rows"], "Неверное количество подстрок"


@pytest.fixture(
    params=[
        {"input": "TEST", "expected": "test"},
        {"input": "test", "expected": "test"},
        {"input": "TeSt", "expected": "test"},
        {"input": "123 ABC", "expected": "123 abc"},
        {"input": "", "expected": ""},
        {
            "input": "СмЕшАнНыЙ РеГиСтР и кириллица",
            "expected": "смешанный регистр и кириллица",
        },
    ]
)
def text_case(request):
    return request.param


def test_convert_to_lower(text_case):
    input_text = text_case["input"]
    expected_result = text_case["expected"]
    assert (
        convert_to_lower(input_text) == expected_result
    ), f"Expected {expected_result}, but got {convert_to_lower(input_text)}"


@pytest.fixture(
    params=[
        {"input": "123test", "expected": "test"},
        {"input": "test123", "expected": "test"},
        {"input": "te123st", "expected": "test"},
        {"input": "test", "expected": "test"},
        {"input": "1234567890", "expected": ""},
        {"input": "Текст с цифрами 123 и без", "expected": "Текст с цифрами  и без"},
        {"input": "123Текст456без7890цифр", "expected": "Текстбезцифр"},
        {"input": "No numbers here!", "expected": "No numbers here!"},
    ]
)
def number_case(request):
    return request.param


def test_remove_numbers(number_case):
    input_text = number_case["input"]
    expected_result = number_case["expected"]
    assert (
        remove_numbers(input_text) == expected_result
    ), f"Expected {expected_result!r}, but got {remove_numbers(input_text)!r}"


@pytest.fixture(
    params=[
        {
            "input": "Проверьте этот сайт http://t.co/example",
            "expected": "Проверьте этот сайт",
        },
        {
            "input": "Посетите https://t.co/uniqueURL для получения информации",
            "expected": "Посетите для получения информации",
        },
        {
            "input": "Этот текст содержит несколько ссылок http://t.co/first и https://t.co/second",
            "expected": "Этот текст содержит несколько ссылок и",
        },
        {"input": "В этом тексте ссылок нет", "expected": "В этом тексте ссылок нет"},
        {
            "input": "Текст с невалидной ссылкой https://example.com",
            "expected": "Текст с невалидной ссылкой https://example.com",
        },
        {"input": "", "expected": ""},
    ]
)
def http_case(request):
    return request.param


def test_remove_http(http_case):
    input_text = http_case["input"]
    expected_result = http_case["expected"]
    assert (
        remove_http(input_text) == expected_result
    ), f"Expected {expected_result!r}, but got {remove_http(input_text)!r}"


@pytest.fixture(
    params=[
        {"input": "Я и ты пошли в зоопарк", "expected": "пошли зоопарк"},
        {"input": "Тут нет коротких слов", "expected": "Тут нет коротких слов"},
        {"input": "An example with English", "expected": "example with English"},
        {"input": "a bc def ghi j klmn op", "expected": "def ghi klmn"},
        {"input": "12 345 6789 0", "expected": "345 6789"},  # Пример с числами
        {"input": "", "expected": ""},
        {"input": "а я", "expected": ""},
        {"input": "This is a test", "expected": "This test"},
    ]
)
def short_word_case(request):
    return request.param


def test_remove_short_words(short_word_case):
    input_text = short_word_case["input"]
    expected_result = short_word_case["expected"]
    actual_result = remove_short_words(input_text)
    assert (
        actual_result == expected_result
    ), f"Expected {expected_result!r}, but got {actual_result!r}"


@pytest.fixture(
    params=[
        {"input": "Привет, мир!", "expected": "Привет мир"},
        {
            "input": "Тестовый текст: без пунктуации?",
            "expected": "Тестовый текст без пунктуации",
        },
        {
            "input": "Какой-то текст (со скобками).",
            "expected": "Какой-то текст со скобками",
        },
        {
            "input": "Текст[с]разными<знаками>пунктуации.",
            "expected": "Текст с разными знаками пунктуации",
        },
        {"input": "БезПунктуации", "expected": "БезПунктуации"},
        {"input": " ", "expected": ""},
    ]
)
def punctuation_case(request):
    return request.param


def test_remove_punctuation(punctuation_case):
    input_text = punctuation_case["input"]
    expected_result = punctuation_case["expected"]
    actual_result = remove_punctuation(input_text)
    assert (
        actual_result == expected_result
    ), f"Expected {expected_result!r}, but got {actual_result!r}"


# Задаём список стоп-слов напрямую в тесте для примера
russian_stopwords = ["и", "в", "на", "с"]


@pytest.fixture(
    params=[
        {"input": "Я иду в магазин", "expected": ["Я", "иду", "магазин"]},
        {"input": "на столе лежит книга", "expected": ["столе", "лежит", "книга"]},
        {"input": "Солнце светит ярко", "expected": ["Солнце", "светит", "ярко"]},
        {"input": "и в с", "expected": []},  # Только стоп-слова
        {"input": "", "expected": []},  # Пустой ввод
    ]
)
def token_case(request):
    return request.param


def test_tokenizing(token_case):
    input_text = token_case["input"]
    expected_result = token_case["expected"]
    assert (
        tokenizing(input_text) == expected_result
    ), f"Expected {expected_result}, but got {tokenizing(input_text)}"


@pytest.fixture(
    params=[
        {"input": "Привет! Как дела?", "expected": "Привет! Как дела?"},
        {"input": "Тест: 1234567890", "expected": "Тест: "},
        {"input": "English text with numbers 123", "expected": "    "},
        {"input": "Смешанный текст: Test 123", "expected": "Смешанный текст:  "},
        {"input": "Символы: @#$%^&*()", "expected": "Символы: ()"},
        {
            "input": "Пример с дефисами - и тире —",
            "expected": "Пример с дефисами - и тире ",
        },
        {"input": "Кавычки: «Кириллица», 'Latin'", "expected": "Кавычки: Кириллица, ''"},
        {"input": "", "expected": ""},
    ]
)
def sentence_case(request):
    return request.param


def test_clean_sentence(sentence_case):
    input_text = sentence_case["input"]
    expected_result = sentence_case["expected"]
    assert (
        clean_sentence(input_text) == expected_result
    ), f"Expected {expected_result!r}, but got {clean_sentence(input_text)!r}"


@pytest.fixture(
    params=[
        {"input": "Это предложение.", "expected": "Это предложение PNT "},
        {"input": "А это - пример с тире.", "expected": "А это DSH пример с тире PNT "},
        {"input": "Как насчет вопроса?", "expected": "Как насчет вопроса QST "},
        {"input": "Восклицание!", "expected": "Восклицание EXCL "},
        {"input": "Двоеточие: так", "expected": "Двоеточие CL  так"},
        {"input": "Точка с запятой;", "expected": "Точка с запятой SMC "},
        {"input": "Эллипсис...", "expected": "Эллипсис MP "},
        {"input": "Двойной эллипсис..", "expected": "Двойной эллипсис MP "},
        {"input": "Запятая, да", "expected": "Запятая CM  да"},
        {
            "input": ", двадцать-два, - номер, помереть: не сейчас!",
            "expected": " CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL ",
        },
    ]
)
def punkt_case(request):
    return request.param


def test_make_punkt(punkt_case):
    input_text = punkt_case["input"]
    expected_result = punkt_case["expected"]
    actual_result = make_punkt(input_text)
    assert (
        actual_result == expected_result
    ), f"Expected {expected_result!r}, but got {actual_result!r}"


@pytest.fixture
def mock_mystem_analyze():
    # Эмулируем результат работы Mystem.analyze
    return [
        {"text": " CM ", "analysis": [{"gr": "CM"}]},
        {"text": " двадцать-два ", "analysis": [{"gr": "NUM=(вин|им)"}]},
        {"text": " DSH ", "analysis": [{"gr": "DSH"}]},
        # Добавьте дополнительные мокированные данные для других слов
    ]


def test_make_grams_with_mocking(mock_mystem_analyze):
    # Мокируем вызовы к Mystem.analyze, возвращая заранее подготовленные данные
    with patch(
        "mlds23_authorship_identification.preprocessing.Mystem.analyze",
        return_value=mock_mystem_analyze,
    ):
        sentence = " CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL "
        expected = (
            "CM NUM=(вин|им) DSH"  # Упрощенный ожидаемый результат для демонстрации
        )
        result = make_grams(sentence)
        assert result == expected, f"Expected {expected!r}, but got {result!r}"


# Мокируемый результат работы Mystem.analyze для тестового предложения
mocked_mystem_output = [
    {"text": " CM ", "analysis": [{"gr": "CONJ="}]},
    {"text": " двадцать-два ", "analysis": [{"gr": "NUM=им"}]},
    {"text": " DSH ", "analysis": []},
    {"text": " номер ", "analysis": [{"gr": "S,муж,неод=им,ед"}]},
    {"text": " CM ", "analysis": [{"gr": "CONJ="}]},
    {"text": " помереть ", "analysis": [{"gr": "V,нп=инф,сов"}]},
    {"text": " CL ", "analysis": []},
    {"text": " не ", "analysis": [{"gr": "PART="}]},
    {"text": " сейчас ", "analysis": [{"gr": "ADV="}]},
    {"text": " EXCL ", "analysis": []},
]


@pytest.fixture
def input_and_expected():
    return {
        "input": " CM  двадцать-два CM  DSH номер CM  помереть CL  не сейчас EXCL ",
        "expected": "CONJ NUM S,муж,неод CONJ V,нп PART ADV",
    }


def test_make_grams_brief(input_and_expected):
    with patch(
        "mlds23_authorship_identification.preprocessing.Mystem.analyze",
        return_value=mocked_mystem_output,
    ):
        sentence = input_and_expected["input"]
        expected = input_and_expected["expected"]
        result = make_grams_brief(sentence)
        assert result == expected, f"Expected {expected!r}, but got {result!r}"
