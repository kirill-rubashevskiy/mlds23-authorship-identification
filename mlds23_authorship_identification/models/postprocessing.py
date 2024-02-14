import numpy as np


label2name = {
    -1: "не знаю",
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


def confident_predict(proba: np.ndarray, threshold: float = 0.4) -> int:
    if np.max(proba) >= threshold:
        return np.argmax(proba)
    return -1
