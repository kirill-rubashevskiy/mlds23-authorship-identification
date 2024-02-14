import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


class GroupCVSplitter:
    def __init__(self, n_splits: int, y: pd.Series, groups: pd.Series):
        """
        CV splitter, чтобы одни и те же произведения не попадали в трейн и тест
        :param n_splits: количество сплитов, не может быть меньше, чем минимально количество произведений у автора
        :param y: целевая переменная
        :param groups: столбец с признаком произведение
        """
        self.n_splits = n_splits
        self.y = y
        self.groups = groups

    def split(self):
        """
        Функция, которую нужно вызывать при указании instance класса в качестве параметра CV
        :return:
        """
        start_indices = []
        splitters = []
        for label in np.unique(self.y):
            label_y = self.y[self.y == label]
            start_indices.append(label_y.index.min())
            splitters.append(
                GroupKFold(n_splits=self.n_splits).split(
                    label_y, groups=self.groups[label_y.index]
                )
            )

        for _ in range(self.n_splits):
            train = np.array([], dtype=int)
            test = np.array([], dtype=int)

            for j, splitter in enumerate(splitters):
                label_train, label_test = next(splitter)

                train = np.append(train, label_train + start_indices[j])
                test = np.append(test, label_test + start_indices[j])

            yield train, test
