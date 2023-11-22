# dummy-модель
import numpy as np
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(strategy="constant", constant="Похоже на Пушкина").fit(
    X=np.array(['text_fragment']), y=["Похоже на Пушкина"])

