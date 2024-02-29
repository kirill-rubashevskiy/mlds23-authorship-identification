from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .preprocessing import TextTransformer


@dataclass
class Classifier:
    pipeline: Pipeline
    params: dict


# TextTransformer params search space
preprocessing_params = {
    "preprocessing__remove_punctuation": [True, False],
    "preprocessing__remove_stopwords": [True, False],
    "preprocessing__lemmatize": [True, False],
}

# TfidfVectorizer and CountVectorizer params search space
vectorizer_params = {
    "vectorizer__min_df": [1, 5, 10, 20],
    "vectorizer__max_df": [0.5, 0.75, 1.0],
    "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
}

clfs = {
    "TF-IDF LR": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", TfidfVectorizer()),
                ("classifier", LogisticRegression(solver="saga", max_iter=2500)),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__C": np.logspace(-3, 3, 10),
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "BoW LR": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", CountVectorizer()),
                ("classifier", LogisticRegression(solver="saga", max_iter=4000)),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__C": np.logspace(-3, 3, 10),
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "TF-IDF MultinomialNB": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", TfidfVectorizer()),
                ("classifier", MultinomialNB()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__alpha": np.logspace(-3, 3, 10),
        },
    ),
    "BoW MultinomialNB": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", CountVectorizer()),
                ("classifier", MultinomialNB()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__alpha": np.logspace(-3, 3, 10),
        },
    ),
    "BoW ComplementNB": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", CountVectorizer()),
                ("classifier", ComplementNB()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__alpha": np.logspace(-3, 3, 10),
            "classifier__norm": [True, False],
        },
    ),
    "TF-IDF SVD GaussianNB": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", TfidfVectorizer()),
                ("dim_reduction", TruncatedSVD()),
                ("classifier", GaussianNB()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "dim_reduction__n_components": np.linspace(20, 200, 10, dtype=int),
            "classifier__var_smoothing": np.logspace(-9, 0, 10),
        },
    ),
    "BoW GaussianNB": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", CountVectorizer()),
                ("classifier", GaussianNB()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__var_smoothing": np.logspace(-9, 0, 10),
        },
    ),
    "TF-IDF SVM": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", TfidfVectorizer()),
                ("classifier", SVC()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__C": np.logspace(-3, 3, 10),
            "classifier__kernel": ["linear", "rbf"],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "TF-IDF Scaled SVM": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", TfidfVectorizer()),
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", SVC()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__C": np.logspace(-3, 3, 10),
            "classifier__kernel": ["linear", "rbf"],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "BoW SVM": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", CountVectorizer()),
                ("classifier", SVC()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__C": np.logspace(-3, 3, 10),
            "classifier__kernel": ["linear", "rbf"],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "TF-IDF SVD Scaled KNN": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", TfidfVectorizer()),
                ("dim_reduction", TruncatedSVD()),
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", KNeighborsClassifier(metric="cosine")),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "dim_reduction__n_components": np.linspace(20, 200, 10, dtype=int),
            "classifier__n_neighbors": [5, 10, 25, 50],
            "classifier__weights": ["uniform", "distance"],
        },
    ),
    "BoW Scaled KNN": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", CountVectorizer()),
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", KNeighborsClassifier(metric="cosine")),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "classifier__n_neighbors": [5, 10, 25, 50],
            "classifier__weights": ["uniform", "distance"],
        },
    ),
    "TF-IDF SVD RF": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", TfidfVectorizer()),
                ("dim_reduction", TruncatedSVD()),
                ("classifier", RandomForestClassifier()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "dim_reduction__n_components": np.linspace(20, 200, 10, dtype=int),
            "classifier__n_estimators": np.linspace(100, 300, 5, dtype=int),
            "classifier__max_depth": [5, 10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__max_features": ["sqrt", "log2"],
        },
    ),
    "BoW SVD RF": Classifier(
        pipeline=Pipeline(
            [
                ("preprocessing", TextTransformer()),
                ("vectorizer", CountVectorizer()),
                ("dim_reduction", TruncatedSVD()),
                ("classifier", RandomForestClassifier()),
            ]
        ),
        params={
            **preprocessing_params,
            **vectorizer_params,
            "dim_reduction__n_components": np.linspace(20, 200, 10, dtype=int),
            "classifier__n_estimators": np.linspace(100, 300, 5, dtype=int),
            "classifier__max_depth": [5, 10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__max_features": ["sqrt", "log2"],
        },
    ),
}

ensembles = {
    "BoW MultinomialNB + TF-IDF LR + VC": Classifier(
        pipeline=Pipeline(
            [
                (
                    "classifier",
                    VotingClassifier(
                        estimators=[
                            (
                                "BoW MultinomialNB",
                                Pipeline(
                                    [
                                        (
                                            "preprocessing",
                                            TextTransformer(
                                                remove_punctuation=False,
                                                remove_stopwords=False,
                                            ),
                                        ),
                                        (
                                            "vectorizer",
                                            CountVectorizer(
                                                min_df=20, max_df=1.0, ngram_range=(1, 2)
                                            ),
                                        ),
                                        (
                                            "classifier",
                                            MultinomialNB(alpha=0.46415888336127775),
                                        ),
                                    ]
                                ),
                            ),
                            (
                                "TF-IDF LR",
                                Pipeline(
                                    [
                                        (
                                            "preprocessing",
                                            TextTransformer(
                                                remove_punctuation=False,
                                                remove_stopwords=False,
                                            ),
                                        ),
                                        (
                                            "vectorizer",
                                            TfidfVectorizer(
                                                min_df=20, max_df=0.5, ngram_range=(1, 3)
                                            ),
                                        ),
                                        (
                                            "classifier",
                                            LogisticRegression(
                                                solver="saga",
                                                max_iter=2500,
                                                C=46.41588833612773,
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                )
            ]
        ),
        params={"classifier__voting": ["hard", "soft"]},
    )
}
