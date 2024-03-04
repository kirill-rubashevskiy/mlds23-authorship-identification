from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

from .extractors import TextStatsExtractor
from .preprocessing import TextTransformer


@dataclass
class Classifier:
    pipeline: Pipeline
    params: dict


# TextTransformer params search space
texttransformer_params = {
    "texttransformer__remove_punctuation": [True, False],
    "texttransformer__remove_stopwords": [True, False],
    "texttransformer__lemmatize": [True, False],
}

# TfidfVectorizer and CountVectorizer params search space
vectorizer_params = {
    "vectorizer__min_df": [5, 10, 20, 30],
    "vectorizer__max_df": [0.5, 0.75, 1.0],
    "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
}

textstatsextractor_params = {
    "textstatsextractor__pos_stats": [True, False],
    "textstatsextractor__sent_stats": [True, False],
}

truncatedsvd_params = {"truncatedsvd__n_components": np.linspace(20, 200, 10, dtype=int)}

clfs = {
    "TF-IDF stats poly LR": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    ("embed", make_pipeline(TextTransformer(), TfidfVectorizer())),
                    (
                        "stats",
                        make_pipeline(TextStatsExtractor(), PolynomialFeatures(degree=2)),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            LogisticRegression(solver="saga", max_iter=1000),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__tfidf{k}": v for k, v in vectorizer_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "logisticregression__C": np.logspace(-3, 3, 20),
            "logisticregression__class_weight": [None, "balanced"],
        },
    ),
    "BoW stats poly LR": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    ("embed", make_pipeline(TextTransformer(), CountVectorizer())),
                    (
                        "stats",
                        make_pipeline(TextStatsExtractor(), PolynomialFeatures(degree=2)),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            LogisticRegression(solver="saga", max_iter=1000),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__count{k}": v for k, v in vectorizer_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "logisticregression__C": np.logspace(-3, 3, 20),
            "logisticregression__class_weight": [None, "balanced"],
        },
    ),
    "TF-IDF stats poly SVM": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    ("embed", make_pipeline(TextTransformer(), TfidfVectorizer())),
                    (
                        "stats",
                        make_pipeline(TextStatsExtractor(), PolynomialFeatures(degree=2)),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            SVC(),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__tfidf{k}": v for k, v in vectorizer_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "svc__C": np.logspace(-3, 3, 20),
            "svc__class_weight": [None, "balanced"],
            "svc__kernel": ["linear", "rbf"],
        },
    ),
    "BoW stats poly SVM": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    ("embed", make_pipeline(TextTransformer(), CountVectorizer())),
                    (
                        "stats",
                        make_pipeline(TextStatsExtractor(), PolynomialFeatures(degree=2)),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            SVC(),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__count{k}": v for k, v in vectorizer_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "svc__C": np.logspace(-3, 3, 20),
            "svc__class_weight": [None, "balanced"],
            "svc__kernel": ["linear", "rbf"],
        },
    ),
    "TF-IDF stats poly MultinomialNB": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    ("embed", make_pipeline(TextTransformer(), TfidfVectorizer())),
                    (
                        "stats",
                        make_pipeline(TextStatsExtractor(), PolynomialFeatures(degree=2)),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            MultinomialNB(),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__tfidf{k}": v for k, v in vectorizer_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "multinomialnb__alpha": np.logspace(-3, 3, 20),
        },
    ),
    "BoW stats poly MultinomialNB": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    ("embed", make_pipeline(TextTransformer(), CountVectorizer())),
                    (
                        "stats",
                        make_pipeline(TextStatsExtractor(), PolynomialFeatures(degree=2)),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            MultinomialNB(),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__count{k}": v for k, v in vectorizer_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "multinomialnb__alpha": np.logspace(-3, 3, 20),
        },
    ),
    "TF-IDF stats KNN": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    (
                        "embed",
                        make_pipeline(
                            TextTransformer(), TfidfVectorizer(), TruncatedSVD()
                        ),
                    ),
                    ("stats", make_pipeline(TextStatsExtractor())),
                ]
            ),
            StandardScaler(with_mean=False),
            KNeighborsClassifier(metric="cosine"),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__tfidf{k}": v for k, v in vectorizer_params.items()},
            **{f"featureunion__embed__{k}": v for k, v in truncatedsvd_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "kneighborsclassifier__n_neighbors": [5, 10, 25, 50],
            "kneighborsclassifier__weights": ["uniform", "distance"],
        },
    ),
    "BoW stats KNN": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    (
                        "embed",
                        make_pipeline(
                            TextTransformer(), CountVectorizer(), TruncatedSVD()
                        ),
                    ),
                    (
                        "stats",
                        make_pipeline(
                            TextStatsExtractor(),
                        ),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            KNeighborsClassifier(metric="cosine"),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__count{k}": v for k, v in vectorizer_params.items()},
            **{f"featureunion__embed__{k}": v for k, v in truncatedsvd_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "kneighborsclassifier__n_neighbors": [5, 10, 25, 50],
            "kneighborsclassifier__weights": ["uniform", "distance"],
        },
    ),
    "TF-IDF stats poly RF": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    (
                        "embed",
                        make_pipeline(
                            TextTransformer(), TfidfVectorizer(), TruncatedSVD()
                        ),
                    ),
                    (
                        "stats",
                        make_pipeline(TextStatsExtractor(), PolynomialFeatures(degree=2)),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            RandomForestClassifier(),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__tfidf{k}": v for k, v in vectorizer_params.items()},
            **{f"featureunion__embed__{k}": v for k, v in truncatedsvd_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "randomforestclassifier__n_estimators": np.linspace(100, 300, 5, dtype=int),
            "randomforestclassifier__max_depth": [5, 10, 20, None],
            "randomforestclassifier__min_samples_leaf": [1, 2, 4, 8],
            "randomforestclassifier__max_features": ["sqrt", "log2"],
        },
    ),
    "BoW stats poly RF": Classifier(
        pipeline=make_pipeline(
            FeatureUnion(
                [
                    (
                        "embed",
                        make_pipeline(
                            TextTransformer(), CountVectorizer(), TruncatedSVD()
                        ),
                    ),
                    (
                        "stats",
                        make_pipeline(TextStatsExtractor(), PolynomialFeatures(degree=2)),
                    ),
                ]
            ),
            StandardScaler(with_mean=False),
            RandomForestClassifier(),
        ),
        params={
            **{f"featureunion__embed__{k}": v for k, v in texttransformer_params.items()},
            **{f"featureunion__embed__count{k}": v for k, v in vectorizer_params.items()},
            **{f"featureunion__embed__{k}": v for k, v in truncatedsvd_params.items()},
            **{
                f"featureunion__stats__{k}": v
                for k, v in textstatsextractor_params.items()
            },
            "randomforestclassifier__n_estimators": np.linspace(100, 300, 5, dtype=int),
            "randomforestclassifier__max_depth": [5, 10, 20, None],
            "randomforestclassifier__min_samples_leaf": [1, 2, 4, 8],
            "randomforestclassifier__max_features": ["sqrt", "log2"],
        },
    ),
}
