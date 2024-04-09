import logging

import fire
import joblib
import numpy as np
from hydra import compose, initialize
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

import wandb
from mlds23_authorship_identification.classifiers import clfs
from mlds23_authorship_identification.utils import (
    load_data,
    load_model,
    save_model,
    wb_log,
)


def train(
    model_name: str,
    train_data: str = "split_1000_train.csv",
    test_data: str = "split_1000_test.csv",
    save_fitted_model: bool = True,
    log_experiment: bool = False,
    **model_params,
) -> None:
    """
    Fits model on train data.

    Args:
        model_name: A name of a model from mlds23_authorship_identification.classifiers.clfs.
        train_data: A name of train data file.
        test_data: A name of test data file.
        save_fitted_model: Whether to save the fitted model locally.
        log_experiment: Whether to log the experiment to Weights & Biases.
        model_params: The hyperparameters of model.
    """

    # initialize Hydra config
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")

    if log_experiment:
        # initialize Weights & Biases experiment
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        logging.info("W&B run initialized")

    # load train data
    train_df = load_data(train_data)
    X_train = train_df["text"]
    y_train = train_df["target"]
    logging.info("Train data loaded")

    # initialize model and set hyperparameters
    model = clfs[model_name].pipeline
    if model_params:
        model.set_params(**model_params)
    logging.info("Model initialized")

    # fit model on train data
    model.fit(X_train, y_train)
    logging.info("Model fitted")

    if save_fitted_model:
        # save model locally
        save_model(model_name, model)
        logging.info("Model saved")

    if log_experiment:
        # calculate test score
        test_score = infer(model, test_data=test_data, return_score=True)

        # log experiment params, model params and score to Weights & Biases
        wb_log(
            config={"train": train_data, "test": test_data, "model": model_name},
            params=model_params,
            scores={"test": {"f1_weighted": test_score}},
        )
        logging.info("Experiment logged")

        # finish Weights & Biases experiment
        wandb.finish()
        logging.info("W&B run finished")


def infer(
    model_or_filepath, test_data: str = "split_1000_test.csv", return_score: bool = False
) -> None | float:
    """
    Tests model on test data.

    Args:
        model_or_filepath: Model instance or model filepath.
        test_data: A name of test data file.
        return_score: Whether to return test score

    Returns: test score (optional, if return_score is True)
    """

    # load test data
    test_df = load_data(test_data)
    X_test = test_df["text"]
    y_test = test_df["target"]
    logging.info("Test data loaded")

    # load model
    if isinstance(model_or_filepath, BaseEstimator):
        model = model_or_filepath
    elif model_or_filepath.startswith("models/tmp/"):
        model = joblib.load(model_or_filepath)
    else:
        model = load_model(model_or_filepath.split("/")[-1])
    logging.info("Model loaded")

    # make predictions
    y_pred = model.predict(X_test)

    # calculate score
    f1_weighted_score = np.round(f1_score(y_test, y_pred, average="weighted"), 2)
    logging.info("Model tested")
    logging.info(f"The weighted-averaged F1 test score: {f1_weighted_score}")

    if return_score:
        return f1_weighted_score


def randomsearch(
    model_name,
    train_data: str = "split_1000_train.csv",
    test_data: str = "split_1000_test.csv",
    n_splits: int = 5,
    n_iter: int = 10,
    scoring: str = "f1_weighted",
    save_best_model: bool = True,
    log_experiment: bool = False,
):
    """
    Runs random search

    Args:
        model_name: A name of a model from mlds23_authorship_identification.classifiers.clfs.
        train_data: A name of train data file.
        test_data: A name of test data file.
        n_splits: Number of folds of StratifiedGroupKFold.
        n_iter: Number of parameter settings that are sampled.
        scoring: Strategy to evaluate the performance of the cross-validated model on the test set.
        save_best_model: Whether to save the fitted model locally.
        log_experiment: Whether to log the experiment to Weights & Biases.
    """

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")

    if log_experiment:
        wandb.init(
            entity=cfg.wandb.entity, project=cfg.wandb.project, tags=["random search"]
        )
        logging.info("W&B run initialized")

    # load train data
    train_df = load_data(train_data)
    X_train = train_df["text"]
    y_train = train_df["target"]
    groups_train = train_df["book"]
    logging.info("Train data loaded")

    # initialize model
    model = clfs[model_name].pipeline
    logging.info("Model initialized")

    # run random search
    r_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=clfs[model_name].params,
        random_state=cfg.random_state,
        cv=StratifiedGroupKFold(n_splits=n_splits),
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=-1,
    )
    r_search.fit(X_train, y_train, groups=groups_train)
    logging.info("Random search completed")

    best_model = r_search.best_estimator_

    # calculate cross-validated score of the best estimator
    best_index = r_search.best_index_
    f1_weighted_mean = np.round(r_search.cv_results_["mean_test_score"][best_index], 2)
    f1_weighted_std = np.round(r_search.cv_results_["std_test_score"][best_index], 2)
    logging.info(
        f"Mean cross-validated score of the best estimator: {f1_weighted_mean}±{f1_weighted_std}"
    )

    if save_best_model:
        # save model locally
        save_model(model_name, best_model)
        logging.info("Model saved")

    if log_experiment:
        # calculate test score
        test_score = infer(best_model, test_data=test_data, return_score=True)

        # log experiment params, model params and score to Weights & Biases
        wb_log(
            config={
                "train": train_data,
                "test": test_data,
                "model": model_name,
                "random_state": cfg.random_state,
                "cv": "StratifiedGroupKFold",
                "n_splits": n_splits,
                "n_iter": n_iter,
            },
            params=r_search.best_params_,
            scores={
                "train": {
                    "f1_weighted_mean": f1_weighted_mean,
                    "f1_weighted_std": f1_weighted_std,
                },
                "test": {"f1_weighted": test_score},
            },
        )
        logging.info("Experiment logged")

        # finish Weights & Biases experiment
        wandb.finish()
        logging.info("W&B run finished")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire()
