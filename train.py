import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

import wandb
from mlds23_authorship_identification.classifiers import clfs


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: OmegaConf):

    # initialize wandb run
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, tags=cfg.run.tags)
    logging.info("W&B run initialized")

    # load data
    train = pd.read_csv(f"data/{cfg.run.train}", index_col=0)
    test = pd.read_csv(f"data/{cfg.run.test}", index_col=0)
    X_train = train["text"]
    y_train = train["target"]
    groups_train = train["book"]
    X_test = test["text"]
    y_test = test["target"]
    logging.info("Data loaded")

    # load model
    model = clfs[cfg.run.clf]
    logging.info("Model loaded")

    # run random search
    r_search = RandomizedSearchCV(
        estimator=model.pipeline,
        param_distributions=model.params,
        random_state=cfg.random_state,
        cv=StratifiedGroupKFold(n_splits=cfg.run.n_splits),
        n_iter=cfg.run.n_iter,
        scoring=cfg.run.metric,
        n_jobs=-1,
    )
    r_search.fit(X_train, y_train, groups=groups_train)
    best_index = r_search.best_index_
    f1_weighted_mean = np.round(r_search.cv_results_["mean_test_score"][best_index], 2)
    f1_weighted_std = np.round(r_search.cv_results_["std_test_score"][best_index], 2)
    logging.info("Random search completed")
    logging.info(f"Best mean score: {f1_weighted_mean}±{f1_weighted_std}")

    # test model on test data
    y_pred = r_search.predict(X_test)
    f1_weighted_test = np.round(f1_score(y_test, y_pred, average="weighted"), 2)
    logging.info("Tuned model tested")
    logging.info(f"Test score: {f1_weighted_test}")

    # log run params
    wandb.config["run"] = {
        "model": cfg.run.clf,
        "train": cfg.run.train,
        "test": cfg.run.test,
        "random_state": cfg.random_state,
        "cv": cfg.run.cv,
        "n_splits": cfg.run.n_splits,
    }
    logging.info("Run params logged")

    # log model best params
    params_to_log = dict()
    for key, value in r_search.best_params_.items():
        *_, step, param = key.split("__")
        if step not in params_to_log:
            params_to_log[step] = dict()
        params_to_log[step][param] = value
    for step, params in params_to_log.items():
        wandb.config[step] = params
    logging.info("Model best params logged")

    # log metrics
    wandb.log(
        {
            "train": {
                "f1_weighted_mean": f1_weighted_mean,
                "f1_weighted_std": f1_weighted_std,
            },
            "test": {"f1_weighted": f1_weighted_test},
        }
    )
    logging.info("Metrics logged")

    # завершение эксперимента
    wandb.finish()
    logging.info("W&B run finished")


if __name__ == "__main__":
    main()
