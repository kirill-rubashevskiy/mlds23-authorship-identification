import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

import wandb
from mlds23_authorship_identification.classifiers import clfs, ensembles


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: OmegaConf):

    # initialize wandb run
    wandb.init(
        entity=cfg.wandb.entity, project=cfg.wandb.project, tags=cfg.experiment.tags
    )
    logging.info("W&B run initialized")

    # load data
    train = pd.read_csv(f"data/{cfg.experiment.train}", index_col=0)
    test = pd.read_csv(f"data/{cfg.experiment.test}", index_col=0)
    X_train = train["text"]
    y_train = train["target"]
    groups_train = train["book"]
    X_test = test["text"]
    y_test = test["target"]
    logging.info("Data loaded")

    # load model
    if cfg.experiment.classifier in clfs:
        model = clfs[cfg.experiment.classifier]
    else:
        model = ensembles[cfg.experiment.classifier]
    logging.info("Model loaded")

    # run random search
    r_search = RandomizedSearchCV(
        estimator=model.pipeline,
        param_distributions=model.params,
        random_state=cfg.random_state,
        cv=StratifiedGroupKFold(n_splits=cfg.experiment.n_splits),
        n_iter=cfg.experiment.n_iter,
        scoring=cfg.experiment.metric,
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
        "model": cfg.experiment.classifier,
        "train": cfg.experiment.train,
        "test": cfg.experiment.test,
        "random_state": cfg.random_state,
        "cv": cfg.experiment.cv,
        "n_splits": cfg.experiment.n_splits,
    }
    logging.info("Run params logged")

    # log model best params
    params_to_log = dict()
    for key, value in r_search.best_params_.items():
        step, param = key.split("__")
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
