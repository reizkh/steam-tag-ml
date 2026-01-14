from src.dataset import SteamAppsDataset
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier
import sklearn.metrics
import omegaconf
import mlflow
import numpy as np
from random import sample
from torch.utils.data import random_split
import torch
import re


def run_baseline(X_train, X_test, y_train, y_test):
    model = MultiOutputClassifier(DummyClassifier())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    hamming_accuracy = float(1 - sklearn.metrics.hamming_loss(y_test, y_pred))
    clf_report = sklearn.metrics.classification_report(
        y_test, y_pred, 
        output_dict=True, 
        target_names=list(dataset.tag_label2id.keys()),
        zero_division=1
    )
    clf_report = {
        "val_" + (re.sub(r"[^A-Za-z _]", "", "__".join([cls, key]))): val 
        for cls, metrics in clf_report.items() for key, val in metrics.items() # type: ignore
        if cls in ["macro avg", "weighted avg"]
    }

    mlflow.log_metric("val_accuracy", hamming_accuracy)
    mlflow.log_metrics(clf_report)


if __name__ == "__main__":
    db_params = omegaconf.OmegaConf.load("configs/db_params.yaml")
    cfg = omegaconf.OmegaConf.load("configs/baseline.yaml")
    
    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run():
        mlflow.log_param("seed", cfg.random_seed)
        mlflow.log_param("train samples", cfg.p_train_samples)

        dataset = SteamAppsDataset(db_params)
        n_train_samples = int(len(dataset)*cfg.p_train_samples)
        rng = torch.Generator().manual_seed(cfg.random_seed)

        train, test = random_split(dataset, [n_train_samples, len(dataset)-n_train_samples], generator=rng)
        X_train, y_train = map(list, zip(*train))
        X_test, y_test = map(list, zip(*test))

        run_baseline(X_train, X_test, y_train, y_test)