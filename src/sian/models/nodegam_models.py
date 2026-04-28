import os
import pandas as pd
import numpy as np
import sys
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import nodegam
from nodegam.sklearn import NodeGAMRegressor, NodeGAMClassifier

import sklearn
from sklearn.metrics import (
    mean_squared_error, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score
)

def train_node_ga2m(dataset_obj, exp_folder, IS_GA2M=0, seed=0):
    print(f"Running training with seed = {seed}")
    output_type = dataset_obj.get_task_type()

    trnval_perc = 0.70
    trnval_seed = seed
    dataset_obj.shuffle_and_split_trnval(trnval_seed, trnval_perc)

    X_train_val, y_train_val, _, _ = dataset_obj.pull_data()
    X_train_val = pd.DataFrame(X_train_val).astype(np.float32)
    y_train_val = y_train_val.ravel().astype(np.int64)

    class_count = len(np.unique(y_train_val))

    if output_type == "regression":
        model = NodeGAMRegressor(
            in_features=X_train_val.shape[1],
            validation_size=0.3,
            ga2m=IS_GA2M,
            seed=seed
        )
    else:
        model = NodeGAMClassifier(
            in_features=X_train_val.shape[1],
            num_classes=1 if class_count == 2 else class_count,
            validation_size=0.3,
            ga2m=IS_GA2M,
            seed=seed
        )

    os.makedirs(exp_folder, exist_ok=True)

    hyperparams = {
        'in_features': X_train_val.shape[1],
        'validation_size': 0.3,
        'seed': seed,
        'task_type': output_type,
        'num_classes': class_count if output_type != 'regression' else "N/A",
        'ga2m': IS_GA2M
    }

    type_of_node_gam = 2 if IS_GA2M == 1 else 1

    pd.DataFrame([hyperparams]).to_csv(
        os.path.join(exp_folder, f"node_ga{type_of_node_gam}m_hyperparams.csv"),
        index=False
    )

    start_time = time.time()
    model.fit(X_train_val, y_train_val)
    end_time = time.time()

    total_training_time = end_time - start_time

    with open(os.path.join(exp_folder, f"node_ga{type_of_node_gam}m_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"Training complete. Total training time: {total_training_time:.2f}s")


def test_node_ga2m(dataset_obj, exp_folder, IS_GA2M=0, seed=0):
    print(f"Running testing with seed = {seed}")
    output_type = dataset_obj.get_task_type()
    _, _, X_test, y_test = dataset_obj.pull_data()

    X_test = pd.DataFrame(X_test).astype(np.float32)
    y_test = y_test.ravel().astype(np.int64)

    class_count = len(np.unique(y_test))
    type_of_node_gam = 2 if IS_GA2M == 1 else 1

    with open(os.path.join(exp_folder, f"node_ga{type_of_node_gam}m_model.pkl"), "rb") as f:
        model = pickle.load(f)

    y_pred_test = model.predict(X_test)

    metrics = {}

    if output_type == "regression":
        metrics['Test MSE'] = mean_squared_error(y_test, y_pred_test)
        print("Test MSE:", metrics['Test MSE'])
    else:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            y_pred_test = np.argmax(y_proba, axis=1)

        metrics['Test Accuracy'] = accuracy_score(y_test, y_pred_test)
        metrics['Test Precision'] = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        metrics['Test Recall'] = recall_score(y_test, y_pred_test, average='weighted')
        metrics['Test F1 Score'] = f1_score(y_test, y_pred_test, average='weighted')

        try:
            if class_count == 2:
                metrics['Test ROC AUC'] = roc_auc_score(y_test, y_proba[:, 1])
                metrics['Test AUPRC'] = average_precision_score(y_test, y_proba[:, 1])
                print("Test ROC AUC", metrics['Test ROC AUC'])
            else:
                metrics['Test ROC AUC'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                metrics['Test AUPRC'] = average_precision_score(y_test, y_proba, average='weighted')
                print("Test ROC AUC", metrics['Test ROC AUC'])
        except Exception as e:
            print("ROC/AUPRC error:", e)
            metrics['Test ROC AUC'] = None
            metrics['Test AUPRC'] = None

    pd.DataFrame([metrics]).to_csv(
        os.path.join(exp_folder, f"node_ga{type_of_node_gam}m_metrics.csv"),
        index=False
    )
