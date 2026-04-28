import optuna
import os
import time
import joblib
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sian.utils import get_device_details, get_cpu_details

try:
    from interpret.glassbox import ExplainableBoostingRegressor, ExplainableBoostingClassifier
except ImportError as e:
    print(f"Failed to import EBM: {e}")
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError as e:
    print(f"Failed to import Catboost: {e}")
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError as e:
    print(f"Failed to import LightGBM: {e}")




def int_param(trial, name, low, high):
    return trial.suggest_int(name, low, high)
def float_param(trial, name, low, high):
    return trial.suggest_float(name, low, high)
def categorical_param(trial, name, choices):
    return trial.suggest_categorical(name, choices)




### COMMON SKLEARN HYPERPARAMETERS TO TUNE OVER ###
def get_common_rf_params(trial, task_type, model_seed=0):
    print(f"Running Common RF Params with model_seed = {model_seed}")
    params = {
        'n_estimators': int_param(trial, "n_estimators", 100, 1000),
        'max_depth': int_param(trial, "max_depth", 5, 30),
        'min_samples_split': int_param(trial, "min_samples_split", 2, 20),
        'min_samples_leaf': int_param(trial, "min_samples_leaf", 1, 10),
        'max_features': categorical_param(trial, "max_features", ['sqrt', 'log2']),
        'bootstrap': categorical_param(trial, "bootstrap", [True, False]),
        'random_state': categorical_param(trial, "random_state", [model_seed])
    }
    if params['bootstrap']:
        params['max_samples'] = float_param(trial, "max_samples", 0.5, 1.0)
    return params

def get_common_svm_params(trial, task_type, model_seed=0):
    print(f"Running Common SVM Params with model_seed = {model_seed}")
    params = {
        'kernel': categorical_param(trial, "kernel", ['linear', 'poly', 'rbf', 'sigmoid']),
        'C': float_param(trial, "C", 0.1, 30),
        'gamma': categorical_param(trial, "gamma", ['scale', 'auto'])
    }
    if task_type == "regression":
        params['epsilon'] = float_param(trial, "epsilon", 0.01, 0.5)
    else:
        params.update({
            'probability': categorical_param(trial, "probability", [True]),
            'random_state': categorical_param(trial, "random_state", [model_seed])
        })
    return params

def get_common_xgb_params(trial, task_type, model_seed=0):
    print(f"Running Common XGB Params with model_seed = {model_seed}")
    params = {
        'n_estimators': int_param(trial, "n_estimators", 100, 1000),
        'learning_rate': float_param(trial, "learning_rate", 0.01, 0.3),
        'max_depth': int_param(trial, "max_depth", 3, 15),
        'min_child_weight': int_param(trial, "min_child_weight", 1, 10),
        'subsample': float_param(trial, "subsample", 0.5, 1.0),
        'colsample_bytree': float_param(trial, "colsample_bytree", 0.5, 1.0),
        'gamma': float_param(trial, "gamma", 0, 5),
        'reg_alpha': float_param(trial, "reg_alpha", 0, 10),
        'reg_lambda': float_param(trial, "reg_lambda", 0, 10),
        'tree_method': categorical_param(trial, "tree_method", ["hist"]),
        'device': categorical_param(trial, "device", ["cuda"]),
        'random_state': categorical_param(trial, "random_state", [model_seed])
    }
    if task_type.find("classification") != -1:
        params.update({
            'use_label_encoder': categorical_param(trial, "use_label_encoder", [False]),
            'eval_metric': categorical_param(trial, "eval_metric", ['logloss'])
        })
    return params

def get_common_catboost_params(trial, task_type, model_seed=0):
    print(f"Running Common CatBoost Params with model_seed = {model_seed}")
    params = {
        'iterations': int_param(trial, "iterations", 100, 1000),
        'learning_rate': float_param(trial, "learning_rate", 0.01, 0.3),
        'depth': int_param(trial, "depth", 3, 10),
        'l2_leaf_reg': float_param(trial, "l2_leaf_reg", 1, 10),
        'subsample': float_param(trial, "subsample", 0.5, 1.0),
        'random_strength': float_param(trial, "random_strength", 0, 10),
        'border_count': int_param(trial, "border_count", 32, 254),
        'random_seed': categorical_param(trial, "random_seed", [model_seed]),
        'verbose': categorical_param(trial, "verbose", [0]),
        'task_type': categorical_param(trial, "task_type", ["GPU"]),
        'bootstrap_type': categorical_param(trial, "bootstrap_type", ["Bernoulli"])
    }
    if task_type == "binary_classification":
        params['eval_metric'] = categorical_param(trial, "eval_metric", ['Logloss'])
    elif task_type == "multiclass_classification":
        params['eval_metric'] = categorical_param(trial, "eval_metric", ['MultiClass'])
    return params

def get_common_lgbm_params(trial, task_type, model_seed=0):
    print(f"Running Common LGBM Params with model_seed = {model_seed}")
    params = {
        'n_estimators': int_param(trial, "n_estimators", 100, 1000),
        'learning_rate': float_param(trial, "learning_rate", 0.01, 0.3),
        'max_depth': int_param(trial, "max_depth", 3, 15),
        'num_leaves': int_param(trial, "num_leaves", 20, 150),
        'min_child_samples': int_param(trial, "min_child_samples", 10, 50),
        'subsample': float_param(trial, "subsample", 0.5, 1.0),
        'colsample_bytree': float_param(trial, "colsample_bytree", 0.5, 1.0),
        'reg_alpha': float_param(trial, "reg_alpha", 0, 10),
        'reg_lambda': float_param(trial, "reg_lambda", 0, 10),
        'random_state': categorical_param(trial, "random_state", [model_seed]),
        'verbose': categorical_param(trial, "verbose", [-1]),
        'device_type': categorical_param(trial, "device_type", ["cpu"])
    }
    if task_type == "binary_classification":
        params['objective'] = categorical_param(trial, "objective", ['binary'])
    elif task_type == "multiclass_classification":
        params['objective'] = categorical_param(trial, "objective", ['multiclass'])
    elif task_type == "regression":
        params['objective'] = categorical_param(trial, "objective", ['regression'])
    return params

def get_common_lasso_params(trial, task_type, model_seed=0):
    print(f"Running Common Lasso Params with model_seed = {model_seed}")
    if task_type=="regression":
        params = {
            'alpha': float_param(trial, "alpha", 0.0001, 1000.0),
            'random_state': categorical_param(trial, "random_state", [model_seed])
        }
    else: # classification
        params = { 
            'penalty': categorical_param(trial, "penalty", ['l1',]),
            'solver': categorical_param(trial, "solver", ['liblinear',]),
            'C': float_param(trial, "C", 0.0001, 1000.0),
            'random_state': categorical_param(trial, "random_state", [model_seed]),
        }
    return params

def get_common_ebm_params(trial, task_type, model_seed=0):
    print(f"Running Common EBM Params with model_seed = {model_seed}")
    params = {
        'learning_rate': float_param(trial, "learning_rate", 0.0025, 0.2),
        'max_bins': int_param(trial, "max_bins", 256, 1024),
        'max_interaction_bins': int_param(trial, "max_interaction_bins", 16, 64),
        'interactions': float_param(trial, "interactions", 0.0, 0.95),
        'outer_bags': int_param(trial, "outer_bags", 8, 25),
        'inner_bags': int_param(trial, "inner_bags", 0, 25),
        'max_leaves': categorical_param(trial, "max_leaves", [2, 3]),
        'early_stopping_tolerance': float_param(trial, "early_stopping_tolerance", -0.0001, 0.0001),
        'random_state': categorical_param(trial, "random_state", [model_seed])
    }
    if task_type.find("classification") != -1:
        params['objective'] = categorical_param(trial, "objective", ['log_loss'])
    return params


### DEFAULT SKLEARN HYPERPARAMETERS ###
def get_default_random_forest_params(task_type, model_seed=0):
    print(f"Running default RF Params with model_seed = {model_seed}")
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': model_seed
    }
    if task_type == "regression":
        params['criterion'] = 'squared_error'
    else:
        params['criterion'] = 'gini'
    return params

def get_default_svm_params(task_type, model_seed=0):
    print(f"Running default SVM Params with model_seed = {model_seed}")
    params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale'
    }
    if task_type == "regression":
        #SVR has no random state (https://github.com/scikit-learn/scikit-learn/issues/17391)
        params['epsilon'] = 0.1
    else:
        params.update({'probability': True, 'random_state': model_seed})
    return params

def get_default_xgb_params(task_type, model_seed=0):
    print(f"Running default XGB Params with model_seed = {model_seed}")
    params = {
        'n_estimators': 100,
        'learning_rate': 0.3,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'random_state': model_seed
    }
    if task_type.find("classification") != -1:
        params.update({'use_label_encoder': False, 'eval_metric': 'logloss'})
    return params

def get_default_catboost_params(task_type, model_seed=0):
    print(f"Running default CatBoost Params with model_seed = {model_seed}")
    params = {
        'iterations': 100,
        'learning_rate': 0.1,
        'depth': 6,
        'l2_leaf_reg': 3,
        'subsample': 1.0,
        'random_strength': 1,
        'border_count': 128,
        'random_seed': model_seed,
        'verbose': 0,
        'task_type': 'GPU',
        'bootstrap_type': 'Bernoulli'
    }
    if task_type == "binary_classification":
        params['eval_metric'] = 'Logloss'
    elif task_type == "multiclass_classification":
        params['eval_metric'] = 'MultiClass'
    return params

def get_default_lgbm_params(task_type, model_seed=0):
    print(f"Running default LGBM Params with model_seed = {model_seed}")
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 7,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'random_state': model_seed,
        'verbose': -1,
        'device_type': 'cpu'
    }
    if task_type == "binary_classification":
        params['objective'] = 'binary'
    elif task_type == "multiclass_classification":
        params['objective'] = 'multiclass'
    elif task_type == "regression":
        params['objective'] = 'regression'
    return params

def get_default_lasso_params(task_type, model_seed=0):
    print(f"Running default Lasso Params with model_seed = {model_seed}")
    if task_type=="regression":
        params = {
            'alpha': 1.0,
            'random_state': model_seed,
        }
    else: # classification
        params = { 
            'penalty' : 'l1',
            'solver' : 'liblinear',
            'C': 1.0,
            'random_state': model_seed,
        }
    return params

def get_default_ebm_params(task_type, model_seed=0):
    params = {
        'learning_rate': 0.015,
        'max_bins': 1024,
        'max_interaction_bins': 64,
        'interactions': 0.9,
        'outer_bags': 14,
        'inner_bags': 0,
        'max_leaves': 3,
        'early_stopping_tolerance': 1e-5,
        'random_state': model_seed,
        # No random state in default EBM params -- JAM: what does this mean?
    }
    if task_type.find("classification") != -1:
        params['objective'] = 'log_loss'
    return params





full_model_mapping_dictionary = {
    "RF"        : (get_common_rf_params, get_default_random_forest_params, (RandomForestRegressor, RandomForestClassifier)),
    "SVM"       : (get_common_svm_params, get_default_svm_params, (SVR, SVC)),
    "XGB"       : (get_common_xgb_params, get_default_xgb_params, (XGBRegressor, XGBClassifier)),

    "CatBoost"  : (get_common_catboost_params, get_default_catboost_params, (CatBoostRegressor, CatBoostClassifier)),
    "LightGBM"  : (get_common_lgbm_params, get_default_lgbm_params, (LGBMRegressor, LGBMClassifier)),

    "EBM"       : (get_common_ebm_params, get_default_ebm_params, (ExplainableBoostingRegressor, ExplainableBoostingClassifier)),
    "Lasso"     : (get_common_lasso_params, get_default_lasso_params, (Lasso, LogisticRegression)),
}


def get_model_stuff_helper(model_name, task_type):
    if model_name not in full_model_mapping_dictionary:
        raise ValueError(f"Unknown model: {model_name}")
    get_optuna_params, get_default_params, SklearnModel_tuple = full_model_mapping_dictionary[model_name]
    if len(SklearnModel_tuple)<3:  # binary classification and multiclass classification are the same unless specified
        SklearnModel_tuple = (SklearnModel_tuple[0],SklearnModel_tuple[1],SklearnModel_tuple[1])
    task_list = ["regression", "binary_classification", "multiclass_classification"] #NOTE: better to pull these out from an enum
    SklearnModel = SklearnModel_tuple[task_list.index(task_type)]
    return get_optuna_params, get_default_params, SklearnModel




def optuna_objective(trial, X_train, y_train, X_val, y_val, model_name, task_type, model_seed=0):
    print(f"Running optuna_objective with model_seed = {model_seed}")
    if task_type.find("classification") != -1 and len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train = np.argmax(y_train, axis=1)
        y_val = np.argmax(y_val, axis=1)


    get_optuna_params, __, SklearnModel = get_model_stuff_helper(model_name, task_type)
    params = get_optuna_params(trial, task_type, model_seed)
    model = SklearnModel(**params)


    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    if task_type == "regression":
        score = mean_squared_error(y_val, y_val_pred)
    else:
        try:
            y_proba = model.predict_proba(X_val)
            score = roc_auc_score(y_val, y_proba[:, 1] if len(np.unique(y_val)) == 2 else y_proba, multi_class='ovr')
        except Exception:
            score = 0.0
    trial.set_user_attr("model", model)
    return score


def train_and_get_ml_model_results(model_name, dataset_obj, task_type, exp_folder, use_optuna=True, n_trials=50, seed=0, device=None):
    if model_name=="Random Forest": #alias for 'RF'
        model_name = "RF"

    use_default = not use_optuna #NOTE: make this consistent probably (test() below takes different arugments)
    trnval_seed = seed
    model_seed = seed


    device_details_dict = get_device_details(device)
    cpu_details_dict = get_cpu_details()

    print(f"Running train_and_get_ml_model_results with trnval_seed = {trnval_seed}")
    trnval_perc = 0.70
    dataset_obj.shuffle_and_split_trnval(trnval_seed, trnval_perc)
    X_train, y_train, X_val, y_val = dataset_obj.pull_trnval_data()
    print(f"===== Training {model_name} =====")

    if task_type.find("classification") != -1 and len(y_train.shape) > 1 and y_train.shape[1] > 1: #reformat multiclass onehot vectors as integers
        y_train = np.argmax(y_train, axis=1)
        y_val = np.argmax(y_val, axis=1)

    best_model = None
    best_score = float('inf') if task_type == "regression" else -float('inf')
    best_params = {}



    if use_optuna and not use_default:
        direction = "minimize" if task_type == "regression" else "maximize"
        study = optuna.create_study(direction=direction)

        os.makedirs(exp_folder, exist_ok=True)
        csv_path = os.path.join(exp_folder, f"{model_name}_hyperparameters.csv")
        results_list = []

        def callback(study, trial):
            nonlocal best_model, best_score, best_params
            score = trial.value
            trial_duration = (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else 0.0
            if (task_type == "regression" and score < best_score) or (task_type.find("classification") != -1 and score > best_score):
                best_score = score
                best_params = trial.params
                best_model = trial.user_attrs["model"]

            result_dict = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': model_name,
                'task_type': task_type,
                'trial_number': trial.number,
                'val_score': score,
                'trial_duration': trial_duration,

                "device_details_dict" : device_details_dict,
                "cpu_details_dict" : cpu_details_dict,
            }
            result_dict.update(trial.params)
            results_list.append(result_dict)

        study.optimize(lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, model_name, task_type, model_seed),
                       n_trials=n_trials, callbacks=[callback])

        result_df = pd.DataFrame(results_list)
        result_df.to_csv(csv_path, mode='a' if os.path.exists(csv_path) else 'w', header=not os.path.exists(csv_path), index=False)
        print(f"All trial hyperparameters saved at: {csv_path}")
        print(f"\nBest Hyperparameters for {model_name}: {best_params}")
        print(f"Best Validation {'MSE' if task_type == 'regression' else 'ROC AUC'}: {best_score:.3f}")

    else:
        __, get_default_params, SklearnModel = get_model_stuff_helper(model_name, task_type)
        params = get_default_params(task_type, model_seed)
        model = SklearnModel(**params)

        

        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        y_val_pred = model.predict(X_val)
        if task_type == "regression":
            best_score = mean_squared_error(y_val, y_val_pred)
        else:
            try:
                y_proba = model.predict_proba(X_val)
                best_score = roc_auc_score(y_val, y_proba[:, 1] if len(np.unique(y_val)) == 2 else y_proba, multi_class='ovr')
            except Exception:
                best_score = accuracy_score(y_val, y_val_pred)
        best_model = model
        best_params = params
        print(f"Training time: {end_time - start_time:.2f} seconds")
        print(f"Validation {'MSE' if task_type == 'regression' else 'ROC AUC'}: {best_score:.3f}")

        os.makedirs(exp_folder, exist_ok=True)
        csv_path = os.path.join(exp_folder, f"{model_name}_default_hyperparameters.csv")
        result_dict = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'task_type': task_type,
            'trial_number': 0,
            'val_score': best_score,
            'trial_duration': end_time - start_time,

            "device_details_dict" : device_details_dict,
            "cpu_details_dict" : cpu_details_dict,
        }
        result_dict.update(best_params)
        result_df = pd.DataFrame([result_dict])
        result_df.to_csv(csv_path, mode='a' if os.path.exists(csv_path) else 'w', header=not os.path.exists(csv_path), index=False)
        print(f"Hyperparameters saved at: {csv_path}")

    os.makedirs(exp_folder, exist_ok=True)
    model_path = os.path.join(exp_folder, f"{model_name}_default_best_model.pkl" if use_default else f"{model_name}_best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved at: {model_path}")

    return best_model, model_name, best_params, best_score

def test_ml_model_results(model, model_name, dataset_obj, task_type, exp_folder, use_default=False):
    print('use_default',use_default)
    if use_default == False:
        model = joblib.load(f"{exp_folder}{model_name}_best_model.pkl")
    else:
        model = joblib.load(f"{exp_folder}{model_name}_default_best_model.pkl")
    
    _, _, X_test, y_test = dataset_obj.pull_data()

    if task_type.find("classification") != -1 and len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    y_test_pred = model.predict(X_test)
    
    if y_test_pred.dtype == object or isinstance(y_test_pred[0], str):
        y_test_pred = y_test_pred.astype(float).astype(int)

    print(f"\n===== Testing {model_name} =====")

    result_dict = {
        'model_name': model_name,
        'dataset_str': dataset_obj.get_dataset_id(),
        'task_type': task_type
    }

    if task_type == "regression":
        test_mse = mean_squared_error(y_test, y_test_pred)
        print(f"Test MSE: {test_mse:.3f}")
        result_dict['test_mse'] = test_mse
    elif task_type.find("classification") != -1:
        acc = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')

        print(f"Test Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

        result_dict.update({
            'test_accuracy': acc,
            'test_precision': prec,
            'test_recall': rec,
            'test_f1': f1
        })

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            try:
                if len(np.unique(y_test)) == 2:
                    rocauc = roc_auc_score(y_test, y_proba[:, 1])
                    auprc = average_precision_score(y_test, y_proba[:, 1])
                else:
                    rocauc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                    auprc = average_precision_score(y_test, y_proba, average='weighted')
                print(f"Test ROC AUC: {rocauc:.3f} | AUPRC: {auprc:.3f}")
                result_dict.update({
                    'test_roc_auc': rocauc,
                    'test_auprc': auprc
                })
            except Exception as e:
                print("ROC AUC Error:", e)
                result_dict.update({
                    'test_roc_auc': None,
                    'test_auprc': None
                })

    os.makedirs(exp_folder, exist_ok=True)
    if use_default == False:
        csv_path = os.path.join(exp_folder, f"{model_name}_test_metrics.csv")
    if use_default == True:
        csv_path = os.path.join(exp_folder, f"{model_name}_default_test_metrics.csv")
    result_df = pd.DataFrame([result_dict])
    if os.path.exists(csv_path):
        result_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        result_df.to_csv(csv_path, mode='w', header=True, index=False)
    print(f"Test metrics saved at: {csv_path}")



