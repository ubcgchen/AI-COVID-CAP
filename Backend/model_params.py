from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

vaso_params = {
    "bootstrap": [True],
    "max_depth": [None],
    "max_features": ['log2'],
    "min_samples_leaf": [1],
    "min_samples_split": [2],
    "n_estimators": [200]
}

vaso = {
    "name": "vaso",
    "intervention": "vasopressor",
    "target": "med_vaso",
    "scaler": "scaler_vaso.pkl",
    "imputer": "knnimputer_vaso.pkl",
    "PCA": "PCA_vaso.pkl",
    "PCA_scaler": "PCA_scaler_vaso.pkl",
    "model": "rfc_vaso.pkl",
    "n_neighbors": 12,
    "n_components": 0.6,
    "sampling_strategy_adasyn": 0.7,
    "sampling_strategy_smote": 0.9,
    "params": vaso_params
}

vent_params = {
    "bootstrap": [True],
    "max_depth": [None],
    "max_features": ['log2'],
    "min_samples_leaf": [1],
    "min_samples_split": [2],
    "n_estimators": [200]
}

vent = {
    "name": "vent",
    "intervention": "ventilator",
    "target": "dis_ventilation",
    "scaler": "scaler_vent.pkl",
    "imputer": "knnimputer_vent.pkl",
    "PCA": "PCA_vent.pkl",
    "PCA_scaler": "PCA_scaler_vent.pkl",
    "model": "rfc_vent.pkl",
    "n_neighbors": 10,
    "n_components": 0.75,
    "sampling_strategy_adasyn": 0.75,
    "sampling_strategy_smote": 0.95,
    "params": vent_params,
}

rrt_params = {
    "bootstrap": [True],
    "max_depth": [None],
    "max_features": ['log2'],
    "min_samples_leaf": [1],
    "min_samples_split": [5],
    "n_estimators": [500]
}

rrt = {
    "name": "rrt",
    "intervention": "renal replacement",
    "target": "dis_rrt",
    "scaler": "scaler_rrt.pkl",
    "imputer": "knnimputer_rrt.pkl",
    "PCA": "PCA_rrt.pkl",
    "PCA_scaler": "PCA_scaler_rrt.pkl",
    "model": "rfc_rrt.pkl",
    "n_neighbors": 7,
    "n_components": 0.6,
    "sampling_strategy_adasyn": 0.12,
    "sampling_strategy_smote": 0.17,
    "params": rrt_params
}

rfc_param_grid = {
    'n_estimators': [500], 
    'max_depth': [None, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_leaf': [1],
    'max_features': ['log2'],
    'bootstrap': [True]
}

xgb_param_grid = {
    'n_estimators': [150, 300, 400],
    'max_depth': [None, 10],
    'learning_rate': [0.2, 0.3, 0.4],
    'subsample': [0.7, 0.85, 1.0],
    'reg_alpha': [0, 0.01, 1],
}

lr_param_grid = {
    'solver': ['sag', 'saga', 'lbfgs', 'newton-cholesky'],
    'penalty': ['l2', 'none'],
    'C': [0.001, 0.1, 1.0],
    'max_iter': [100, 1000],
}

svm_param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'tol': [1e-3, 1e-4, 1e-5],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4],
}

mlp_param_grid = {
    'hidden_layer_sizes': [(10,6,10), (7, 12, 7), (13, 13)],
    'max_iter': [2500],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}


rfc = {
    "classifier": RandomForestClassifier(random_state=42),
    "name": "rfc",
    "param_grid": rfc_param_grid 
}

mlp = {
    "classifier": MLPClassifier(),
    "name": "mlp",
    "param_grid": mlp_param_grid
}

xgboost = {
    "classifier": XGBClassifier(),
    "name": "xgb",
    "param_grid": xgb_param_grid
}

svm = {
    "classifier": SVC(),
    "name": "svm",
    "param_grid": svm_param_grid
}

logreg = {
    "classifier": LogisticRegression(),
    "name": "lr",
    "param_grid": lr_param_grid
}

