import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from columns import *
from model_params import *
from preprocessing_helpers import *
from utils import *
import os

# Fit curve to feature importances, remove features past inflection point.
# re-train with SVM

random_seed = 42

def load_dataset():
    return pd.read_csv('Backend/Data/ARBsI_AI_data_part1-2023-04-25.csv')

def preprocess_data(df, model):
    preprocessed_data_path = "Backend/Preprocessed Datasets/preprocessing_" + model["name"] + ".csv"

    if os.path.exists(preprocessed_data_path):
        df = (df.pipe(remove_early_deaths)                          # remove all patients who passed within 72 hours of admission
                .pipe(remove_unknown_outcome)                       # remove all patients with unknown outcome or who were discharged to another facility
                .pipe(process_med_columns)                          # remove medication data if patient was NOT on the medication on the day of admission
                .pipe(remove_non_numeric)                           # remove non-numeric features
                .pipe(remove_non_medical)                           # remove non-medical features
                .pipe(lambda df: remove_non_day_0(df, model))       # remove features not obtained on the day of admission except the target
                .pipe(remove_non_covid)                             # remove patients who do not have/were not admitted for COVID
                .pipe(combine_features)                             # condense features - reduce granularity
                .pipe(correct_ordinality)                           # correct ordinality of features, as necessary
                .pipe(remove_redundant)                             # remove redundant features (blank columns, features with 0 variance)
                .pipe(lambda df: process_vaso(df, model))           # process vasopressor columns
                .pipe(lambda df: remove_empty_target(df, model))    # remove patients who do not have data recorded for the target variable
                .pipe(impute_labs)                                  # impute lab values first (only troponin + ddimer, as these tend to be highly skewed)
                .pipe(lambda df: normalize_data(df, model))         # normalize data
                .pipe(lambda df: impute_rest(df, model))            # impute remaining features using knn imputer
                .pipe(lambda df: balance_data_custom(df, model))    # balance data with SMOTE
                .pipe(lambda df: boruta_select(df, model))          # further automatic feature selection with the boruta algorithm
            )  
        
        df.to_csv(preprocessed_data_path, index=False)
    
    return pd.read_csv(preprocessed_data_path)

def rewrite_preprocessed_dataset(X_train, X_test, y_train, y_test, model):
    stacked_X = pd.concat([X_train, X_test], axis=0).sort_index()
    stacked_y = pd.concat([y_train, y_test], axis=0).sort_index()
    df = pd.concat([stacked_X, stacked_y], axis=1)

    df.to_csv("Backend/Preprocessed Datasets/preprocessing_" + model["name"] + ".csv", index=False)

def train_model(df, model, classifier):
    X = df.drop(model["target"], axis=1)
    X.columns = X.columns.astype(str)
    y = df[model["target"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    model_path = f'{classifier["name"] + "_" + model["name"]}.pkl'

    if os.path.exists(model_path):
        # PCA should be trained on the training set only, and not the test set, to prevent data leakage.
        # As such, we do this step after train-test splitting.
        X_train = engineer_features(X_train, model)

        # Transform the test set based on the PCA trained on the training set.
        X_test = pca_transform(X_test, model)
        rewrite_preprocessed_dataset(X_train, X_test, y_train, y_test, model)

        grid_search = GridSearchCV(estimator=classifier["classifier"], 
                                param_grid=model["params"], 
                                cv=StratifiedKFold(n_splits=10), 
                                scoring='accuracy',
                                verbose = 2)
        grid_search.fit(X_train, y_train)

        # Get the best model and its parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        filename = f'{classifier["name"] + "_" + model["name"]}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(best_model, file)

        # Write the parameters of the best model to file
        df_params = pd.DataFrame.from_dict(best_params, orient='index', columns=['Value'])
        df_params.to_excel('Backend/Model Metrics/best_params_' + classifier["name"] + '_' + model["name"] + '.xlsx', index_label='Parameter')
    
        # Retrieve feature importances
        importance_scores = best_model.feature_importances_
        feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': importance_scores})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
        feature_importances.to_excel("Backend/Model Metrics/importances_" + classifier["name"] + "_" + model["name"] + ".xlsx", index=False)

    with open(model_path, 'rb') as file:
        best_model = pickle.load(file) # save the best model to file

    y_pred = best_model.predict(X_test) # make predictions
    probabilities = best_model.predict_proba(X_test) # get the model's confidence in each prediction. To be used in misclassification analysis.
    probability_positive = probabilities[:, 1]

    calc_and_write_metrics(y_pred, y_test, probability_positive, classifier["name"] + "_" + model["name"])
    plot_roc_curve(y_test, probability_positive, classifier["name"] + "_" + model["name"], model["intervention"])

    return

def coordinate_pipeline(model, classifier):
    df = load_dataset()
    df_preprocessed = preprocess_data(df, model)
    train_model(df_preprocessed, model, classifier)

coordinate_pipeline(rrt, rfc)

# for _ in range(1):
#     coordinate_pipeline(vaso, rfc)
