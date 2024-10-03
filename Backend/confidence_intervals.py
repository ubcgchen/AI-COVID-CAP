# Author:       George Chen
# Date:         September 15, 2024
# Email:        gschen@student.ubc.ca
# Description:  This file contains a reformatted version of the code to train machine learning models to make predictions about vasopressors, ventilators,
#               and RRT use using the derivation COVID dataset. It trains 100 different models.

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from columns import *
from model_params import *
from preprocessing_helpers import *
from utils import *
import os


# Description:  Loads the derivation dataset.
# Inputs:       Nil.
# Outputs:      1) Derivation COVID-19 dataset.
def load_dataset():
    return pd.read_csv('Backend/Data/ARBsI_AI_data_part1-2023-04-25.csv')

# Description:  This function carries out the preprocessing pipeline.
# Inputs:       1) Unprocessed dataframe.
#               2) Model for which you wish to process the dataframe for (vasopressor, ventilation, or RRT).
# Outputs:      1) Pre-processed dataframe.
def preprocess_data(df, model):
    preprocessed_data_path = f'Backend/Preprocessed Datasets/preprocessing_{model["name"]}.csv'

    if not os.path.exists(preprocessed_data_path):
        df = (df.pipe(process_early_deaths)                         # remove all patients who passed within 72 hours of admission
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

# Description:  Generates new pre-processed dataset following PCA feature engineering.
# Inputs:       1) Features from training dataset
#               2) Features from test dataset
#               3) Target from training dataset
#               4) Target from test dataset
#               5) Model to be trained
# Outputs:      1) Updated pre-processed dataset.
def rewrite_preprocessed_dataset(X_train, X_test, y_train, y_test, model):
    stacked_X = pd.concat([X_train, X_test], axis=0).sort_index()
    stacked_y = pd.concat([y_train, y_test], axis=0).sort_index()
    df = pd.concat([stacked_X, stacked_y], axis=1)

    df.to_csv(f'Backend/Preprocessed Datasets/preprocessing_{model["name"]}.csv', index=False)

# Description:  Trains the model.
# Inputs:       1) The pre-processed dataframe
#               2) Model that you wish to train (vasopressor, ventilator, or RRT)
#               3) Classifier that you wish to use to train (RFC, SVM, etc...)
# Outputs:      1) The trained model
def train_model(df, model, classifier):

    # Train-test split. Test size is 20%, train size is 80%.
    X = df.drop(model["target"], axis=1)
    X.columns = X.columns.astype(str)
    y = df[model["target"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifiers = {
        'vaso': RandomForestClassifier(
                    n_estimators=200,          # Number of trees
                    max_depth=30,              # Maximum depth of the trees
                    min_samples_split=2,       # Minimum number of samples required to split a node
                    min_samples_leaf=1,        # Minimum number of samples required in a leaf node
                    bootstrap=False,            # Whether bootstrap samples are used
                    max_features='sqrt',
                ),
        'vent': RandomForestClassifier(
                    n_estimators=200,          # Number of trees
                    max_depth=10,              # Maximum depth of the trees
                    min_samples_split=5,       # Minimum number of samples required to split a node
                    min_samples_leaf=2,        # Minimum number of samples required in a leaf node
                    bootstrap=False,            # Whether bootstrap samples are used
                    max_features='sqrt',
                ),
        'rrt': RandomForestClassifier(
                    n_estimators=200,          # Number of trees
                    max_depth=10,              # Maximum depth of the trees
                    min_samples_split=2,       # Minimum number of samples required to split a node
                    min_samples_leaf=1,        # Minimum number of samples required in a leaf node
                    bootstrap=False,            # Whether bootstrap samples are used
                    max_features='sqrt',
                )
    }

    rfc = classifiers[model["name"]]
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test) # make predictions
    probabilities = rfc.predict_proba(X_test) # get the model's confidence in each prediction. To be used in misclassification analysis.
    probability_positive = probabilities[:, 1]

    # Calculate metrics.
    calc_and_write_metrics_list(y_pred, y_test, probability_positive, model["name"], classifier["name"])

    return

# Description:  Coordinates the training pipeline.
# Inputs:       1) Model that you wish to train (vasopressor, ventilator, or RRT)
#               2) Classifier that you wish to use to train (RFC, SVM, etc...)
# Outputs:      Nil.
def coordinate_pipeline(model, classifier):
    df = load_dataset()
    df_preprocessed = preprocess_data(df, model)
    train_model(df_preprocessed, model, classifier)

# Run the pipeline 100 times

for i in range(100):
    coordinate_pipeline(vent, rfc)

