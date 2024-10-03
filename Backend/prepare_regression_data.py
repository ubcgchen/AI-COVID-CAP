# Author:       George Chen
# Date:         September 15, 2024
# Email:        gschen@student.ubc.ca
# Description:  This file prepares data for regression analysis.

# Imports
import pandas as pd
from columns import *
from model_params import *
from preprocessing_helpers import *
from utils import *
import os

# Description:  Loads the derivation dataset.
# Inputs:       Nil.
# Outputs:      1) Derivation COVID-19 dataset.
def load_dataset(path):
    return pd.read_csv(path)

# Description:  This function carries out the preprocessing pipeline.
# Inputs:       1) Unprocessed dataframe.
#               2) Model for which you wish to process the dataframe for (vasopressor, ventilation, or RRT).
# Outputs:      1) Pre-processed dataframe.
def preprocess_data(df, file_name):
    preprocessed_data_path = file_name

    if not os.path.exists(preprocessed_data_path):
        df = (df.pipe(process_early_deaths)                         # remove all patients who passed within 72 hours of admission
                .pipe(remove_unknown_outcome)                       # remove all patients with unknown outcome or who were discharged to another facility
                .pipe(process_med_columns)                          # remove medication data if patient was NOT on the medication on the day of admission
                .pipe(remove_non_numeric)                           # remove non-numeric features
                # .pipe(remove_non_covid)                           # remove patients who do not have/were not admitted for COVID
                .pipe(correct_ordinality)                           # correct ordinality of features, as necessary
                .pipe(remove_redundant)                             # remove redundant features (blank columns, features with 0 variance)
                .pipe(lambda df: process_vaso(df, vaso))            # process vasopressor columns
                .pipe(lambda df: remove_empty_target(df, vaso))     # remove patients who do not have data recorded for the target variable
                .pipe(lambda df: remove_empty_target(df, vent))     # remove patients who do not have data recorded for the target variable
                .pipe(lambda df: remove_empty_target(df, rrt))      # remove patients who do not have data recorded for the target variable
                .pipe(impute_labs)                                  # impute lab values first (only troponin + ddimer, as these tend to be highly skewed)
                .pipe(lambda df: impute_rest(df, vaso))             # impute remaining features using knn imputer
            )  
        
        df.to_csv(preprocessed_data_path, index=False)
    
    return pd.read_csv(preprocessed_data_path)

paths = ['Backend/Data/randomized_COVID_derivation_patients.csv',
            'Backend/Data/randomized_COVID_validation_patients.csv',
            'Backend/Data/ARBsI_AI_data_part1-2023-04-25.csv',
            'Backend/Data/ARBsI_AI_data_part2-2023-04-25.csv',
            "Backend/Data/ARBsI_AI_data_CAP_patients-2023-08-31.csv", 
            "Backend/Data/ARBsI_AI_data_CAP_patients-part2-2024-03-14.csv",
            'Backend/Data/randomized_CAP_derivation_patients.csv',
            'Backend/Data/randomized_CAP_validation_patients.csv']

file_names = ['Backend/Preprocessed Datasets/preprocessing_derivation_randomized_regression_COVID.csv',
            'Backend/Preprocessed Datasets/preprocessing_validation_randomized_regression_COVID.csv',
            'Backend/Preprocessed Datasets/preprocessing_derivation_original_regression_COVID.csv',
            'Backend/Preprocessed Datasets/preprocessing_validation_original_regression_COVID.csv',
            'Backend/Preprocessed Datasets/preprocessing_validation1_original_regression_CAP.csv', 
            'Backend/Preprocessed Datasets/preprocessing_validation2_original_regression_CAP.csv',
            'Backend/Preprocessed Datasets/preprocessing_derivation_randomized_regression_CAP.csv',
            'Backend/Preprocessed Datasets/preprocessing_validation_randomized_regression_CAP.csv']

for i in range(4, 8):
    df = load_dataset(paths[i])
    df = preprocess_data(df, file_names[i])