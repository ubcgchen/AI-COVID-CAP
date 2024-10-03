# Author:       George Chen
# Date:         September 15, 2024
# Email:        gschen@student.ubc.ca
# Description:  This file contains the code to allow for testing of the performance of a specified model with an unseen dataset.

# Imports
import pandas as pd
from utils import *
from preprocessing_helpers import *
from model_params import *
import numpy as np
import pickle

def test_new_dataset(name, dataset_index, dataset_type = ""):

    models = {
        "vaso": vaso, 
        "vent": vent,
        "rrt": rrt
    }

    curr_model = models[name]

    # Available test datasets
    test_datasets = ["ARBsI_AI_data_part2-2023-04-25.csv", 
                     "ARBsI_AI_data_CAP_patients-2023-08-31.csv", 
                     "ARBsI_AI_data_CAP_patients-part2-2024-03-14.csv",
                     "randomized_COVID_validation_patients.csv",
                     "randomized_CAP_validation_patients.csv"]
    dataset_names = ["Part2", "CAP", "CAP2", "COVID_randomized", "CAP_randomized"]

    dataset_name = test_datasets[dataset_index]

    path_to_preprocessed_data = "Backend/Preprocessed Datasets/preprocessing_" + name + ".csv"

    ########################################################################
    # Preprocessing pipeline here.

    df = pd.read_csv('Backend/Data/' + dataset_name)
    # Process dataset so that patient population is the same as the population the model was trained on.
    # Process dataset so that the features are the same as the features the model was trained on.
    # Any feature selection/encoding/engineering strategy uses the model trained on the original dataset.

    if dataset_index == 0 or dataset_index == 3:  # Predicting COVID or CAP? If COVID, remove non-COVID patients
        df = (df.pipe(process_early_deaths)                             # remove all patients who passed within 72 hours of admission
                .pipe(remove_unknown_outcome)                           # remove all patients with unknown outcome or d/c to another facility
                .pipe(process_med_columns)                              # only keep meds that were administered on the first day
                .pipe(remove_non_numeric)                               # remove non-numeric features
                .pipe(remove_non_covid)                                 # remove patients who do not have/were not admitted for COVID
                .pipe(combine_features)                                 # condense features - reduce granularity
                .pipe(correct_ordinality)                               # correct ordinality of features, as necessary
                .pipe(lambda df: process_vaso(df, curr_model))          # process vasopressor columns
                .pipe(lambda df: remove_empty_target(df, curr_model))   # remove patients who do not have a target value)
                .pipe(lambda df: consolidate_cols(df, curr_model, dataset_type))      # consolidate all columns
            )  
    else:   # Else if CAP, do not remove non-COVID patients.
        df = (df.pipe(process_early_deaths)                             # remove all patients who passed within 72 hours of admission
                .pipe(remove_unknown_outcome)                           # remove all patients with unknown outcome or d/c to another facility
                .pipe(process_med_columns)                              # only keep meds that were administered on the first day
                .pipe(remove_non_numeric)                               # remove non-numeric features
                .pipe(combine_features)                                 # condense features - reduce granularity
                .pipe(correct_ordinality)                               # correct ordinality of features, as necessary
                .pipe(lambda df: process_vaso(df, curr_model))          # process vasopressor columns
                .pipe(lambda df: remove_empty_target(df, curr_model))   # remove patients who do not have a target value)
                .pipe(lambda df: consolidate_cols(df, curr_model, dataset_type))      # consolidate all columns
            )  
        
    # Note: the difference when testing for CAP is that we do not filter out patients who do not have a primary diagnosis of COVID

    ########################################################################

    # Load models trained during preprocessing of derivation dataset
    with open(curr_model["scaler"], 'rb') as file:
        scaler = pickle.load(file)
    with open(curr_model["imputer"], 'rb') as file:
        imputer = pickle.load(file)

    # Transform new dataset according to models trained during preprocessing of derivation dataset.
    df = pd.DataFrame(scaler.transform(df), columns = df.columns)
    df = pd.DataFrame(imputer.transform(df), columns = df.columns)

    # Get the pre-preprocessed derivation dataset for the current model (vaso, vent, or rrt).
    path_to_preprocessed_data = "Backend/Preprocessed Datasets/preprocessing_" + curr_model["name"] + dataset_type + ".csv"
    file_path = path_to_preprocessed_data
    df_original = pd.read_csv(file_path)
    common_columns = [item for item in df.columns if item in df_original.columns]

    # Remove all the columns in the test dataset that are not in the preprocessed dataset.
    df = df[common_columns]

    with open(f'rfc_{name}{dataset_type}.pkl', 'rb') as file:
        model = pickle.load(file)

    X = df.drop(curr_model["target"], axis=1)
    X.columns = X.columns.astype(str)
    y = df[curr_model["target"]]

    y_pred = model.predict(X)
    y_pred = pd.Series(y_pred, index=y.index)
    probabilities = model.predict_proba(X)
    probability_positive = probabilities[:, 1]
    magnitudes = np.maximum(probabilities[:, 0], probabilities[:, 1])
    magnitudes = pd.Series(magnitudes, index=y.index)
    probabilities = pd.DataFrame(probabilities, index=y.index)

    incorrect_indices = y.index[y != y_pred].tolist()
    incorrect_examples = df.loc[incorrect_indices]
    incorrect_probabilities = probabilities.loc[incorrect_indices]

    false_negative_probabilities = pd.Series()
    false_positive_probabilities = pd.Series()

    # Identify false negatives and false positives
    for index, row in incorrect_probabilities.iterrows():
        if row[0] > row[1]:
            false_negative_probabilities.loc[index] = row[0]
        elif row[1] > row[0]:
            false_positive_probabilities.loc[index] = row[1]

    correct_indices = X.index.difference(incorrect_indices)
    correct_examples = df.loc[correct_indices]
    correct_probabilities = magnitudes.loc[correct_indices]

    # false_negative_probabilities.to_csv('Backend/Misclassification Analysis/false_negative_probabilities_' + name + '.csv', index=True)
    # false_positive_probabilities.to_csv('Backend/Misclassification Analysis/false_positive_probabilities_' + name + '.csv', index=True)
    # correct_probabilities.to_csv('Backend/Misclassification Analysis/correct_probabilities_' + name + '.csv', index=True)
    # incorrect_examples.to_csv('Backend/Misclassification Analysis/incorrectly_classified_examples_' + name + '.csv', index=True)
    # correct_examples.to_csv('Backend/Misclassification Analysis/correctly_classified_examples_' + name + '.csv', index=True)

    calc_and_write_metrics(y_pred, y, probability_positive, name, f'validation data/{dataset_names[dataset_index]}') # calculate and write metrics to file
    plot_roc_curve(y, probability_positive, name, curr_model["intervention"], f'validation data/{dataset_names[dataset_index]}')


################################

names = ["rrt", "vent", "vaso"]
indices = [3]

for index in indices:
    for name in names:
        test_new_dataset(name, index, "_randomized_COVID")

################################