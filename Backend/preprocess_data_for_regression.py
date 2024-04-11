import pandas as pd
from preprocessing_helpers import *
from model_params import *

datasets = ["Backend/Data/ARBsI_AI_data_part2-2023-04-25.csv", 
            "Backend/Data/ARBsI_AI_data_CAP_patients-2023-08-31.csv", 
            "Backend/Data/ARBsI_AI_data_CAP_patients-part2-2024-03-14.csv"]

dataset_name = ["validation", "CAP1", "CAP2"]

index = 2
df = pd.read_csv(datasets[index])
model = rrt

# pre-processing with no data balancing, no normalization
# remove pts who received RRT, ventilation or vaso on day 0 from the regression prediction model
# removed non-numerical features as well as features that were completely empty,
# process med vaso

def remove_empty_target(df):
    targets = ["dis_rrt", "dis_ventilation", "med_vaso"]
    for target in targets:
        df = df.dropna(subset=[target])
    return df

def process_vaso(df):
    # If currently predicting vasopressor use, combine vaso columns. Otherwise, drop vasopressor columns.
    targets = [col for col in df.columns if col.startswith("med_vaso___")]
    conflicting_rows = []

    # Find conflincting data...
    # Get rows where med_vaso___8 and look for the indices of rows where other columns are not all 0.
    # This incidicates conflicting rows (patient cannot both receive and not receive a vasopressor).
    for index, row in df.iterrows():
        if row['med_vaso___8'] == 1 and not all(row[targets[:-1]] == 0):
            conflicting_rows.append(index)
    df = df.drop(conflicting_rows)

    df['med_vaso'] = df[targets[:-1]].any(axis=1).astype(int) # If target is vaso, build the med_vaso column
    
    df = df.drop(columns=targets)                                   # drop all initial target columns
    df = df.loc[:, ~df.columns.str.startswith('dly_day0_vaso___')]  # drop all day_0 med vaso columns

    return df

def remove_day0_intervention(df):
    # Convert date columns to datetime if they are not already
    df['bl_admission_date'] = pd.to_datetime(df['bl_admission_date'])
    df['dis_ventilation_start'] = pd.to_datetime(df['dis_ventilation_start'])
    df['dis_rrt_start'] = pd.to_datetime(df['dis_rrt_start'])

    # Filter out rows where either ventilation or RRT was started on admission day
    df = df[~((df['bl_admission_date'] == df['dis_ventilation_start']) | (df['bl_admission_date'] == df['dis_rrt_start']))]

    # Filter out rows where any vasopressor was started on admission day
    vaso_columns = [col for col in df.columns if col.startswith('med_vaso_start')]
    for col in vaso_columns:
        df = df[~(df['bl_admission_date'] == df[col])]

    return df

def remove_redundant(df):
    threshold = 1.0
    missing_percent = (df.isna().sum() / len(df)).sort_values(ascending=False)
    columns_to_remove = missing_percent[missing_percent >= threshold].index.tolist()
    df = df.drop(columns=columns_to_remove) # Remove features with > threshold % missing values.
    return df

df = (df.pipe(remove_early_deaths)                          # remove all patients who passed within 72 hours of admission
        .pipe(remove_unknown_outcome)                       # remove all patients with unknown outcome or who were discharged to another facility
        .pipe(process_med_columns)                          # remove medication data if patient was NOT on the medication on the day of admission
        .pipe(correct_ordinality)                           # correct ordinality of features, as necessary
        .pipe(process_vaso)                                 # process vasopressor columns
        .pipe(remove_day0_intervention)                     # remove patients who were on any intervention on the day of admission
        .pipe(remove_non_numeric)                           # remove non-numeric features
        .pipe(remove_empty_target)                          # remove patients who do not have data recorded for the target variable
        .pipe(remove_redundant)                             # remove redundant features (blank columns, features with 0 variance)
        .pipe(impute_labs)                                  # impute lab values first (only troponin + ddimer, as these tend to be highly skewed)
        .pipe(lambda df: impute_rest(df, model))            # impute remaining features using knn imputer
    ) 

df.to_csv("Backend/Data/" + dataset_name[index] + "_preprocessed_for_regression.csv")