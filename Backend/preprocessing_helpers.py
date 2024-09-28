# Author:       George Chen
# Date:         September 15, 2024
# Email:        gschen@student.ubc.ca
# Description:  This file contains the code to conduct the preprocessing pipeline.

# Imports
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import pickle
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from statistics import mode
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from statistics import mean

from columns import *
from model_params import *
from utils import *

random_seed = 42 # randomly chosen seed for reproducibility.

# Change the outcome of all patients who died without ICU admission as experiencing the outcome - 
# assume these patients would have benefitted from such intervention.
def process_early_deaths(df):
    mask = (
        (df['dis_outcome'] == 0) &  # patient passed away and...
        (df['bl_admission_icu'] == 0)  # they were not admitted to ICU
    )

    # Update the specified columns based on the mask.
    # Classify patients who died without ICU admission as experiencing every outcome.
    # RRT = 1, vent = 1, vaso = 1 (we are not interested in individual pressors at the moment, so we set an arbitrary one to 1 and set med_vaso___8, which is "no vasopressors received" to 0).
    df.loc[mask, ['dis_rrt', 'dis_ventilation', 'med_vaso___0']] = 1
    df.loc[mask, 'med_vaso___8'] = 0

    df.to_csv('output_file.csv', index=False)

    return df

# Drop all patients with unknown outcome or who were discharged to another facility.
def remove_unknown_outcome(df):
    mask = ~df['dis_outcome'].isin([2, 3, 4])
    df = df[mask]

    return df

# Remove data for medications that were not being administered on the day of admission
def process_med_columns(df):

    # Convert date columns to datetime objects
    df.loc[:, 'bl_admission_date'] = pd.to_datetime(df['bl_admission_date'])           # Date of admission
    df.loc[:, 'med_abx_start'] = pd.to_datetime(df['med_abx_start'])                   # Date that drugs were started  
    df.loc[:, 'med_antifungal_start'] = pd.to_datetime(df['med_antifungal_start'])
    df.loc[:, 'med_steroid_start'] = pd.to_datetime(df['med_steroid_start'])

    df.loc[:, 'med_abx_end'] = pd.to_datetime(df['med_abx_end'])                       # Date that drugs were ended
    df.loc[:, 'med_antifungal_end'] = pd.to_datetime(df['med_antifungal_end'])
    df.loc[:, 'med_steroid_end'] = pd.to_datetime(df['med_steroid_end'])

    # Only keep the data if the admission date falls between drug start and end date (inclusive).
    # Update antibiotics column
    df.loc[:, 'med_abx'] = ((df['med_abx_start'] <= df['bl_admission_date']) & 
                            (df['med_abx_end'] >= df['bl_admission_date'])).astype(int)

    # Update antifungal column
    df.loc[:, 'med_antifungal'] = ((df['med_antifungal_start'] <= df['bl_admission_date']) & 
                                   (df['med_antifungal_end'] >= df['bl_admission_date'])).astype(int)

    # Update steroid column
    df.loc[:, 'med_steroid'] = ((df['med_steroid_start'] <= df['bl_admission_date']) & 
                                (df['med_steroid_end'] >= df['bl_admission_date'])).astype(int)

    return df

# Remove non-numeric features
def remove_non_numeric(df):
    numeric_columns = df.select_dtypes(include=[int, float])
    return df[numeric_columns.columns]

# Drop other non-medical features
def remove_non_medical(df):
    df = df.drop(drop_non_medical, axis = 1)
    return df

# Drop non-day 0 features
def remove_non_day_0(df, model):
    # Get all columns from day 1 - day 14 (any non-day-0 feature).
    drop_cols_days = df.columns[(df.columns.get_loc("org_day1_status")):
                        (df.columns.get_loc("organ_dysfunction_day_14_complet")+1)]
    df.drop(columns=drop_cols_days, inplace=True) # remove all columns from "day_1" to "organ_dysfunction_day_14_complet". This removes data collected during the stay not on day 0.

    # drop all "discharge" features (ie d/c columns that are NOT the target). Also remove "ICU admission".
    columns_to_remove = [col for col in df.columns if col.startswith("dis")     # Drop all "discharge" information ...
                         and col != model["target"]]                            # but not if it is the target.
    
    df = df.drop(columns=(columns_to_remove + ["bl_admission_icu"])) # ICU admission would make prediction trivial

    return df

# If COVID is not the primary reason for admission... (bl_admission_reason)
# Or if COVID negative/indeterminate/not applicable... (bl_pathogen___4, bl_pathogen___5, bl_pathogen___6)
def remove_non_covid(df):
    not_covid = ((df['bl_admission_reason'] == 1) & # primary reason for admission is COVID and ...
                 (df['bl_pathogen___4'] == 0) & # pt does not have a definitive negative COVID test and ...
                 (df['bl_pathogen___5'] == 0) & # the pathogen is not indeterminate and ...
                 (df['bl_pathogen___6'] == 0))
    df = df[not_covid] # the pathogen is not "not applicable"
    
    # Then remove the COVID status columns because they are otherwise irrelevant.
    df = df.drop(columns = drop_covidstatus)
    return df

# Exclude blank features and features with no variance
def remove_redundant(df):
    threshold = 0.75
    missing_percent = (df.isna().sum() / len(df)).sort_values(ascending=False)
    columns_to_remove = missing_percent[missing_percent > threshold].index.tolist()
    df = df.drop(columns=columns_to_remove) # Remove features with > threshold % missing values.

    vt = VarianceThreshold() # remove features with no variance
    _ = vt.fit(df.fillna(df.mean()))
    mask = vt.get_support()
    df = df.loc[:, mask]

    return df

# Combine certain columns to reduce granularity
def combine_features(df):
    # Get all the columns describing patient comorbidities
    comorbid_columns = [col for col in df.columns if col.startswith("co_")  # Get all comorbidity columns...
                        and col.find("___") != -1]                          # that are "split" into ___0, ___1, and ___2.

    # Iterate over the comorbid columns and combine the columns. 
    # The unprocessed data is split 3 columns: no, yes, and n/a. Turn this into one column where no = 0, yes = 1, n/a = blank
    for column in comorbid_columns:
        # Extract the prefix and suffix from the column name (eg co_cardiac___2 -> prefix = co_cardiac, suffix = 2)
        prefix, suffix = column.split('___')
        
        # Check if the suffix is '2' and the value is 1. If so, there is no comorbidity information available.
        if suffix == '2':
            df.loc[df[column] == 1, prefix + '___1'] = np.nan # set the corresponding cell to be blank
        
        if column.endswith(('0', '2')): # drop the "no" and "no data" columns
            df.drop(columns=column, inplace=True)

    # Combine ACE inhibitor/ARB columns (eg. ramipril, captopril, ... become one general "ACE inhibitor" column)
    ace_hospital = [f'med_ace_post___{i}' for i in range(1, 12)]
    df['med_ace_post'] = df[ace_hospital].any(axis=1).astype(int)
    df = df.drop(columns=ace_hospital + ['med_ace_post___12'])

    ace = [f'med_ace___{i}' for i in range(1, 12)]
    df['med_ace'] = df[ace].any(axis=1).astype(int)
    df = df.drop(columns=ace + ['med_ace___12'])

    arb = [f'med_arbs___{i}' for i in range(0, 8)]
    df['med_arbs'] = df[arb].any(axis=1).astype(int)
    df = df.drop(columns=arb + ['med_arbs___8'])

    arb_hospital = [f'med_arbs_after___{i}' for i in range(0, 8)]
    df['med_arbs_after'] = df[arb_hospital].any(axis=1).astype(int)
    df = df.drop(columns=arb_hospital + ['med_arbs_after___8'])

    # Combine antiviral columns
    antiviral_columns = [f'med_antiviral___{i}' for i in range(6)]
    df['med_antiviral'] = df[antiviral_columns].any(axis=1).astype(int)
    df = df.drop(columns=antiviral_columns + ['med_antiviral___6'])
    
    return df

# Adjust ordinality of the smoking column to match risk level
def correct_ordinality(df):
    # Change labelling in the 'co_smoking' column. Makes more sense for risk stratification.
    # Before: co_smoking = 1 is current smoker and co_smoking = 2 is prior.
    # After: co_smoking = 2 is current smoker and co_smoking = 1 is current. Assuming current smoker risk generally > previous smoker risk
    df['co_smoking'] = df['co_smoking'].replace({1: 2, 2: 1})

    return df

# Preprocess the vasopressor columns, depending on the current model
def process_vaso(df, model):
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

    # Only keep vasopressor data if we are trying to predict vasopressors
    if model["target"] == "med_vaso":
        df['med_vaso'] = df[targets[:-1]].any(axis=1).astype(int) # If target is vaso, build the med_vaso column
    
    df = df.drop(columns=targets)                                   # drop all initial target columns
    df = df.loc[:, ~df.columns.str.startswith('dly_day0_vaso___')]  # drop all day_0 med vaso columns

    return df

# Remove patients where the target value is empty.
def remove_empty_target(df, model):
    df = df.dropna(subset=[model["target"]])
    return df

# Impute missing lab values by using the average of the normal lab values.
def impute_labs(df):
    # Assume that, if not measured, troponin and d-dimer is normal (troponin + d-dimer is probably only measured if there is a clinical suspicion it might be high)
    # Replace with the average value of all normal troponins
    labs = {
        "troponin": 0.04,
        "ddimer": 0.5
    }

    for key, lab_value in labs.items():
        column_name = f"bl_lab_{key}"
        average = df[df[column_name] < lab_value][column_name].mean()

        # Replace NaN with the calculated average
        df[column_name].fillna(average, inplace=True)
        df[column_name].fillna(lab_value, inplace=True)

    return df

# Normalize data
def normalize_data(df, model):
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

    with open('scaler_' + model['name'] + '.pkl', 'wb') as file:
        pickle.dump(scaler, file) # save the scaler
    
    return df

# Impute the remaining features with KNN imputer
def impute_rest(df, model):
    imputer = KNNImputer(n_neighbors=model["n_neighbors"])
    df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

    with open('knnimputer_' + model['name'] + '.pkl', 'wb') as file:
        pickle.dump(imputer, file) # save the scaler

    return df

# Balance data using the ADASYN and SMOTE methods.
def balance_data_custom(df, model):
    # Separate features and target variable
    X = df.drop(model['target'], axis=1)
    y = df[model['target']]
    proportions = y.value_counts(normalize=True)

    proportion_0 = proportions.get(0, 0) * 100  # Proportion of 0 in percentage
    proportion_1 = proportions.get(1, 0) * 100  # Proportion of 1 in percentage

    minority = min(proportion_0, proportion_1)
    majority = max(proportion_0, proportion_1)

    # At most, the number of examples imputed each step does not exceed the number of known samples.
    # If less samples are needed, split the imputation evenly between ADASYN and SMOTE.
    proportion_ADASYN = min(minority/majority * 2, mean((1, minority/majority))) #min(minority/majority * 2, round_up_to_multiple_of_10(mean((1, minority/majority))))
    proportion_SMOTE = min(minority/majority * 3, 1)
    additor = 0

    # Create a pipeline with SMOTE for oversampling and RandomUnderSampler for undersampling steps
    while 1:
        try:
            pipeline = Pipeline([
                ('adasyn', ADASYN(sampling_strategy=proportion_ADASYN + additor, random_state=random_seed)),
                ('over', SMOTE(sampling_strategy=proportion_SMOTE, random_state=random_seed)),
                ('under', RandomUnderSampler(sampling_strategy='auto', random_state=random_seed))
            ])

            # Fit and transform the data
            X, y = pipeline.fit_resample(X, y)
            break
        except ValueError as e:
            additor += 0.01

    # Create a new DataFrame with the resampled data
    df = pd.concat([X, y], axis=1)

    return df

# Feature selection with boruta algorithm
def boruta_select(df, model):
    X = df.drop(model["target"], axis=1)
    y = df[model["target"]]

    forest = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=5, random_state=random_seed)
    forest.fit(X, y)
    boruta = BorutaPy(
        estimator = forest, 
        n_estimators = 'auto',
        random_state=random_seed)

    # fit Boruta
    boruta.fit(np.array(X), np.array(y))

    X = X.loc[:, boruta.support_.tolist()]
    return pd.concat([X, y], axis=1)

# This function does a few things:
# 1) Remove all columns in the test dataset that are not in the preprocessed derivation dataset.
# 2) Fills in lab values for troponin and ddimer.
def consolidate_cols(df, model):
    # load the columns within the preprocessed dataset of interest.
    with open(model["scaler"], 'rb') as file:
        scaler = pickle.load(file)
    common_columns = scaler.feature_names_in_

    # Remove all the columns in the test dataset that are not in the preprocessed dataset.
    df = df[common_columns]

    # Impute labs using the preprocessed dataset.
    labs = {
        "bl_lab_troponin": 0,
        "bl_lab_ddimer": 0
    }

    path_to_preprocessed_data = "Backend/Preprocessed Datasets/preprocessing_" + model["name"] + ".csv"
    file_path = path_to_preprocessed_data
    df_original = pd.read_csv(file_path)

    # Iterate over the dictionary and calculate the mode for each column
    for key in labs:
        if key in df_original.columns:
            mode_value = mode(df_original[key])
            labs[key] = mode_value

    for key, value in labs.items():
        if key in df.columns:
            df[key] = df[key].fillna(value)

    return df