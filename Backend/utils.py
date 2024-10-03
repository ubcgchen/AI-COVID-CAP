# Author:       George Chen
# Date:         September 15, 2024
# Email:        gschen@student.ubc.ca
# Description:  This file contains code of frequently used utility functions.

# Imports
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

import os

random_seed = 42

def calc_metrics(y_pred, y_test, y_prob_positive):
     # Calculate the metrics for the model
     accuracy = accuracy_score(y_test, y_pred)
     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
     sensitivity = tp / (tp + fn)
     specificity = tn / (tn + fp)
     ppv = tp / (tp + fp)
     npv = tn / (tn + fn)
     fpr, tpr, _ = roc_curve(y_test, y_prob_positive)
     roc_auc = auc(fpr, tpr)   

     return [accuracy, sensitivity, specificity, ppv, npv, roc_auc]
file_path = 'Backend/Data/patients_data.xlsx'

def calc_and_write_metrics_list(y_pred, y_test, y_prob_positive, model, classifier):
     values = calc_metrics(y_pred, y_test, y_prob_positive)
     new_data_df = pd.DataFrame([values])
     file_path = f'Backend/Model Metrics/{classifier}/{model}/metrics_repeated.xlsx'

     # Check if the file exists
     if os.path.exists(file_path):
          # If the file exists, read the existing data
          existing_data = pd.read_excel(file_path)
          
          # Append the new data to the existing data
          combined_data = pd.concat([existing_data, new_data_df], ignore_index=True)
     
          # Write the combined data back to the same Excel file
          with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
               combined_data.to_excel(writer, index=False)
     else:
          # If the file does not exist, create a new Excel file with the data
          with pd.ExcelWriter(file_path, mode='w', engine='openpyxl') as writer:
               new_data_df.to_excel(writer, index=False)


# Description:  This function calculates model metrics.
# Inputs:       1) Predictions
#               2) Ground truth
#               3) Probability scores for predictions
#               4) Model these metrics are claculated for (vasopressor, ventilator, or RRT)
# Outputs:      ROC curves for each dataset.
def calc_and_write_metrics(y_pred, y_test, y_prob_positive, model, classifier, dataset_type = ''):
     # Calculate the metrics for the model

     class_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
     values = calc_metrics(y_pred, y_test, y_prob_positive)

     # Format data and write details to excel
     data = {
     'Metrics': ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', "AUC ROC"],
     'Values': values
     }
     metrics = pd.DataFrame(data)

     # Write metrics to Excel
     os.makedirs(f'Backend/Model Metrics/{classifier}/{model}', exist_ok=True)
     metrics.to_excel(f'Backend/Model Metrics/{classifier}/{model}/metrics{dataset_type}.xlsx', index=False)
     class_report.to_excel(f'Backend/Model Metrics/{classifier}/{model}/classification_report{dataset_type}.xlsx', sheet_name='Classification Report')

# Description:  This function calculates metrics for roc curves.
# Inputs:       1) Predictions
#               2) Probability scores for predictions
#               3) Name of the endpoint this ROC curve is for.
#               4) Name of the intervention this ROC curve is for.
#               5) Name of the classifier this ROC curve is for.
# Outputs:      ROC curves for validation dataset for inputted endpoint.
def plot_roc_curve(y_test, y_prob_positive, model, intervention, classifier_name):
     fpr, tpr, _ = roc_curve(y_test, y_prob_positive)
     roc_auc = auc(fpr, tpr)

     data = {'intervention': intervention, 'fpr': fpr, 'tpr': tpr}
     df_to_write = pd.DataFrame(data)
     csv_path = f'Backend/Model Metrics/{classifier_name}/{model}/fpr_tpr.csv'
     df_to_write.to_csv(csv_path, index=False)

     # Plot the ROC curve
     plt.figure(figsize=(8, 8))
     plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'AUC = {roc_auc:.2f}')
     plt.plot([0, 1], [0, 1], color='navy', lw=0.75, linestyle='--')
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.legend(loc='lower right')

     # Save the plot to a file (e.g., PNG)
     plt.savefig(f'Backend/Model Metrics/{classifier_name}/{model}/roc_curve_.png')

def round_up_to_multiple_of_10(x):
    return round(math.ceil(x * 10) / 10, 2)

# Description:  This function combines two datasets into a single dataset.
# Inputs:       1) df1: dataset 1
#               2) df2: dataset 2
#               3) dataset_type: either COVID or CAP
# Outputs:      Combined dataset
def combine_datasets(df1, df2, dataset_type):
     df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)
     df_combined.to_csv(f'Backend/Data/combined_{dataset_type}_patients.csv', index = False)
     return(df_combined)

# Description:  This function randomly splits a dataset into two datasets.
# Inputs:       1) df: the dataset to be split
# Outputs:      1) df1: the first randomized dataset
#               2) the second randomized dataset
def randomly_split_datasets(df, dataset_type):
     df1, df2 = train_test_split(df, test_size=0.5, random_state=random_seed)
     df1.to_csv(f'Backend/Data/randomized_{dataset_type}_derivation_patients.csv', index = False)
     df2.to_csv(f'Backend/Data/randomized_{dataset_type}_validation_patients.csv', index = False)

df1 = pd.read_csv('Backend/Data/ARBsI_AI_data_part1-2023-04-25.csv')
df2 = pd.read_csv('Backend/Data/ARBsI_AI_data_part2-2023-04-25.csv')
df3 = pd.read_csv('Backend/Data/ARBsI_AI_data_CAP_patients-2023-08-31.csv') 
df4 = pd.read_csv('Backend/Data/ARBsI_AI_data_CAP_patients-part2-2024-03-14.csv')

df_combined = combine_datasets(df1, df2, 'COVID')
randomly_split_datasets(df_combined, 'COVID')
df_combined_cap = combine_datasets(df3, df4, 'CAP')
randomly_split_datasets(df_combined_cap, 'CAP')
