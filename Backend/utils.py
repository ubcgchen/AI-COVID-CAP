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

random_seed = 42

# Description:  This function calculates model metrics.
# Inputs:       1) Predictions
#               2) Ground truth
#               3) Probability scores for predictions
#               4) Model these metrics are claculated for (vasopressor, ventilator, or RRT)
# Outputs:      ROC curves for each dataset.
def calc_and_write_metrics(y_pred, y_test, y_prob_positive, model, classifier):
     # Calculate the metrics for the model
     accuracy = accuracy_score(y_test, y_pred)
     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
     sensitivity = tp / (tp + fn)
     specificity = tn / (tn + fp)
     ppv = tp / (tp + fp)
     npv = tn / (tn + fn)
     fpr, tpr, _ = roc_curve(y_test, y_prob_positive)
     roc_auc = auc(fpr, tpr)

     class_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
     values = [accuracy, sensitivity, specificity, ppv, npv, roc_auc]

     # Format data and write details to excel
     data = {
     'Metrics': ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV', "AUC ROC"],
     'Values': values
     }
     metrics = pd.DataFrame(data)

     # Write metrics to Excel
     metrics.to_excel(f'Backend/Model Metrics/{classifier}/{model}/metrics.xlsx', index=False)
     class_report.to_excel(f'Backend/Model Metrics/{classifier}/{model}/classification_report.xlsx', sheet_name='Classification Report')

# Description:  This function calculates roc curves as a visualization of the models performance for the validation dataset.
#               Serves as quick visualization tool. For the function that generates graphs for all data, see plot_roc in plot_graphs.py.
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

# Description:  This function randomly splits a dataset into two datasets.
# Inputs:       1) df: the dataset to be split
# Outputs:      1) df1: the first randomized dataset
#               2) the second randomized dataset
def randomly_split_datasets(df):
     df1, df2 = train_test_split(df, test_size=0.5, random_state=random_seed)
     df1.to_csv('Backend/Data/randomized_COVID_derivation_patients.csv', index = False)
     df2.to_csv('Backend/Data/randomized_COVID_validation_patients.csv', index = False)

