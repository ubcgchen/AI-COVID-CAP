from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from openpyxl import load_workbook

def calc_and_write_metrics(y_pred, y_test, y_prob_positive, model):
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

     # workbook = load_workbook(f'Backend/Model Metrics/metrics_{model}_list.xlsx')
     # sheet = workbook.active
     # sheet.append(values)
     # workbook.save(f'Backend/Model Metrics/metrics_{model}_list.xlsx')

     # Write metrics to Excel
     metrics.to_excel(f'Backend/Model Metrics/metrics_{model}.xlsx', index=False)
     class_report.to_excel(f'Backend/Model Metrics/classification_report_{model}.xlsx', sheet_name='Classification Report')

def plot_roc_curve(y_test, y_prob_positive, model, intervention):
     fpr, tpr, _ = roc_curve(y_test, y_prob_positive)
     roc_auc = auc(fpr, tpr)

     data = {'intervention': intervention, 'fpr': fpr, 'tpr': tpr}
     df_to_write = pd.DataFrame(data)
     csv_path = f'Backend/Graphs/fpr_tpr_{model}.csv'
     df_to_write.to_csv(csv_path, index=False)

     # Plot the ROC curve
     plt.figure(figsize=(8, 8))
     plt.plot(fpr, tpr, color='darkorange', lw=1, label=f'AUC = {roc_auc:.2f}')
     plt.plot([0, 1], [0, 1], color='navy', lw=0.75, linestyle='--')
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.legend(loc='lower right')

     # Save the plot to a file (e.g., PNG)
     plt.savefig('Backend/Graphs/roc_curve_' + model + '.png')
