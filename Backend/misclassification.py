# Author:       George Chen
# Date:         September 15, 2024
# Email:        gschen@student.ubc.ca
# Description:  This file contains the code to perform misclassification analysis for all three models (vaso, vent, RRT)
#               trained on the derivation COVID-19 dataset. 

# Imports
import pandas as pd
from model_params import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.contingency_tables import mcnemar
from columns import feature_mapping

# Description:  Performs McNemar's test to determine if there is a statistically significant difference in the frequency between
#               type I and type II errors.
# Inputs:       1) model whose misclassifications you wish to analyze
#               2) dataframe of misclassified data.
#               3) dataframe of correctly classified data.
# Outputs:      1) McNemar's test statistic.
#               2) p-value
def false_positive_negative_analysis(curr_model, misclassified_data, correct_data):

    target = curr_model["target"]
    misclassifications = misclassified_data[target]
    correct_classifications = correct_data[target]

    a = (correct_classifications == 1).sum()    # True positives
    b = (misclassifications == 0).sum()         # False negatives
    c = (misclassifications == 1).sum()         # False positives
    d = (correct_classifications == 0).sum()    # True negatives

    # Create a 2x2 contingency table
    contingency_table = [[a, b], [c, d]]

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=True)

    # Output the test results
    print("McNemar's test statistic:", result.statistic)
    print("P-value:", result.pvalue)

# Description:  Overlays the distributions of feature values for correctly classified and incorrectly classified examples, 
#               if distributions that are significantly different from each other. The Mann-Whitney U test is used to 
#               determine if the two distributions are significantly different.
# Inputs:       1) model whose misclassifications you wish to analyze
#               2) dataframe of misclassified data.
#               3) dataframe of correctly classified data.
# Outputs:      Overlayed histograms of distribution of features for correctly and incorrectly classified examples, 
#               if these distributions are significantly different.
def plot_misclassified_features(curr_model, misclassified_data, correct_data):

    # Get misclassified and correctly classified examples from the original (unprocessed) derivation data.
    df = pd.read_csv('Backend/Data/ARBsI_AI_data_part2-2023-04-25.csv')
    misclassified_indices = misclassified_data.index.tolist()
    correct_indices = correct_data.index.tolist()
    misclassified_data_new = df.iloc[misclassified_indices]
    correct_data_new = df.iloc[correct_indices]

    # Extract all feature names, and exclude the target column, e.g., 'dis_ventilation' and those not in the unprocessed derivation data.
    features_to_compare = [col for col in misclassified_data.columns if col != curr_model["target"]]
    features_to_compare = [col for col in features_to_compare if col in df.columns]

    # Iterate through all features
    for feature in features_to_compare:
        temp_misclassified = misclassified_data_new.dropna(subset=[feature])
        temp_correct = correct_data_new.dropna(subset=[feature])
        
        # Plot the overlayed distributions of the given feature if significantly different.
        stat, p_value = mannwhitneyu(temp_misclassified[feature], temp_correct[feature], alternative='two-sided')
        if p_value < 0.05:
            print(f'Test for {feature}:')
            print(f'   U statistic: {stat}')
            print(f'   p-value: {p_value}')

            sns.histplot(data=temp_misclassified, x=feature, kde=True, color='coral', 
                        stat = 'density', label='Misclassified', alpha=0.5, common_norm=False, bins=15)
            sns.histplot(data=temp_correct, x=feature, kde=True, color='cornflowerblue', 
                        stat = 'density', label='Correctly Classified', alpha=0.5, common_norm=False, bins=15, lw=0.25)
            plt.title(f'Distribution of {feature_mapping[feature]} By Classification'.title(),  
                      fontdict={'family': 'Arial', 'weight': 'bold', 'size': 15})
            plt.xlabel(f'{feature_mapping[feature]}'.title(), fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
            plt.ylabel('Density', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
            plt.legend()
            plt.savefig('Backend/Graphs/Feature Distributions/dist_' + curr_model["name"] + '_' + feature + '.png')
            plt.clf()
            plt.cla()

# Description:  Overlays the distribution of confidences in the model's incorrect classifications versus its correct classifications.
# Inputs:       Model whose misclassifications you wish to analyze
# Outputs:      Overlayed histograms of distribution of confidences correctly and incorrectly classified examples.
def plot_misclassified_distributions(curr_model, a, b):
    # Load model confidences in its falsely negative classifications, falsely positive classifications, and correctly classified examples.
    false_negatives = pd.read_csv('Backend/Misclassification Analysis/false_negative_probabilities_' + curr_model["name"] + '.csv', skiprows=1, index_col=0).squeeze()
    false_positives = pd.read_csv('Backend/Misclassification Analysis/false_positive_probabilities_' + curr_model["name"] + '.csv', skiprows=1, index_col=0).squeeze()
    correct_probabilities = pd.read_csv('Backend/Misclassification Analysis/correct_probabilities_' + curr_model["name"] + '.csv', skiprows=1, index_col=0).squeeze()

    # compare the model's confidence for its falsely negative predictions, versus its false positive predictions.
    _, p_value = mannwhitneyu(false_negatives, false_positives, alternative='two-sided')
    print(p_value)
    # compare the model's confidence for its incorrect predictions, versus its correct predictions.
    _, p_value = mannwhitneyu(pd.concat([false_positives, false_negatives]), correct_probabilities, alternative='two-sided')
    print(p_value)

    # Plot histograms
    sns.histplot(data=false_negatives, bins=20, alpha=0.5, label='False Negatives', stat='density', edgecolor ='black', lw = '0.3', color='darkkhaki')
    sns.histplot(data=false_positives, bins=20, alpha=0.5, label='False Positives', stat='density', edgecolor ='black', lw = '0.3', color='darkseagreen')

    # Add labels and title
    plt.xlabel('Model Confidence', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
    plt.ylabel('Density', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
    plt.legend(prop={'family': 'Arial'})
    plt.savefig('Backend/Graphs/FN_FP_Dist_' + curr_model["name"] + '.png')
    plt.clf()
    plt.cla()

    # Show the plot
    sns.histplot(data=pd.concat([false_positives, false_negatives], axis=0), bins=20, alpha=0.5, label='Incorrect Classification', stat = 'density', edgecolor ='black', lw = '0.3', color='coral')
    sns.histplot(data=correct_probabilities, bins=20, alpha=0.5, label='Correct Classification', stat = 'density', edgecolor='black', lw='0.3', color='cornflowerblue')

    # Add labels and title
    plt.xlabel('Model Confidence', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
    plt.ylabel('Density', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
    plt.title(curr_model["intervention"].title(), fontdict={'family': 'Arial', 'weight': 'bold', 'size': 16})
    plt.legend(prop={'family': 'Arial'})
    plt.savefig('Backend/Graphs/Correct_Incorrect_Dist_' + curr_model["name"] + '.png')

# Description:  Entrance into analysis functions
# Inputs:       1) model whose misclassifications you wish to analyze
#               2) analysis you wish to perofrm.
def analyze_misclassification(curr_model, analysis):
    # Load in misclassified examples for selected model.
    misclassified_path = 'Backend/Misclassification Analysis/incorrectly_classified_examples_' + curr_model["name"] + '.csv'
    misclassified_data = pd.read_csv(misclassified_path, index_col = 0)

    # Load in correctly classified examples for selected model.
    correct_path = 'Backend/Misclassification Analysis/correctly_classified_examples_' + curr_model["name"] + '.csv'
    correct_data = pd.read_csv(correct_path, index_col = 0)

    analysis(curr_model, misclassified_data, correct_data)

analyze_misclassification(vaso, plot_misclassified_distributions)