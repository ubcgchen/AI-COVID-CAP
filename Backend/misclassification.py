import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import pickle
from model_params import *
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.contingency_tables import mcnemar
from columns import feature_mapping

# Load the model
index = 2

models = [vaso, vent, rrt]
curr_model = models[index]

model_paths = ['rfc_vaso.pkl', 'rfc_vent.pkl', 'rfc_rrt.pkl']
model_path = model_paths[index]

with open(curr_model["model"], 'rb') as file:
    model = pickle.load(file)

# Load misclassified examples from CSV
misclassified_path = 'Backend/Misclassification Analysis/incorrectly_classified_examples_' + curr_model["name"] + '.csv'
misclassified_data = pd.read_csv(misclassified_path, index_col = 0)

correct_path = 'Backend/Misclassification Analysis/correctly_classified_examples_' + curr_model["name"] + '.csv'
correct_data = pd.read_csv(correct_path, index_col = 0)

def false_positive_negative_analysis():
    # VASO
    # a = 43  # True positives
    # b = 92  # False negatives
    # c = 18   # False positives
    # d = 490  # True negatives

    # VENT
    a = 43  # True positives
    b = 81  # False negatives
    c = 16   # False positives
    d = 503  # True negatives

    # Create a 2x2 contingency table
    contingency_table = [[a, b], [c, d]]

    # Perform McNemar's test
    result = mcnemar(contingency_table, exact=True)

    # Output the test results
    print("McNemar's test statistic:", result.statistic)
    print("P-value:", result.pvalue)

def plot_misclassified_features():
    df = pd.read_csv('Backend/Data/ARBsI_AI_data_part2-2023-04-25.csv')
    # Get all feature names (exclude the label column, e.g., 'dis_ventilation')
    features_to_compare = [col for col in misclassified_data.columns if col != curr_model["target"]]
    false_positives = misclassified_data[misclassified_data[curr_model["target"]] == 1]
    false_negatives = misclassified_data[misclassified_data[curr_model["target"]] == 0]

    misclassified_indices = misclassified_data.index.tolist()
    correct_indices = correct_data.index.tolist()

    print(misclassified_indices)
    print(correct_indices)

    features_to_compare = [col for col in features_to_compare if col in df.columns]

    misclassified_data_new = df.iloc[misclassified_indices]
    correct_data_new = df.iloc[correct_indices]

    for feature in features_to_compare:
        temp_misclassified = misclassified_data_new.dropna(subset=[feature])
        temp_correct = correct_data_new.dropna(subset=[feature])
        
        stat, p_value = mannwhitneyu(temp_misclassified[feature], temp_correct[feature], alternative='two-sided')
        # Vent: patients with abx and steroids are more often misclassified when compared.

        if p_value < 0.05:
            print(f'Test for {feature}:')
            print(f'   U statistic: {stat}')
            print(f'   p-value: {p_value}')

            sns.histplot(data=temp_misclassified, x=feature, kde=True, color='coral', 
                        stat = 'density', label='Misclassified', alpha=0.5, common_norm=False, bins=15)
            # sns.histplot(data=false_negatives, x=feature, kde=True, color='darkseagreen', 
            #             stat = 'density', label='False Negatives', alpha=0.5, common_norm=False, bins=15, lw=0.25)
            # sns.histplot(data=false_positives, x=feature, kde=True, color='coral', 
            #             stat = 'density', label='False Positives', alpha=0.5, common_norm=False, bins=15, lw=0.25)
            sns.histplot(data=temp_correct, x=feature, kde=True, color='cornflowerblue', 
                        stat = 'density', label='Correctly Classified', alpha=0.5, common_norm=False, bins=15, lw=0.25)
            plt.title(f'Distribution of {feature_mapping[feature]} By Classification'.title(),  
                      fontdict={'family': 'Arial', 'weight': 'bold', 'size': 15})
            plt.xlabel(f'{feature_mapping[feature]}'.title(), fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
            plt.ylabel('Density', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
            plt.legend()
            # plt.show()
            plt.savefig('Backend/Graphs/Feature Distributions/dist_' + curr_model["name"] + '_' + feature + '.png')
            plt.clf()
            plt.cla()

def plot_misclassified_distributions():
    false_negatives = pd.read_csv('Backend/Misclassification Analysis/false_negative_probabilities_' + curr_model["name"] + '.csv', skiprows=1, index_col=0).squeeze()
    false_positives = pd.read_csv('Backend/Misclassification Analysis/false_positive_probabilities_' + curr_model["name"] + '.csv', skiprows=1, index_col=0).squeeze()
    correct_probabilities = pd.read_csv('Backend/Misclassification Analysis/correct_probabilities_' + curr_model["name"] + '.csv', skiprows=1, index_col=0).squeeze()

    stat, p_value = mannwhitneyu(false_negatives, false_positives, alternative='two-sided')
    print(p_value)
    stat, p_value = mannwhitneyu(pd.concat([false_positives, false_negatives]), correct_probabilities, alternative='two-sided')
    print(p_value)

    # Plot histograms
    sns.histplot(data=false_negatives, bins=20, alpha=0.5, label='False Negatives', stat='density', edgecolor ='black', lw = '0.3', color='darkkhaki')
    sns.histplot(data=false_positives, bins=20, alpha=0.5, label='False Positives', stat='density', edgecolor ='black', lw = '0.3', color='darkseagreen')

    # Add labels and title
    plt.xlabel('Model Confidence', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
    plt.ylabel('Density', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 11})
    # plt.title('Distribution of Model Confidences', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 16})
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

plot_misclassified_features()