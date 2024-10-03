# Author:       George Chen
# Date:         September 15, 2024
# Email:        gschen@student.ubc.ca
# Description:  This file contains the code to plot graphs to summarize the performance of our machine learning model.

# Imports
import pandas as pd
from model_params import *
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from columns import feature_mapping

# Description:  This function plots all ROC curves.
# Inputs:       Nil
# Outputs:      ROC curves for each dataset.
def plot_roc():
    models = [vaso, vent, rrt]
    fpr_dict = {}
    tpr_dict = {}

    result_sets = ['rfc','validation data/Part2', 'validation data/CAP', 'validation data/CAP2']

    # Data to plot for ROC curves
    # paths = [f'Backend/Graphs/fpr_tpr_rfc_',
    #         f'Backend/Graphs/fpr_tpr_rfc_Part2_',
    #         f'Backend/Graphs/fpr_tpr_rfc_CAP_',
    #         f'Backend/Graphs/fpr_tpr_rfc_CAP2_']
    
    # Titles to be used for each curve
    titles= ['Derivation COVID-19 Pneumonia Dataset',
            'Validation COVID-19 Pneumonia Dataset',
            'CAP Dataset 1',
            'CAP Dataset 2',]
    
    # File names to be used for each curve
    file_names = ['roc_curve_OG',
                'roc_curve_Part2',
                'roc_curve_CAP',
                'roc_curve_CAP2']
    
    # Colour scheme to be used for each curve
    colors = [['Gold', 'DarkOrange', 'Firebrick'],
            ['lightgreen', 'cornflowerblue', 'black'],
            ['BurlyWood', 'OliveDrab', 'Chocolate'],
            ['black', 'dimgray', 'lightgray']] 
    
    index = 0

    # Plot ROC curves for each dataset.
    for result_set in result_sets:
        for model in models:
            path = f'Backend/Model Metrics/{result_set}/{model["name"]}/fpr_tpr.csv' 
            df = pd.read_csv(path)
            fpr_dict[model["name"]] = df["fpr"].tolist()
            tpr_dict[model["name"]] = df["tpr"].tolist()

        plt.figure(figsize=(8, 8))

        for model, (fpr, tpr, color) in zip(models, zip(fpr_dict.values(), tpr_dict.values(), colors[index])):
            roc_auc = "{:.4f}".format(auc(fpr, tpr))
            plt.plot(fpr, tpr, lw=2, label=f'{model["intervention"]}, AUC {roc_auc}', color=color)

        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')  # Diagonal reference line

        plt.xlabel('False Positive Rate', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 16})
        plt.ylabel('True Positive Rate', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 16})
        plt.legend(loc='lower right', prop={'family': 'Arial', 'weight': 'normal', 'size': 14})
        plt.title(titles[index], fontdict={'family': 'Arial', 'weight': 'bold', 'size': 20})

        plt.savefig('Backend/Graphs/ROC Curves/' + file_names[index] + '.png')
        index += 1

# Description:  This function plots the Pearson's correlation coefficient of the 10 features most correlated with the 
#               specified target, by magnitude.
# Inputs:       The model used to plot the graph (vasopressor, ventilator, or RRT)
# Outputs:      Correlation curves for the specified endpoint.
def plot_correlation_graphs(model):

    # Load in the preprocessed dataset
    preprocessed_dataset = "Backend/Preprocessed Datasets/preprocessing_" + model["name"] + ".csv"
    df = pd.read_csv(preprocessed_dataset)

    # The 10 most correlated features as readable names for the x-axis.
    t10_feature_importances = pd.read_excel("Backend/Model Metrics/rfc/" + model["name"] + "/importances.xlsx", skiprows=0).head(10)
    t10_feature_names = t10_feature_importances.iloc[:, 0].astype(str).tolist()
    x_labels = [feature_mapping[feature] for feature in t10_feature_names]

    # Perform correlation analysis between the top 10 most important features and the target.
    target = df[model["target"]]
    df = pd.concat([df[t10_feature_names], target], axis=1)
    correlation_matrix = df.corr()
    correlations = correlation_matrix[model["target"]].drop(model["target"])
    print(correlations)
        
    # Set up the matplotlib figure
    _, ax = plt.subplots(figsize=(7, 6))

    cmap = LinearSegmentedColormap.from_list('custom', ['cornflowerblue', 'white', 'lightcoral'])

    norm = TwoSlopeNorm(vcenter=0, vmin=np.min(correlations), vmax=np.max(correlations))
    normalized_correlations = norm(correlations)

    # Create a bar plot using seaborn with custom X-axis labels and a diverging colormap
    sns.barplot(x=np.arange(len(correlations)), y=correlations, ax=ax, palette=cmap(normalized_correlations))

    # Add labels and title
    ax.set_xlabel('Feature', fontfamily="Arial", fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontfamily="Arial", fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Correlation with {model["intervention"]} Use'.title(), fontfamily="Arial", fontsize=14, fontweight='bold')

    # Tilt X-axis labels at an angle
    ax.set_xticklabels(x_labels, rotation=45, ha='right')

    # Adjust tick label font size
    ax.tick_params(axis='both', labelsize=10)

    # Show the plot
    plt.tight_layout()
    plt.savefig('Backend/Graphs/Correlation Plots/correlation_plot_' + model["name"] + '.png', dpi=300)  # Save the plot


def is_integer_string(series):
    return series.apply(lambda x: x.isdigit()).all()

# Description:  This function processes the original dataset for the purposes of plotting boxplots. It goes through the example
#               preprocessing steps without editing the values, so that the original values can be plotted.
# Inputs:       Original dataframe
# Outputs:      Processed dataframe
def process_dataset_for_boxplots(df):
    targets = [col for col in df.columns if col.startswith("med_vaso___")]
    conflicting_rows = []

    # Get the rows where med_vaso___8 is 1 and look for the indices of rows where other columns are not all 0.
    # This incidicates conflicting rows as it means the patient both received and did not receive a vasopressor.
    for index, row in df.iterrows():
        if row['med_vaso___8'] == 1 and not all(row[targets[:-1]] == 0):
            conflicting_rows.append(index)

    # Remove such conflicting patients.
    df = df.drop(conflicting_rows)
    df['med_vaso'] = df[targets[:-1]].any(axis=1).astype(int) # If target is vaso, build the med_vaso column
    
    df = df.drop(columns=targets)                                   # drop all initial target columns
    df = df.loc[:, ~df.columns.str.startswith('dly_day0_vaso___')]  # drop all day_0 med vaso columns
    return df

# Description:  This function creates side-by-side boxplots of the feature distributions for the 10 features with the highest
#               feature importance in patients who received each intervention and patients who did not receive each intervention, 
#               if the distributions are significantly different, as per the Mann-Whitney U test.
# Inputs:       Nil
# Outputs:      Box plots of the feature distributions for patients who received each intervention and patients 
#               who did not receive each intervention
def plot_boxplots():

    models = [vaso, vent, rrt]
    
    # The original dataset is used here as we want to plot the distribution of original data rather than preprocessed data.
    # We pre-process the original dataset using the same strategies as the pre-processed dataset, but omit the 
    # step where we normalize and standardize the data.
    original_dataset = "Backend/Data/ARBsI_AI_data_part1-2023-04-25.csv" 
    df = pd.read_csv(original_dataset)
    df = process_dataset_for_boxplots(df)

    # Get feature importances. Show the boxplots for the 10 most important features.
    importances = ["Backend/Model Metrics/rfc/vaso/importances.xlsx",
                   "Backend/Model Metrics/rfc/vent/importances.xlsx",
                   "Backend/Model Metrics/rfc/rrt/importances.xlsx"]

    # Plot the graphs.
    sns.set(style="whitegrid")
    target_labels = [{0: "No Vaso", 1: "Vaso"},
                     {0: "No Vent", 1: "Vent"},
                     {0: "No RRT", 1: "RRT"}]
    
    target_colors = {1: "lightcoral", 0: "cornflowerblue"}

    # For each endpoint (vasopressor, ventilator, and RRT)
    for index in range(0,3):
        model = models[index]
        features= pd.read_excel(importances[index]).head(10).iloc[:,0]
        features = [feature for feature in features if feature in df.columns]
        target_label = target_labels[index]

        # Loop through each feature and create box plots
        for feature in features:
            plt.figure(figsize=(2, 6))
            temp = df.dropna(subset=[feature])

            # Create a box plot for target = 0
            ax = sns.boxplot(data=temp, x=model["target"], y=feature, palette=target_colors)
            print(feature)

            # Set plot labels and title
            plt.xlabel("")
            ax.set_ylabel('')
            ax.set_xticklabels([target_label[label] for label in ax.get_xticks()])
            
            medians = temp.groupby(model["target"])[feature].median()
            for target_value, median_value in medians.items():
                print(f'Median for {feature} = {target_value}: {median_value}')
            
            _, y_max = ax.get_ylim()
            plt.title(feature_mapping[feature])

            # Add statistical significance stars
            grouped_data = [temp[temp[model["target"]] == 0][feature], temp[temp[model["target"]] == 1][feature]]
            _, p_value = mannwhitneyu(grouped_data[0], grouped_data[1])
            print("p-value: " + str(p_value))

            # Define the significance levels and corresponding stars
            alpha_levels = [0.001, 0.01, 0.05]
            significance_stars = ['***', '**', '*']

            # Add stars to the plot if the p-value is significant
            for alpha, star in zip(alpha_levels, significance_stars):
                if p_value < alpha:
                    plt.text(0.5, max(max(grouped_data[0]), max(grouped_data[1])) + y_max*0.07, star, fontsize=20,
                            horizontalalignment='center', verticalalignment='center')
                    break

            alpha_level = 0.05

            # Draw a bracket if the p-value is significant
            if p_value < alpha_level:
                # Explicitly specify the positions of the boxes
                box_positions = [0, 1]  # Adjust the positions as needed

                # Calculate the height of the bracket using the upper whisker of the first box
                bracket_height = max(max(grouped_data[0]), max(grouped_data[1])) + y_max*0.07

                # Draw the bracket
                plt.hlines(bracket_height, box_positions[0], box_positions[1], color='black', linestyle='-', linewidth=2)

                for pos in box_positions:
                    plt.vlines(pos, bracket_height, max(grouped_data[pos]) + y_max*0.04, color='black', linestyle='-', linewidth=2)
                    plt.hlines(max(grouped_data[pos]) + y_max*0.04, pos - 0.04, pos + 0.04, color='black', linestyle='-', linewidth=2)

                # Show the plot
                plt.tight_layout()
                plt.savefig('Backend/Graphs/Boxplots/boxplot_' + model["name"] + '_' + feature_mapping[feature] + '.png')            

plot_boxplots()