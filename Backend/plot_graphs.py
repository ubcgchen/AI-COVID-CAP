import pandas as pd
from model_params import *
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import seaborn as sns
import numpy as np
from scipy.stats import mannwhitneyu

def plot_roc():
    models = [vaso, vent, rrt]
    fpr_dict = {}
    tpr_dict = {}

    paths = [f'Backend/Graphs/fpr_tpr_rfc_',
            f'Backend/Graphs/fpr_tpr_rfc_Part2_',
            f'Backend/Graphs/fpr_tpr_rfc_CAP_',
            f'Backend/Graphs/fpr_tpr_rfc_CAP2_']

    titles= ['Derivation COVID-19 Pneumonia Dataset',
            'Validation COVID-19 Pneumonia Dataset',
            'CAP Dataset 1',
            'CAP Dataset 2',]

    file_names = ['roc_curve_OG',
                'roc_curve_Part2',
                'roc_curve_CAP',
                'roc_curve_CAP2']
    

    colors = [['Gold', 'DarkOrange', 'Firebrick'],
            ['lightgreen', 'cornflowerblue', 'black'],
            ['BurlyWood', 'OliveDrab', 'Chocolate'],
            ['black', 'dimgray', 'lightgray']]  # Adjust these colors

    for index in range(0, 4):
        for model in models:
            path = f'{paths[index]}{model["name"]}.csv' 
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

        plt.savefig('Backend/Graphs/' + file_names[index] + '.png')

def plot_correlation_graphs():
    index = 1

    models = [vaso, vent, rrt]
    preprocessed_datasets = ["Backend/Preprocessed Datasets/preprocessing_vaso.csv",
                             "Backend/Preprocessed Datasets/preprocessing_vent.csv",
                             "Backend/Preprocessed Datasets/preprocessing_rrt.csv",]
    importances = ["Backend/Model Metrics/importances_rfc_vaso.xlsx",
                   "Backend/Model Metrics/importances_rfc_vent.xlsx",
                   "Backend/Model Metrics/importances_rfc_rrt.xlsx"]

    x_labels = [["FiO2", "MAP", "1", "GCS", "Other Ethnicity", "3", "SaO2", "East Asian Ethnicity", "AST", "2"],
                ["FiO2", "MAP", "2", "Other Ethnicity", "GCS", "East Asian Ethnicity", "1", "0", "4", "AST"],
                ["Creatinine", "0", "Potassium", "3", "SaO2", "Temperature", "CKD?", "Respiratory Rate", "Troponin", "Hemoglobin"]]

    model = models[index]
    df = pd.read_csv(preprocessed_datasets[index])
    importances= pd.read_excel(importances[index]).head(10)
    features = importances.iloc[:,0]
    target = df[model["target"]]
    df = pd.concat([df[features], target], axis=1)

    correlation_matrix = df.corr()
    correlations = correlation_matrix[model["target"]].drop(model["target"])

    print(correlations)
        
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['red' if corr > 0 else 'blue' for corr in correlations.values]

    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list('custom', ['cornflowerblue', 'white', 'lightcoral'])

    norm = TwoSlopeNorm(vcenter=0, vmin=np.min(correlations), vmax=np.max(correlations))
    normalized_correlations = norm(correlations)

    # Create a bar plot using seaborn with custom X-axis labels and a diverging colormap
    # sns.barplot(x=x_labels[index], y=correlations.values, ax=ax, palette=colors)
    sns.barplot(x=np.arange(len(correlations)), y=correlations, ax=ax, palette=cmap(normalized_correlations))

    # Add labels and title
    ax.set_xlabel('Feature', fontfamily="Arial", fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontfamily="Arial", fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Correlation with {model["intervention"]} Use'.title(), fontfamily="Arial", fontsize=14, fontweight='bold')

    # Tilt X-axis labels at an angle
    ax.set_xticklabels(x_labels[index], rotation=45, ha='right')

    # Adjust tick label font size
    ax.tick_params(axis='both', labelsize=10)

    # Show the plot
    plt.tight_layout()
    plt.savefig('Backend/Graphs/correlation_plot_' + model["name"] + '.png', dpi=300)  # Save the plot to a high-resolution image

def plot_correlation_with_components():
    index = 1

    models = [vaso, vent, rrt]
    model = models[index]
    preprocessed_datasets = ["Backend/Preprocessed Datasets/preprocessing_vaso.csv",
                             "Backend/Preprocessed Datasets/preprocessing_vent.csv",
                             "Backend/Preprocessed Datasets/preprocessing_rrt.csv",]
    
    x_labels = [{
            'med_antifungal': "Antifungal Use",
            'co_other___1': "Other\nComorbidity",
            'bl_oxy_status': "Oxygen Therapy\nStatus",
            'demo_ethnicity___6': "Caucasian\nEthnicity",
            'med_abx': "Antibiotics",
            'med_steroid': "Steroids",
            'demo_ethnicity___3': "South Asian\nEthnicity",
            'co_dementia___1': "Dementia",
            "org_day0_fio2": "FiO2",
            "co_smoking": "Smoking Status",
        },
        {
            'med_abx': "Antibiotics",
            'co_dementia___1': "Comorbid\nDementia",
            'demo_ethnicity___6': "Caucasian\nEthnicity",
            'med_steroid': "Steroids",
            'org_day0_fio2': "FiO2",
            'demo_ethnicity___3': "South Asian\nEthnicity",
            'co_smoking': "Smoking Status",
            'bl_oxy_status': "Oxygen Therapy\nStatus",
            'med_antifungal': "Antifungal Use",
            'demo_age_years': "Age (years)"
        },
        {
            'med_steroid': "Steroids",
            'bl_lab_creatinine': 'Creatinine',
            'co_ckd___1': 'Chronic\n  Kidney Disease',
            'med_antifungal': "   Antifungal Use",
            'demo_ethnicity___6': "Caucasian\nEthnicity",
            'co_smoking': "Smoking Status",
            'bl_sao2': "SaO2",
            'org_day0_map': "MAP",
            'bl_lab_haemo': "Hemoglobin",
            'org_day0_fio2': "FiO2",
            'bl_resp_rate': "Respiratory Rate",

        },
    ]

    # Create a DataFrame with the lists
    df = pd.read_csv(preprocessed_datasets[index])

    # Specify the target variable (e.g., list4)
    target_variables = [["0", "1", "2", "3"],
                        ["0", "1", "4"],
                        ["0", "1", "2", "3"]]

    # Specify the variables you want to correlate with the target variables
    correlation_variables = list(set(df.columns) - set(target_variables[index]))

    # Calculate the correlation matrix for all variables
    correlation_matrix = df.corr()

    # Extract the subset of the correlation matrix
    subset_correlation_matrix = correlation_matrix.loc[correlation_variables, target_variables[index]]
    row_mask = (subset_correlation_matrix.abs() > 0.6).any(axis=1)
    subset_correlation_matrix = subset_correlation_matrix[row_mask]
    subset_correlation_matrix.rename(index=x_labels[index], inplace=True)

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 5))

    # Plot the heatmap using seaborn
    sns.heatmap(subset_correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.xlabel('Principal Component', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 16})
    plt.ylabel('Feature', fontdict={'family': 'Arial', 'weight': 'bold', 'size': 16})

    # Set the title and show the plot
    plt.title(f'{model["intervention"]} Use'.title(), fontdict={'family': 'Arial', 'weight': 'bold', 'size': 20})
    
    plt.tight_layout()
    plt.savefig('Backend/Graphs/PCA_HeatMap' + model["name"] + '.png', dpi=300)  # Save the plot to a high-resolution image
    
plot_correlation_with_components()

def plot_boxplots():
    y_labels = {
        'org_day0_fio2': 'FiO2',
        'org_day0_map': 'MAP',
        'ord_day0_gcs': 'GCS',
        '0': 'PC 0',
        '3': 'PC 3',
        'demo_ethnicity___2': 'East Asian Ethnicity',
        '7': 'PC 7',
        'bl_sao2': 'SaO2',
        'bl_lab_ast': 'AST',
        'demo_ethnicity___5': "Latin American Ethnicity",
        '4': 'PC 4',
        'demo_ethnicity___6': 'Caucasian Ethnicity',
        'bl_lab_creatinine': 'Creatinine',
        '1': 'PC 1',
        '2': 'PC 2',
        'bl_lab_potassium': 'Potassium',
        'bl_lab_troponin': 'Troponin',
        'med_antifungal': 'Antifungal Use',
        'bl_lab_haemo': 'Hemoglobin',
        'bl_temp': 'Body Temperature',
        'bl_hr': 'Heart Rate',
        'demo_ethnicity___8': 'Other Ethnicity',
        'co_ckd___1': 'CKD',
        'bl_resp_rate': "Respiratory Rate",
    }

    index = 0

    models = [vaso, vent, rrt]
    preprocessed_datasets = ["Backend/Preprocessed Datasets/preprocessing_vaso.csv",
                             "Backend/Preprocessed Datasets/preprocessing_vent.csv",
                             "Backend/Preprocessed Datasets/preprocessing_rrt.csv",]
    
    original_dataset = "Backend/Data/ARBsI_AI_data_part1-2023-04-25.csv"
    df = pd.read_csv(original_dataset)

    targets = [col for col in df.columns if col.startswith("med_vaso___")]
    conflicting_rows = []

    # Remove such conflicting patients.
    df = df.drop(conflicting_rows)
    df['med_vaso'] = df[targets[:-1]].any(axis=1).astype(int) # If target is vaso, build the med_vaso column
    
    df = df.drop(columns=targets)                                   # drop all initial target columns
    df = df.loc[:, ~df.columns.str.startswith('dly_day0_vaso___')]  # drop all day_0 med vaso columns

    # Get the rows where med_vaso___8 is 1 and look for the indices of rows where other columns are not all 0.
    # This incidicates conflicting rows as it means the patient both received and did not receive a vasopressor.
    for index, row in df.iterrows():
        if row['med_vaso___8'] == 1 and not all(row[targets[:-1]] == 0):
            conflicting_rows.append(index)
    
    importances = ["Backend/Model Metrics/importances_rfc_vaso.xlsx",
                   "Backend/Model Metrics/importances_rfc_vent.xlsx",
                   "Backend/Model Metrics/importances_rfc_rrt.xlsx"]

    # Set a common style for seaborn
    sns.set(style="whitegrid")
    target_labels = [{0: "No Vaso", 1: "Vaso"},
                     {0: "No Vent", 1: "Vent"},
                     {0: "No RRT", 1: "RRT"}]
    
    target_colors = {1: "lightcoral", 0: "cornflowerblue"}

    for index in range(0,3):
        model = models[index]
        df = pd.read_csv(preprocessed_datasets[index])
        features= pd.read_excel(importances[index]).head(10).iloc[:,0]
        target_label = target_labels[index]

        # Loop through each feature and create box plots
        for feature in features:
            plt.figure(figsize=(6, 6))

            # Create a box plot for target = 0
            ax = sns.boxplot(data=df, x=model["target"], y=feature, palette=target_colors)
            print(feature)
            print()

            # Set plot labels and title
            plt.xlabel("")
            ax.set_xticklabels([target_label[label] for label in ax.get_xticks()])
            
            medians = df.groupby(model["target"])[feature].median()
            for target_value, median_value in medians.items():
                print(f'Median for {feature} = {target_value}: {median_value}')
            
            plt.ylim(0, 1.2)
            plt.title(y_labels[feature])

            # Add statistical significance stars
            grouped_data = [df[df[model["target"]] == 0][feature], df[df[model["target"]] == 1][feature]]
            stat, p_value = mannwhitneyu(grouped_data[0], grouped_data[1])
            print(p_value)


            # Define the significance levels and corresponding stars
            alpha_levels = [0.001, 0.01, 0.05]
            significance_stars = ['***', '**', '*']

            # Add stars to the plot if the p-value is significant
            for alpha, star in zip(alpha_levels, significance_stars):
                if p_value < alpha:
                    plt.text(0.5, max(max(grouped_data[0]), max(grouped_data[1]))/1.2 + 0.05, star, fontsize=20, transform=plt.gca().transAxes,
                            horizontalalignment='center', verticalalignment='center')
                    break

            alpha_level = 0.05

            # Draw a bracket if the p-value is significant
            if p_value < alpha_level:
                # Explicitly specify the positions of the boxes
                box_positions = [0, 1]  # Adjust the positions as needed

                # Calculate the height of the bracket using the upper whisker of the first box
                bracket_height = max(max(grouped_data[0]), max(grouped_data[1])) + 0.05

                # Draw the bracket
                plt.hlines(bracket_height, box_positions[0], box_positions[1], color='black', linestyle='-', linewidth=2)

                for pos in box_positions:
                    plt.vlines(pos, bracket_height, max(grouped_data[pos]) + 0.04, color='black', linestyle='-', linewidth=2)
                    plt.hlines(max(grouped_data[pos]) + 0.04, pos - 0.04, pos + 0.04, color='black', linestyle='-', linewidth=2)

                # Show the plot
                plt.savefig('Backend/Graphs/Boxplots/boxplot_' + model["name"] + '_' + y_labels[feature] + '.png')
                # plt.show()
            
            # break

def plot_boxplots():
    y_labels = {
        'org_day0_fio2': 'FiO2',
        'org_day0_map': 'MAP',
        'ord_day0_gcs': 'GCS',
        '0': 'PC 0',
        '3': 'PC 3',
        'demo_ethnicity___2': 'East Asian Ethnicity',
        '7': 'PC 7',
        'bl_sao2': 'SaO2',
        'bl_lab_ast': 'AST',
        'demo_ethnicity___5': "Latin American Ethnicity",
        '4': 'PC 4',
        'demo_ethnicity___6': 'Caucasian Ethnicity',
        'bl_lab_creatinine': 'Creatinine',
        '1': 'PC 1',
        '2': 'PC 2',
        'bl_lab_potassium': 'Potassium',
        'bl_lab_troponin': 'Troponin',
        'med_antifungal': 'Antifungal Use',
        'bl_lab_haemo': 'Hemoglobin',
        'bl_temp': 'Body Temperature',
        'bl_hr': 'Heart Rate',
        'demo_ethnicity___8': 'Other Ethnicity',
        'co_ckd___1': 'CKD',
        'bl_resp_rate': "Respiratory Rate",
    }

    models = [vaso, vent, rrt]
    
    original_dataset = "Backend/Data/ARBsI_AI_data_part1-2023-04-25.csv"
    df = pd.read_csv(original_dataset)

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

    importances = ["Backend/Model Metrics/importances_rfc_vaso.xlsx",
                   "Backend/Model Metrics/importances_rfc_vent.xlsx",
                   "Backend/Model Metrics/importances_rfc_rrt.xlsx"]

    # Set a common style for seaborn
    sns.set(style="whitegrid")
    target_labels = [{0: "No Vaso", 1: "Vaso"},
                     {0: "No Vent", 1: "Vent"},
                     {0: "No RRT", 1: "RRT"}]
    
    target_colors = {1: "lightcoral", 0: "cornflowerblue"}

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
            
            y_min, y_max = ax.get_ylim()
            plt.title(y_labels[feature])

            # Add statistical significance stars
            grouped_data = [temp[temp[model["target"]] == 0][feature], temp[temp[model["target"]] == 1][feature]]
            stat, p_value = mannwhitneyu(grouped_data[0], grouped_data[1])
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
                plt.savefig('Backend/Graphs/Boxplots/boxplot_' + model["name"] + '_' + y_labels[feature] + '.png')
                # plt.show()
            
            # break

