# Author:       George Chen
# Date:         September 15, 2024
# Email:        gschen@student.ubc.ca
# Description:  This file contains defined lists of grouped columns that are used together for processing/analysis.

# Columns that do not have any medical meaning, or columns that are duplicated.
drop_non_medical = ["ii", "admission_complete", "comorbidities_complete", "medications_complete",  "discharge_complete", 
                   "demographics_complete", "org_day0_status", "organ_dysfunction_day_0_complete", "demo_ethnicity___9",
                   "med_steroid_route", "org_day0_mechvent", "org_day0_creatinine", "org_day0_spo2", "bl_lab_platelets"] 

# columns that pertain to covid status
drop_covidstatus = ['bl_admission_reason', 'bl_pathogen___0', 'bl_pathogen___1', 'bl_pathogen___2', 
                    'bl_pathogen___3', 'bl_pathogen___4', 'bl_pathogen___5', 'bl_pathogen___6']

# Map the label of the column in the database to an interpretable name.
feature_mapping = {
    'demo_age_years': 'Age (years)',
    'demo_ethnicity___2': 'East Asian Ethnicity',
    'demo_ethnicity___3': 'South Asian Ethnicity',
    'demo_ethnicity___4': 'West Asian Ethnicity',
    'demo_ethnicity___5': 'Latin American Ethnicity',
    'demo_ethnicity___6': 'White Ethnicity',
    'demo_ethnicity___7': 'Aboriginal/First Nations Ethnicity',
    'demo_ethnicity___8': 'Other Ethnicity',
    'co_dementia___1': 'Dementia',
    'co_smoking': 'Smoking Status',
    'co_ckd___1': 'CKD?',
    'co_other___1': "Other\nComorbidity",
    'bl_temp': 'Temperature',
    'bl_hr': 'Heart Rate',
    'bl_resp_rate': 'Respiratory Rate',
    'bl_kg': 'Weight (kg)',
    'bl_sao2': 'Oxygen Saturation',
    'bl_oxy_status': 'Oxygen Status',
    'bl_lab_wbc': 'White Blood Cell Count (WBC)',
    'bl_lab_haemo': 'Haemoglobin',
    'bl_lab_creatinine': 'Creatinine',
    'bl_lab_potassium': 'Potassium',
    'bl_lab_alt': 'ALT',
    'bl_lab_ast': 'AST',
    'bl_lab_inr': 'INR',
    'bl_lab_troponin': 'Troponin Level',
    'org_day0_fio2': 'FiO2',
    'org_day0_platlet': 'Platelet Count',
    'ord_day0_gcs': 'Glasgow Coma Scale',
    'org_day0_bilirubin': 'Bilirubin Level',
    'org_day0_map': 'MAP',
    'org_day0_creatinine': 'Creatinine Level',
    'med_abx': 'Antibiotic Use',
    'med_steroid': 'Steroid Use',
    'med_antifungal': 'Antifungal Use',
    '0': 'Principal Component 0',
    '1': 'Principal Component 1',
    '2': 'Principal Component 2',
    '3': 'Principal Component 3',
    '4': 'Principal Component 4',
    '5': 'Principal Component 5',
    '6': 'Principal Component 6',
    '7': 'Principal Component 7',
}