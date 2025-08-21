import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


# Load the dataset
df = pd.read_csv('mentalhealth-Solomon_Hutchings_oyHWDkP.csv')

# Plot style
plt.style.use('default')
sns.set_palette("husl")

# 1. Group courses based on the 5% rule
course_counts = df['Course'].value_counts()
total_students = len(df)
threshold = 0.05 * total_students  # 5% of total students   

# Identify courses with sufficient representation
major_courses = course_counts[course_counts >= threshold].index.tolist()

# Create new course category
df['Course_Grouped'] = df['Course'].apply(lambda x: x if x in major_courses else 'Other')

# Modified unistats function with missing value analysis
def unistats(df):
    output_df = pd.DataFrame(columns=['Count', 'Missing', 'Missing%', 'Unique', 'Type', 'Min',
                                      'Max', '25%', '50%', '75%', 'Mean', 'Median', 'Mode', 
                                      'Std', 'Skew', 'Kurt'])
    
    total_records = df.shape[0]
    
    for col in df.columns:
        # Calculate count, missing values, and missing percentage
        count = df[col].count()
        missing = total_records - count
        missing_percent = round((missing / total_records) * 100, 2) if total_records > 0 else 0
        
        # Calculate other statistics
        unique = df[col].nunique()
        dtype = str(df[col].dtype)

        if pd.api.types.is_numeric_dtype(df[col]):
            min_ = round(df[col].min(), 2)
            max_ = round(df[col].max(), 2)
            quar_1 = round(df[col].quantile(.25), 2)
            quar_2 = round(df[col].quantile(.50), 2)
            quar_3 = round(df[col].quantile(.75), 2)
            mean = round(df[col].mean(), 2)
            median = round(df[col].median(), 2)
            mode_val = df[col].mode()
            mode = round(mode_val.values[0], 2) if not mode_val.empty else '-'
            std = round(df[col].std(), 2)
            skew = round(df[col].skew(), 2)
            kurt = round(df[col].kurt(), 2)
            
            output_df.loc[col] = (count, missing, f'{missing_percent}%', unique, dtype, min_, max_, 
                                 quar_1, quar_2, quar_3, mean, median, mode, std, skew, kurt)
        else:
            output_df.loc[col] = (count, missing, f'{missing_percent}%', unique, dtype, '-', '-', 
                                 '-', '-', '-', '-', '-', '-', '-', '-', '-')
    
    return output_df

# Define the bivariate_stats function as provided
def bivariate_stats(df, label, roundto=4):
    output_df = pd.DataFrame(columns=['missing', 'p', 'r', 'y = m(x) + b', 'F', 'X2'])

    for feature in df.columns:
        if feature != label:
            df_temp = df[[feature, label]].dropna()
            missing = (df.shape[0] - df_temp.shape[0]) / df.shape[0]

            if pd.api.types.is_numeric_dtype(df_temp[feature]) and pd.api.types.is_numeric_dtype(df_temp[label]):
                m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
                output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), round(r, roundto),
                                          f'y = {round(m, roundto)}(x) + {round(b, roundto)}', '-', '-']

            elif not pd.api.types.is_numeric_dtype(df_temp[feature]) and not pd.api.types.is_numeric_dtype(df_temp[label]):
                contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
                X2, p, dof, expected = stats.chi2_contingency(contingency_table)
                output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), '-', '-', '-', round(X2, roundto)]

            else:
                if pd.api.types.is_numeric_dtype(df_temp[feature]):
                    num, cat = feature, label
                else:
                    num, cat = label, feature

                groups = [df_temp[df_temp[cat] == g][num] for g in df_temp[cat].unique()]
                F, p = stats.f_oneway(*groups)
                output_df.loc[feature] = [f'{missing:.2%}', round(p, roundto), '-', '-', round(F, roundto), '-']

    return output_df.sort_values(by=['p'])

#===========QUESTION 1: Execute the unistats function to get descriptive statistics============================
print("Q1: UNIVARIATE STATISTICS:")
print("======================")
univariate_results = unistats(df)
print(univariate_results)

print("\nQ1: BIVARIATE STATISTICS WITH DEPRESSION AS TARGET:")
print("================================================")
bivariate_results = bivariate_stats(df, 'Depression')
print(bivariate_results)

# Analyze specialist treatment rates by gender
gender_analysis = df.groupby('Gender')['SpecialistTreatment'].agg([
    ('Total', 'count'),
    ('Sought_Specialist', 'sum'),
    ('Rate', lambda x: (x.sum() / x.count()) * 100)
]).round(2)

#===================QUESTION 2: Analyze specialist treatment rates by gender and year of study===================   
year_of_study_analysis = df.groupby('YearOfStudy')['SpecialistTreatment'].agg([
    ('Total', 'count'),
    ('Sought_Specialist', 'sum'),
    ('Rate', lambda x: (x.sum() / x.count()) * 100)
]).round(2)


print("\nSPECIALIST TREATMENT RATE BY GENDER:")
print("====================================")
print(gender_analysis)

print("\nSPECIALIST TREATMENT RATE BY YEAR OF STUDY:")
print("===========================================")
print(year_of_study_analysis)

import pandas as pd

# Load the dataset
df = pd.read_csv('mentalhealth-Solomon_Hutchings_oyHWDkP.csv')

#QUESTION 2:  Display the first 5 records
print("\nQ2: FIRST 5 RECORDS OF THE DATASET:")
print("===============================")
print(df.head())

# Check unique values for potential Boolean columns to answer

print("\nQ2: UNIQUE VALUES FOR POTENTIAL BOOLEAN COLUMNS:")
print("============================================")
potential_bool_cols = ['Depression', 'Anxiety', 'PanicAttack', 'SpecialistTreatment', 'HasMentalHealthSupport']
for col in potential_bool_cols:
    unique_vals = df[col].unique()
    print(f"{col}: {sorted(unique_vals)}")

# QUESTION 3    Data quality assessment
print("\nQ3: DATA QUALITY ASSESSMENT:")
print("========================")
total_missing = univariate_results['Missing'].sum()
total_cells = df.shape[0] * df.shape[1]
completeness_rate = round((1 - (total_missing / total_cells)) * 100, 2) if total_cells > 0 else 100

print(f"Total records: {df.shape[0]}")
print(f"Total variables: {df.shape[1]}")
print(f"Total data cells: {total_cells}")
print(f"Total missing values: {total_missing}")
print(f"Overall data completeness: {completeness_rate}%")
print(f"Variables with missing values: {sum(univariate_results['Missing'] > 0)}")

# Check if any variable has significant missing data (>5%)
significant_missing = univariate_results[univariate_results['Missing'] / df.shape[0] > 0.05]
if len(significant_missing) > 0:
    print(f"\nVariables with significant missing data (>5%):")
    for col in significant_missing.index:
        missing_pct = significant_missing.loc[col, 'Missing%']
        print(f"  - {col}: {missing_pct}")
else:
    print(f"\nNo variables have significant missing data (>5%)")

# Execute the modified unistats function
print("Q3: UNIVARIATE STATISTICS WITH MISSING VALUE ANALYSIS:")
print("==================================================")
univariate_results = unistats(df)
print(univariate_results)


# Question 4 Execute bivariate analysis with SpecialistTreatment as the target variable
print("\n\nQ4: BIVARIATE STATISTICS: PREDICTORS OF SEEKING SPECIALIST TREATMENT")
print("================================================================")
print("Research Question: What causes university students to seek mental health treatment?")
print("Target Variable: SpecialistTreatment (0 = Did not seek treatment, 1 = Sought treatment)")
print("=" * 80)

bivariate_results = bivariate_stats(df, 'SpecialistTreatment')
print(bivariate_results)

# Extract and highlight the most significant predictors
significant_predictors = bivariate_results[bivariate_results['p'] < 0.05]
print(f"\nSIGNIFICANT PREDICTORS (p < 0.05): {len(significant_predictors)} variables")
print("=" * 60)

for feature in significant_predictors.index:
    p_value = significant_predictors.loc[feature, 'p']
    stats_type = ''
    
    if 'r' in significant_predictors.columns and pd.notna(significant_predictors.loc[feature, 'r']):
        correlation = significant_predictors.loc[feature, 'r']
        stats_type = f"Correlation: r = {correlation}"
    elif 'F' in significant_predictors.columns and pd.notna(significant_predictors.loc[feature, 'F']):
        f_value = significant_predictors.loc[feature, 'F']
        stats_type = f"ANOVA: F = {f_value}"
    elif 'X2' in significant_predictors.columns and pd.notna(significant_predictors.loc[feature, 'X2']):
        chi2 = significant_predictors.loc[feature, 'X2']
        stats_type = f"Chi-square: X² = {chi2}"
    
    print(f"• {feature}: p = {p_value}, {stats_type}")
    
    # QUESTION 5

    # Load the dataset
df = pd.read_csv('mentalhealth-Solomon_Hutchings_oyHWDkP.csv')

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# 1. Group courses based on the 5% rule
course_counts = df['Course'].value_counts()
total_students = len(df)
threshold = 0.05 * total_students  # 5% of total students

# Identify courses with sufficient representation
major_courses = course_counts[course_counts >= threshold].index.tolist()

# Create new course category
df['Course_Grouped'] = df['Course'].apply(lambda x: x if x in major_courses else 'Other')

# Verify the grouping
print("COURSE GROUPING RESULTS:")
print("========================")
print(f"Total students: {total_students}")
print(f"5% threshold: {threshold} students")
print(f"Major courses: {major_courses}")
print("\nCourse distribution after grouping:")
print(df['Course_Grouped'].value_counts())

# 2. Create bivariate charts for all features with SpecialistTreatment
def create_bivariate_charts(df, target_var):
    features = [col for col in df.columns if col != target_var and col != 'Course_Grouped']
    
    # Calculate grid size
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        if pd.api.types.is_numeric_dtype(df[feature]):
            # Scatter plot for numeric features
            sns.regplot(x=feature, y=target_var, data=df, ax=ax, 
                       scatter_kws={'alpha':0.5, 's':20}, line_kws={'color':'red'})
            ax.set_title(f'{feature} vs {target_var}')
            ax.set_ylabel('Probability of Treatment')
            
        else:
            # Bar plot for categorical features
            if df[feature].nunique() > 10:  # Too many categories
                ax.text(0.5, 0.5, f'Too many categories\n({df[feature].nunique()})', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{feature} (Too many categories)')
            else:
                # Calculate mean target by category
                grouped = df.groupby(feature)[target_var].mean().reset_index()
                bars = ax.bar(range(len(grouped)), grouped[target_var])
                ax.set_title(f'{feature} vs {target_var}')
                ax.set_ylabel('Mean Treatment Rate')
                ax.set_xticks(range(len(grouped)))
                ax.set_xticklabels(grouped[feature], rotation=90 if len(grouped) > 5 else 0)
                
                # Add value labels on bars
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# 3. Create specific charts for Course_Grouped and YearOfStudy
def create_specific_charts(df, target_var):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Chart 1: Course_Grouped vs SpecialistTreatment
    course_data = df.groupby('Course_Grouped')[target_var].agg(['mean', 'count', 'std']).reset_index()
    course_data['ci'] = 1.96 * course_data['std'] / np.sqrt(course_data['count'])
    
    bars = ax1.bar(range(len(course_data)), course_data['mean'], 
                  yerr=course_data['ci'], capsize=5, alpha=0.7)
    ax1.set_title('Specialist Treatment Rate by Course Group', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Treatment Rate', fontsize=12)
    ax1.set_xlabel('Course Group', fontsize=12)
    ax1.set_xticks(range(len(course_data)))
    ax1.set_xticklabels(course_data['Course_Grouped'], rotation=90)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}\nn={course_data["count"][i]}', 
                ha='center', va='bottom', fontsize=9)
    
    # Chart 2: YearOfStudy vs SpecialistTreatment
    year_data = df.groupby('YearOfStudy')[target_var].agg(['mean', 'count', 'std']).reset_index()
    year_data['ci'] = 1.96 * year_data['std'] / np.sqrt(year_data['count'])
    
    bars = ax2.bar(range(len(year_data)), year_data['mean'], 
                  yerr=year_data['ci'], capsize=5, alpha=0.7, color='orange')
    ax2.set_title('Specialist Treatment Rate by Year of Study', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Treatment Rate', fontsize=12)
    ax2.set_xlabel('Year of Study', fontsize=12)
    ax2.set_xticks(range(len(year_data)))
    ax2.set_xticklabels(year_data['YearOfStudy'], rotation=90)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}\nn={year_data["count"][i]}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistical summary
    print("\n\nQ5: STATISTICAL SUMMARY:")
    print("===================")
    print("Course Group Analysis:")
    print(course_data[['Course_Grouped', 'mean', 'count']].to_string(index=False))
    print(f"\nYear of Study Analysis:")
    print(year_data[['YearOfStudy', 'mean', 'count']].to_string(index=False))

# Execute the chart creation
print("\n\nQ5: CREATING BIVARIATE CHARTS FOR SPECIALIST TREATMENT")
print("=================================================")

# Create specific charts for Course and YearOfStudy
create_specific_charts(df, 'SpecialistTreatment')