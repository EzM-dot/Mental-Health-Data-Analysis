import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for the plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Data from the analysis
course_data = pd.DataFrame({
    'Course_Grouped': ['BCS', 'BIT', 'Engineering', 'Other'],
    'mean': [0.163842, 0.089109, 0.077778, 0.027675],
    'count': [177, 101, 180, 542],
    'std': [0.3709, 0.2865, 0.2686, 0.1641]  # Approximate standard deviations
})

year_data = pd.DataFrame({
    'YearOfStudy': ['Year 1', 'Year 2', 'Year 3', 'Year 4'],
    'mean': [0.067961, 0.109489, 0.0375, 0.0],
    'count': [412, 274, 240, 74],
    'std': [0.2519, 0.3127, 0.1903, 0.0]  # Approximate standard deviations
})

# Calculate confidence intervals
course_data['ci'] = 1.96 * course_data['std'] / np.sqrt(course_data['count'])
year_data['ci'] = 1.96 * year_data['std'] / np.sqrt(year_data['count'])

def create_visualizations():
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Course Group Visualization
    sns.barplot(x='Course_Grouped', y='mean', data=course_data, ax=ax1, 
                order=['BCS', 'BIT', 'Engineering', 'Other'])
    ax1.set_title('Treatment Rate by Course Group', fontsize=14, pad=15)
    ax1.set_xlabel('Course Group', fontsize=12)
    ax1.set_ylabel('Proportion Sought Treatment', fontsize=12)
    ax1.set_ylim(0, 0.25)
    
    # Add value labels
    for i, (_, row) in enumerate(course_data.iterrows()):
        ax1.text(i, row['mean'] + 0.01, 
                f"{row['mean']*100:.1f}%\n(n={row['count']})", 
                ha='center', va='bottom', fontsize=10)
    
    # 2. Year of Study Visualization
    sns.barplot(x='YearOfStudy', y='mean', data=year_data, ax=ax2,
               order=['Year 1', 'Year 2', 'Year 3', 'Year 4'])
    ax2.set_title('Treatment Rate by Year of Study', fontsize=14, pad=15)
    ax2.set_xlabel('Year of Study', fontsize=12)
    ax2.set_ylabel('Proportion Sought Treatment', fontsize=12)
    ax2.set_ylim(0, 0.25)
    
    # Add value labels
    for i, (_, row) in enumerate(year_data.iterrows()):
        ax2.text(i, row['mean'] + 0.01, 
                f"{row['mean']*100:.1f}%\n(n={row['count']})", 
                ha='center', va='bottom', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('treatment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations have been saved as 'treatment_analysis.png'")

if __name__ == "__main__":
    create_visualizations()
