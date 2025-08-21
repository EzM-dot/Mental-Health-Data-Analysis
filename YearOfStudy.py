# Analyze specialist treatment rates by gender
gender_analysis = df.groupby('Gender')['SpecialistTreatment'].agg([
    ('Total', 'count'),
    ('Sought_Specialist', 'sum'),
    ('Rate', lambda x: (x.sum() / x.count()) * 100)
]).round(2)

# Analyze by year of study
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