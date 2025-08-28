import pandas as pd

# Nustatome stulpelių pavadinimus pagal Breast Cancer Wisconsin rinkinį
column_names = ["ID", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape", 
                "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei", 
                "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]

# Įkeliame duomenis
file_path = "breast-cancer-wisconsin.data"
df = pd.read_csv(file_path, names=column_names)

# Pašaliname ID stulpelį
df.drop(columns=["ID"], inplace=True)

# Pakeičiame klaustukus (?) į NaN
df.replace("?", pd.NA, inplace=True)

# Pašaliname trūkstamas reikšmes turinčias eilutes
df.dropna(inplace=True)

# Pakeičiame 'Class' reikšmes:
df["Class"] = df["Class"].astype(int).replace({2: 0, 4: 1})

# Atsitiktinai permaišome eilutes
df = df.sample(frac=1, random_state=100).reset_index(drop=True)

# Išsaugome išvalytus duomenis
df.to_csv("cleaned_breast_cancer.csv", index=False)

# Parodome pirmas 5 eilutes
print(df.head())
