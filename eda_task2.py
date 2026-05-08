# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("titanic.csv")

print(df.columns)
print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# -------------------------------
# SUMMARY STATISTICS
# -------------------------------
print("\nSummary Statistics:\n", df.describe(include='all'))

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# -------------------------------
# 1. DISTRIBUTION ANALYSIS
# -------------------------------
# Histogram + Boxplot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(df['Age'], bins=30, kde=True, ax=ax[0])
ax[0].set_title("Age Distribution")

sns.boxplot(x='Pclass', y='Fare', data=df, ax=ax[1])
ax[1].set_title("Fare by Class")

plt.show()

# Countplot
plt.figure(figsize=(8, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# -------------------------------
# 2. CORRELATION & RELATIONSHIP
# -------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Crosstab
print("\nSurvival % by Class:\n")
print(pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100)

# -------------------------------
# 3. OUTLIER DETECTION
# -------------------------------
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.show()

from scipy import stats

z_scores = np.abs(stats.zscore(df['Fare']))
outliers = df[z_scores > 3]

print("\nNumber of Fare Outliers:", len(outliers))

# -------------------------------
# 4. ADVANCED VISUALIZATION
# -------------------------------
g = sns.FacetGrid(df, col='Survived', row='Pclass', height=3)
g.map(sns.histplot, 'Age', bins=20)
plt.show()

sns.pairplot(df[['Age', 'Fare', 'Parch', 'Survived']], hue='Survived')
plt.show()

# -------------------------------
# SAVE CLEAN DATA
# -------------------------------
df.to_csv("titanic_cleaned.csv", index=False)
# Fill missing values dynamically
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\n✅ EDA Completed Successfully!")