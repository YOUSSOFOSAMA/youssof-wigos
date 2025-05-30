# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:04:57 2025

@author: omaro
"""

import pandas as p
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

data = p.read_csv("C:\\Users\\omaro\\Desktop\\Semester 6\\bank dataset\\bank.csv")
#data.isnull().sum()

numerical = data.select_dtypes(include=['int64', 'float64']).columns
categorical = data.select_dtypes(include=['object']).columns

for col in numerical:
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
    
for col in numerical:
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
    

#Age
# Skewness BEFORE transformation
original_skew = data['age'].skew()
print(f"🔹 Skewness of 'age' (original): {original_skew:.4f}")
# Apply square root transformation
data['age'] = np.sqrt(data['age'])

# Skewness after transformation
print("Skewness after sqrt transformation:", data['age'].skew())

# Detect outliers using IQR
Q1 = data['age'].quantile(0.25)
Q3 = data['age'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['age'] < lower_bound) | (data['age'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")

# Remove outliers
data = data[(data['age'] >= lower_bound) & (data['age'] <= upper_bound)]

# Plot histogram after cleaning
sns.histplot(data['age'], kde=True)
plt.title('Age Distribution After sqrt + Outlier Removal')
plt.xlabel('Transformed Age')
plt.ylabel('Frequency')
plt.show()

# Plot boxplot after cleaning
sns.boxplot(x=data['age'])
plt.title('Boxplot of Transformed Age')
plt.xlabel('Transformed Age')
plt.show()

#Balance

def optimal_bins_fd(data):
    # Freedman-Diaconis Rule (Data-Driven Approach)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(data) ** (1/3))
    n_bins = int((data.max() - data.min()) / bin_width)
    return max(2, n_bins)

# Balance transformation and outlier removal
if (data['balance'] < 0).any():
    shift = abs(data['balance'].min()) + 1
    # Apply log1p transformation with shift
    data['balance_log'] = np.log1p(data['balance'] + shift)
    print(f"🔹 Shifted 'balance' by {shift} to apply transformations.")
else:
    data['balance_log'] = np.log1p(data['balance'])

# Show skewness results
skew_val = data['balance_log'].skew()
print(f"Skewness of 'balance_log': {skew_val:.4f}")

# Detect outliers using IQR
Q1 = data['balance_log'].quantile(0.25)
Q3 = data['balance_log'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['balance_log'] < lower_bound) | (data['balance_log'] > upper_bound)]
print(f"Number of outliers in 'balance': {len(outliers)}")

# Calculate optimal bins for histogram
bins = optimal_bins_fd(data['balance_log'])

# Plot histogram using Freedman-Diaconis bins
sns.histplot(data['balance_log'], bins=bins, kde=True)
plt.title('Balance (log transformed & cleaned)')
plt.xlabel('balance_log')
plt.ylabel('Frequency')
plt.show()

# Plot boxplot
sns.boxplot(x=data['balance_log'])
plt.title('Boxplot of Balance (log transformed)')
plt.xlabel('balance_log')
plt.show()

#Duration
# Skewness BEFORE transformation
original_skew = data['duration'].skew()
print(f"🔹 Skewness of 'duration' (original): {original_skew:.4f}")

# Apply square root transformation
data['duration_sqrt'] = np.sqrt(data['duration'])

# Skewness AFTER transformation
transformed_skew = data['duration_sqrt'].skew()
print(f"🔹 Skewness of 'duration_sqrt': {transformed_skew:.4f}")

#data['duration_log'] = np.log1p(data['duration'])
# Skewness after transformation
#print("Skewness after log1p:", data['duration_log'].skew())

# Detect outliers using IQR
Q1 = data['duration_sqrt'].quantile(0.25)
Q3 = data['duration_sqrt'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['duration_sqrt'] < lower_bound) | (data['duration_sqrt'] > upper_bound)]
print(f"🔸 Number of outliers in 'duration_sqrt': {len(outliers)}")

# Remove outliers
data = data[(data['duration_sqrt'] >= lower_bound) & (data['duration_sqrt'] <= upper_bound)]

# Plot cleaned data
sns.histplot(data['duration_sqrt'], kde=True)
plt.title('Transformed Duration (sqrt) – After Outlier Removal')
plt.xlabel('duration_sqrt')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x=data['duration_sqrt'])
plt.title('Boxplot of Duration (sqrt) – After Outlier Removal')
plt.xlabel('duration_sqrt')
plt.show()

# campaign
# Skewness BEFORE transformation
original_skew = data['campaign'].skew()
print(f"🔹 Skewness of 'campaign' (original): {original_skew:.4f}")

#  Apply square root transformation
#data['campaign_sqrt'] = np.sqrt(data['campaign'])

# Skewness AFTER transformation
#transformed_skew = data['campaign_sqrt'].skew()
#print(f"🔹 Skewness of 'campaign_sqrt': {transformed_skew:.4f}")

#data['campaign_log'] = np.log1p(data['campaign'])  # log(1 + x) to handle zero
#print("Skewness after log1p:", data['campaign_log'].skew())

pt = PowerTransformer(method='yeo-johnson')
data['campaign_yeojohnson'] = pt.fit_transform(data[['campaign']])

with open("campaign_yeojohnson_transformer.pkl", "wb") as f:
    pickle.dump(pt, f)

# Skewness after transformation
print("Skewness after Yeo-Johnson:", p.Series(data['campaign_yeojohnson']).skew())

# Detect outliers using IQR
Q1 = data['campaign_yeojohnson'].quantile(0.25)
Q3 = data['campaign_yeojohnson'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['campaign_yeojohnson'] < lower_bound) | (data['campaign_yeojohnson'] > upper_bound)]
print(f"🔸 Number of outliers in 'campaign_yeojohnson': {len(outliers)}")

# Plot after removing skew
sns.histplot(data['campaign_yeojohnson'], kde=True)
plt.title('Campaign (Yeo-Johnson) – After Outlier Removal')
plt.xlabel('campaign_yeojohnson')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x=data['campaign_yeojohnson'])
plt.title('Boxplot of Campaign (Yeo-Johnson) – After Outlier Removal')
plt.xlabel('campaign_yeojohnson')
plt.show()

# Pdays
#  Replace 999 and -1 with NaN
data['pdays_cleaned'] = data['pdays'].replace([999, -1], np.nan)

sns.histplot(data['pdays_cleaned'].dropna(), kde=True, bins=30)
plt.title('Original Distribution of pdays (excluding 999/-1)')
plt.xlabel('pdays')
plt.ylabel('Frequency')
plt.show()

# Skewness BEFORE transformation
print(f"Original Skewness (before transformation): {data['pdays_cleaned'].skew():.4f}")

# Apply square root transformation
data['pdays_sqrt'] = np.sqrt(data['pdays_cleaned'])

# Skewness after transformation
print(f"Skewness after sqrt transformation: {data['pdays_sqrt'].skew():.4f}")

# Detect outliers using IQR
Q1 = data['pdays_sqrt'].quantile(0.25)
Q3 = data['pdays_sqrt'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['pdays_sqrt'] < lower_bound) | (data['pdays_sqrt'] > upper_bound)]
print(f"Number of outliers: {len(outliers)}")

# Remove outliers
data = data[(data['pdays_sqrt'] >= lower_bound) & (data['pdays_sqrt'] <= upper_bound)]

# Plot after cleaning
sns.histplot(data['pdays_sqrt'], kde=True, bins=30)
plt.title('pdays (sqrt-transformed) – After Outlier Removal')
plt.xlabel('sqrt(pdays)')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x=data['pdays_sqrt'])
plt.title('Boxplot of pdays (sqrt) – After Outlier Removal')
plt.xlabel('sqrt(pdays)')
plt.show()

# Previous
# Skewness BEFORE transformation
print(f"🔹 Original Skewness: {data['previous'].skew():.4f}")

# Apply sqrt transformation
#data['previous_sqrt'] = np.sqrt(data['previous'])

# Skewness after transformation
#print(f"🔹 Skewness after sqrt transformation: {data['previous_sqrt'].skew():.4f}")

#data['previous_log'] = np.log1p(data['previous'])
#print("🔹 Skewness after log1p transformation:", data['previous_log'].skew())

pt = PowerTransformer(method='yeo-johnson')
data['previous_yeojohnson'] = pt.fit_transform(data[['previous']])
# Skewness after transformation
print("🔹 Skewness after Yeo-Johnson transformation:", p.Series(data['previous_yeojohnson']).skew())

# Detect Outliers using IQR 
Q1 = data['previous_yeojohnson'].quantile(0.25)
Q3 = data['previous_yeojohnson'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data['previous_yeojohnson'] < lower_bound) | (data['previous_yeojohnson'] > upper_bound)]
print(f"🔸 Number of outliers: {len(outliers)}")

# 5. Remove outliers
data = data[(data['previous_yeojohnson'] >= lower_bound) & (data['previous_yeojohnson'] <= upper_bound)]

# Plot after cleaning
sns.histplot(data['previous_yeojohnson'], kde=True, bins=30)
plt.title('Previous (yeojohnson-transformed) – After Outlier Removal')
plt.xlabel('yeojohnson(previous)')
plt.ylabel('Frequency')
plt.show()

sns.boxplot(x=data['previous_yeojohnson'])
plt.title('Boxplot of Previous (yeojohnson) – After Outlier Removal')
plt.xlabel('yeojohnson(previous)')
plt.show()

final_features = [
    'age',  
    'job',       
    'marital', 
    'education',
    'default',            
    'balance_log', 
    'housing',
    'loan' ,
    'contact',
    'day',      
    'month',
    'duration_sqrt',            
    'campaign_yeojohnson',     
    'pdays_sqrt',           
    'previous_yeojohnson',     
    'poutcome',
]

X = data[final_features].copy()
y = data['deposit'].apply(lambda x: 1 if x == 'yes' else 0)

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numeric_cols = ['age', 'balance_log', 'day', 'duration_sqrt', 'campaign_yeojohnson', 'pdays_sqrt', 'previous_yeojohnson']

#One hot encoder
#X_encoded = p.get_dummies(X[categorical_cols], drop_first=True)
#X_encoded_cat = p.concat([X[numeric_cols], X_encoded], axis=1)

# Label encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_results = p.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Visualize top 10 features
mi_results.head(10).plot(kind='barh')
plt.title("Top 10 Features by Mutual Information")
plt.xlabel("MI Score")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.show()

threshold = 0.01  # Adjust this as needed
selected_features = mi_results[mi_results > threshold].index.tolist()
X_selected = X[selected_features]

print(f"✅ Features with MI > {threshold}: {len(selected_features)} kept")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_pred_train_lr = lr_model.predict(X_train)
y_pred_test_lr = lr_model.predict(X_test)

print("🔹 Logistic Regression Results:")
print("Train Accuracy:", accuracy_score(y_train, y_pred_train_lr))
print("Test Accuracy:", accuracy_score(y_test, y_pred_test_lr))
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test_lr))

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_test_lr)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No Deposit", "Deposit"],
            yticklabels=["No Deposit", "Deposit"])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save Logistic Regression model
with open("lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# Save label encoders dictionary
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
    
with open("selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)
