#!/usr/bin/env python
# coding: utf-8

# In[16]:


##Extreme gradient boosting final
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance, to_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import os
import graphviz  # Ensure graphviz is installed and accessible

# Load the data from the Excel file
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist.")

# Attempt to read the Excel file
try:
    data = pd.read_excel(file_path)
except PermissionError:
    raise PermissionError(f"Permission denied for file at path {file_path}. Please close the file if it's open in another application.")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# If there are missing values, fill them with the mean of the column
data = data.fillna(data.mean())

# Separate features (X) and target (y)
# Assuming the rightmost 4 columns are the targets (diseases)
X = data.iloc[:, :-4]
y = data.iloc[:, -4:]

# One-Hot Encode categorical variables (if any)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifiers
model1 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model4 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the models
print("Fitting the models...")
model1.fit(X_train, y_train.iloc[:, 0])
model2.fit(X_train, y_train.iloc[:, 1])
model3.fit(X_train, y_train.iloc[:, 2])
model4.fit(X_train, y_train.iloc[:, 3])
print("Model fitting complete.")

# Predict and evaluate the models
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)

# Accuracy
accuracy1 = accuracy_score(y_test.iloc[:, 0], y_pred1)
accuracy2 = accuracy_score(y_test.iloc[:, 1], y_pred2)
accuracy3 = accuracy_score(y_test.iloc[:, 2], y_pred3)
accuracy4 = accuracy_score(y_test.iloc[:, 3], y_pred4)

# Specificity function
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn = cm[0, 0]
        fp = cm[0, 1]
        return tn / (tn + fp)
    else:
        return None

# Specificity
specificity1 = calculate_specificity(y_test.iloc[:, 0], y_pred1)
specificity2 = calculate_specificity(y_test.iloc[:, 1], y_pred2)
specificity3 = calculate_specificity(y_test.iloc[:, 2], y_pred3)
specificity4 = calculate_specificity(y_test.iloc[:, 3], y_pred4)

# Print results
print("Evaluation for Jaundice:")
print(f"Accuracy: {accuracy1:.2f}")
print(f"Specificity: {specificity1 if specificity1 is not None else 'Not applicable'}")
print("Classification Report:\n", classification_report(y_test.iloc[:, 0], y_pred1, zero_division=1))

print("Evaluation for Hemolytic Jaundice:")
print(f"Accuracy: {accuracy2:.2f}")
print(f"Specificity: {specificity2 if specificity2 is not None else 'Not applicable'}")
print("Classification Report:\n", classification_report(y_test.iloc[:, 1], y_pred2, zero_division=1))

print("Evaluation for Hepatic Jaundice:")
print(f"Accuracy: {accuracy3:.2f}")
print(f"Specificity: {specificity3 if specificity3 is not None else 'Not applicable'}")
print("Classification Report:\n", classification_report(y_test.iloc[:, 2], y_pred3, zero_division=1))

print("Evaluation for Obstructive Jaundice:")
print(f"Accuracy: {accuracy4:.2f}")
print(f"Specificity: {specificity4 if specificity4 is not None else 'Not applicable'}")
print("Classification Report:\n", classification_report(y_test.iloc[:, 3], y_pred4, zero_division=1))

# Plot feature importances
plt.figure(figsize=(10, 8))
plot_importance(model1)
plt.title('Feature Importance for Jaundice')
plt.show()

plt.figure(figsize=(10, 8))
plot_importance(model2)
plt.title('Feature Importance for Hemolytic Jaundice')
plt.show()

plt.figure(figsize=(10, 8))
plot_importance(model3)
plt.title('Feature Importance for Hepatic Jaundice')
plt.show()

plt.figure(figsize=(10, 8))
plot_importance(model4)
plt.title('Feature Importance for Obstructive Jaundice')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Box Plot for each feature
plt.figure(figsize=(15, 10))
sns.boxplot(data=data.iloc[:, :-4])
plt.title('Box Plot for Each Feature')
plt.xticks(rotation=90)
plt.show()

# Confusion matrix and classification report for Disease 1
cm1 = confusion_matrix(y_test.iloc[:, 0], y_pred1)
print("Confusion Matrix for Jaundice:")
print(cm1)

plt.figure(figsize=(6, 4))
sns.heatmap(cm1, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Jaundice')
plt.show()

# Confusion matrix and classification report for Disease 2
cm2 = confusion_matrix(y_test.iloc[:, 1], y_pred2)
print("Confusion Matrix for Hemolytic Jaundice:")
print(cm2)

plt.figure(figsize=(6, 4))
sns.heatmap(cm2, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Hemolytic Jaundice')
plt.show()

# Confusion matrix and classification report for Disease 3 (if applicable)
cm3 = confusion_matrix(y_test.iloc[:, 2], y_pred3)
print("Confusion Matrix for Hepatic Jaundice:")
print(cm3)

plt.figure(figsize=(6, 4))
sns.heatmap(cm3, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Hepatic Jaundice')
plt.show()

# Confusion matrix and classification report for Disease 4 (if applicable)
cm4 = confusion_matrix(y_test.iloc[:, 3], y_pred4)
print("Confusion Matrix for Obstructive Jaundice:")
print(cm4)

plt.figure(figsize=(6, 4))
sns.heatmap(cm4, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Obstructive Jaundice')
plt.show()


# In[6]:


#Gradient extended
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance, to_graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data from the Excel file
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist.")

# Attempt to read the Excel file
try:
    data = pd.read_excel(file_path)
except PermissionError:
    raise PermissionError(f"Permission denied for file at path {file_path}. Please close the file if it's open in another application.")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# If there are missing values, fill them with the mean of the column
data = data.fillna(data.mean())

# Separate features (X) and target (y)
# Assuming the rightmost 4 columns are the targets (diseases)
X = data.iloc[:, :-4]
y = data.iloc[:, -4:]

# One-Hot Encode categorical variables (if any)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifiers
model1 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model4 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the models
print("Fitting the models...")
model1.fit(X_train, y_train.iloc[:, 0])
model2.fit(X_train, y_train.iloc[:, 1])
model3.fit(X_train, y_train.iloc[:, 2])
model4.fit(X_train, y_train.iloc[:, 3])
print("Model fitting complete.")

# Predict and evaluate the models
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)

# Accuracy
accuracy1 = accuracy_score(y_test.iloc[:, 0], y_pred1)
accuracy2 = accuracy_score(y_test.iloc[:, 1], y_pred2)
accuracy3 = accuracy_score(y_test.iloc[:, 2], y_pred3)
accuracy4 = accuracy_score(y_test.iloc[:, 3], y_pred4)

# Specificity function
def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn = cm[0, 0]
        fp = cm[0, 1]
        return tn / (tn + fp)
    else:
        return None

# Specificity
specificity1 = calculate_specificity(y_test.iloc[:, 0], y_pred1)
specificity2 = calculate_specificity(y_test.iloc[:, 1], y_pred2)
specificity3 = calculate_specificity(y_test.iloc[:, 2], y_pred3)
specificity4 = calculate_specificity(y_test.iloc[:, 3], y_pred4)

# Print results
print("Evaluation for Jaundice:")
print(f"Accuracy: {accuracy1:.2f}")
print(f"Specificity: {specificity1 if specificity1 is not None else 'Not applicable'}")
print("Classification Report:\n", classification_report(y_test.iloc[:, 0], y_pred1, zero_division=1))

print("Evaluation for Hemolytic Jaundice:")
print(f"Accuracy: {accuracy2:.2f}")
print(f"Specificity: {specificity2 if specificity2 is not None else 'Not applicable'}")
print("Classification Report:\n", classification_report(y_test.iloc[:, 1], y_pred2, zero_division=1))

print("Evaluation for Hepatic Jaundice:")
print(f"Accuracy: {accuracy3:.2f}")
print(f"Specificity: {specificity3 if specificity3 is not None else 'Not applicable'}")
print("Classification Report:\n", classification_report(y_test.iloc[:, 2], y_pred3, zero_division=1))

print("Evaluation for Obstructive Jaundice:")
print(f"Accuracy: {accuracy4:.2f}")
print(f"Specificity: {specificity4 if specificity4 is not None else 'Not applicable'}")
print("Classification Report:\n", classification_report(y_test.iloc[:, 3], y_pred4, zero_division=1))

# Visualize XGBoost trees using graphviz
plt.figure(figsize=(20, 20))

# Tree for Jaundice
graph1 = to_graphviz(model1, num_trees=0, rankdir='LR')
graph1.format = 'png'
graph1.render('tree1', format='png', cleanup=True)
graph1.render(view=True)

# Tree for Hemolytic Jaundice
graph2 = to_graphviz(model2, num_trees=0, rankdir='LR')
graph2.format = 'png'
graph2.render('tree2', format='png', cleanup=True)
graph2.render(view=True)

# Tree for Hepatic Jaundice
graph3 = to_graphviz(model3, num_trees=0, rankdir='LR')
graph3.format = 'png'
graph3.render('tree3', format='png', cleanup=True)
graph3.render(view=True)

# Tree for Obstructive Jaundice
graph4 = to_graphviz(model4, num_trees=0, rankdir='LR')
graph4.format = 'png'
graph4.render('tree4', format='png', cleanup=True)
graph4.render(view=True)

plt.show()


# In[21]:


#Gaussian
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data from the Excel file
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist.")

# Attempt to read the Excel file
try:
    data = pd.read_excel(file_path)
except PermissionError:
    raise PermissionError(f"Permission denied for file at path {file_path}. Please close the file if it's open in another application.")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# If there are missing values, fill them with the mean of the column
data = data.fillna(data.mean())

# Separate features (X) and target (y)
# Assuming the rightmost 4 columns are the targets (diseases)
X = data.iloc[:, :-4]
y = data.iloc[:, -4:]

# One-Hot Encode categorical variables (if any)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
gnb1 = GaussianNB()
gnb2 = GaussianNB()
gnb3 = GaussianNB()
gnb4 = GaussianNB()

# Train the models
print("Fitting the models...")
gnb1.fit(X_train, y_train.iloc[:, 0])
gnb2.fit(X_train, y_train.iloc[:, 1])
gnb3.fit(X_train, y_train.iloc[:, 2])
gnb4.fit(X_train, y_train.iloc[:, 3])
print("Model fitting complete.")

# Predict probabilities
y_prob1 = gnb1.predict_proba(X_test)[:, 1]
y_prob2 = gnb2.predict_proba(X_test)[:, 1]
y_prob3 = gnb3.predict_proba(X_test)[:, 1]
y_prob4 = gnb4.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr1, tpr1, _ = roc_curve(y_test.iloc[:, 0], y_prob1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(y_test.iloc[:, 1], y_prob2)
roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, _ = roc_curve(y_test.iloc[:, 2], y_prob3)
roc_auc3 = auc(fpr3, tpr3)

fpr4, tpr4, _ = roc_curve(y_test.iloc[:, 3], y_prob4)
roc_auc4 = auc(fpr4, tpr4)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'ROC curve (area = {roc_auc1:.2f}) - Jaundice')
plt.plot(fpr2, tpr2, color='green', lw=2, label=f'ROC curve (area = {roc_auc2:.2f}) - Hemolytic Jaundice')
plt.plot(fpr3, tpr3, color='red', lw=2, label=f'ROC curve (area = {roc_auc3:.2f}) - Hepatic Jaundice')
plt.plot(fpr4, tpr4, color='purple', lw=2, label=f'ROC curve (area = {roc_auc4:.2f}) - Obstructive Jaundice')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print results
print("Evaluation for Jaundice:")
print("Classification Report:\n", classification_report(y_test.iloc[:, 0], gnb1.predict(X_test), zero_division=1))

print("Evaluation for Hemolytic Jaundice:")
print("Classification Report:\n", classification_report(y_test.iloc[:, 1], gnb2.predict(X_test), zero_division=1))

print("Evaluation for Hepatic Jaundice:")
print("Classification Report:\n", classification_report(y_test.iloc[:, 2], gnb3.predict(X_test), zero_division=1))

print("Evaluation for Obstructive Jaundice:")
print("Classification Report:\n", classification_report(y_test.iloc[:, 3], gnb4.predict(X_test), zero_division=1))


# In[22]:


#Gaussian Naive Bayes Final
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the data from the Excel file
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist.")

# Attempt to read the Excel file
try:
    data = pd.read_excel(file_path)
except PermissionError:
    raise PermissionError(f"Permission denied for file at path {file_path}. Please close the file if it's open in another application.")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# If there are missing values, fill them with the mean of the column
data = data.fillna(data.mean())

# Separate features (X) and target (y)
# Assuming the rightmost 4 columns are the targets (diseases)
X = data.iloc[:, :-4]
y = data.iloc[:, -4:]

# One-Hot Encode categorical variables (if any)
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
gnb1 = GaussianNB()
gnb2 = GaussianNB()
gnb3 = GaussianNB()
gnb4 = GaussianNB()

# Train the models
print("Fitting the models...")
gnb1.fit(X_train, y_train.iloc[:, 0])
gnb2.fit(X_train, y_train.iloc[:, 1])
gnb3.fit(X_train, y_train.iloc[:, 2])
gnb4.fit(X_train, y_train.iloc[:, 3])
print("Model fitting complete.")

# Predict probabilities
y_prob1 = gnb1.predict_proba(X_test)[:, 1]
y_prob2 = gnb2.predict_proba(X_test)[:, 1]
y_prob3 = gnb3.predict_proba(X_test)[:, 1]
y_prob4 = gnb4.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr1, tpr1, _ = roc_curve(y_test.iloc[:, 0], y_prob1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(y_test.iloc[:, 1], y_prob2)
roc_auc2 = auc(fpr2, tpr2)

fpr3, tpr3, _ = roc_curve(y_test.iloc[:, 2], y_prob3)
roc_auc3 = auc(fpr3, tpr3)

fpr4, tpr4, _ = roc_curve(y_test.iloc[:, 3], y_prob4)
roc_auc4 = auc(fpr4, tpr4)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'ROC curve (area = {roc_auc1:.2f}) - Jaundice')
plt.plot(fpr2, tpr2, color='green', lw=2, label=f'ROC curve (area = {roc_auc2:.2f}) - Hemolytic Jaundice')
plt.plot(fpr3, tpr3, color='red', lw=2, label=f'ROC curve (area = {roc_auc3:.2f}) - Hepatic Jaundice')
plt.plot(fpr4, tpr4, color='purple', lw=2, label=f'ROC curve (area = {roc_auc4:.2f}) - Obstructive Jaundice')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print results
print("Evaluation for Jaundice:")
print("Classification Report:\n", classification_report(y_test.iloc[:, 0], gnb1.predict(X_test), zero_division=1))

print("Evaluation for Hemolytic Jaundice:")
print("Classification Report:\n", classification_report(y_test.iloc[:, 1], gnb2.predict(X_test), zero_division=1))

print("Evaluation for Hepatic Jaundice:")
print("Classification Report:\n", classification_report(y_test.iloc[:, 2], gnb3.predict(X_test), zero_division=1))

print("Evaluation for Obstructive Jaundice:")
print("Classification Report:\n", classification_report(y_test.iloc[:, 3], gnb4.predict(X_test), zero_division=1))

# Plot confusion matrix for each disease
plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
cm1 = confusion_matrix(y_test.iloc[:, 0], gnb1.predict(X_test))
sns.heatmap(cm1, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Jaundice')

plt.subplot(2, 2, 2)
cm2 = confusion_matrix(y_test.iloc[:, 1], gnb2.predict(X_test))
sns.heatmap(cm2, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Hemolytic Jaundice')

plt.subplot(2, 2, 3)
cm3 = confusion_matrix(y_test.iloc[:, 2], gnb3.predict(X_test))
sns.heatmap(cm3, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Hepatic Jaundice')

plt.subplot(2, 2, 4)
cm4 = confusion_matrix(y_test.iloc[:, 3], gnb4.predict(X_test))
sns.heatmap(cm4, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Obstructive Jaundice')

plt.tight_layout()
plt.show()

# Plot feature importance for each disease
plt.figure(figsize=(15, 12))

plt.subplot(2, 2, 1)
feature_importances1 = np.abs(gnb1.theta_[0])  # Mean of each feature for class 0
sorted_idx1 = np.argsort(feature_importances1)[::-1]
sns.barplot(x=feature_importances1[sorted_idx1], y=X.columns[sorted_idx1])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Jaundice')

plt.subplot(2, 2, 2)
feature_importances2 = np.abs(gnb2.theta_[0])  # Mean of each feature for class 0
sorted_idx2 = np.argsort(feature_importances2)[::-1]
sns.barplot(x=feature_importances2[sorted_idx2], y=X.columns[sorted_idx2])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Hemolytic Jaundice')

plt.subplot(2, 2, 3)
feature_importances3 = np.abs(gnb3.theta_[0])  # Mean of each feature for class 0
sorted_idx3 = np.argsort(feature_importances3)[::-1]
sns.barplot(x=feature_importances3[sorted_idx3], y=X.columns[sorted_idx3])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Hepatic Jaundice')

plt.subplot(2, 2, 4)
feature_importances4 = np.abs(gnb4.theta_[0])  # Mean of each feature for class 0
sorted_idx4 = np.argsort(feature_importances4)[::-1]
sns.barplot(x=feature_importances4[sorted_idx4], y=X.columns[sorted_idx4])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance - Obstructive Jaundice')

plt.tight_layout()
plt.show()


# In[10]:


get_ipython().system('pip install --upgrade xgboost')
get_ipython().system('pip install --upgrade xgboost graphviz')


# In[22]:


#Decision Tree Final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Load and preprocess data
data = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Check column names in the DataFrame
print("Columns in the DataFrame:")
print(data.columns)

# Assuming these are the target variables (diseases), verify exact names
target_columns = ['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice']

# Verify each target column exists in the DataFrame
for col in target_columns:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in the DataFrame. Please check your column names or adjust your code accordingly.")

# Separate features (X) and targets (y)
X = data.drop(columns=target_columns)
y = data[target_columns]

# Step 2: Apply SMOTE to each disease column separately
smote = SMOTE(random_state=42)
data_resampled = []

for i in range(y.shape[1]):
    X_resampled_i, y_resampled_i = smote.fit_resample(X, y.iloc[:, i])
    data_resampled.append((X_resampled_i, y_resampled_i, target_columns[i]))

# Step 3: Split resampled data into training and testing sets, train Decision Tree models, and evaluate
dt_models = []

for X_resampled_i, y_resampled_i, target_col in data_resampled:
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_resampled_i, y_resampled_i, test_size=0.2, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_i, y_train_i)
    dt_models.append((dt, target_col))  # Store model and corresponding disease name
    
    # Evaluate the model
    y_pred_i = dt.predict(X_test_i)
    
    print(f"Classification Report for {target_col}:")
    print(classification_report(y_test_i, y_pred_i))
    
    print(f"\nConfusion Matrix for {target_col}:")
    print(confusion_matrix(y_test_i, y_pred_i))
    print("\n")
    
    # Visualize feature importances
    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X.columns[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for {target_col}")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=90)
    plt.show()

    # Visualize Decision Tree
    plt.figure(figsize=(15, 10))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['0', '1'])
    plt.title(f"Decision Tree for {target_col}")
    plt.show()

# Step 4: Top Feature Importances
plt.figure(figsize=(10, 6))
importances_all = np.mean([dt.feature_importances_ for dt, _ in dt_models], axis=0)
indices_all = np.argsort(importances_all)[::-1][:10]  # Top 10 features
plt.barh(range(10), importances_all[indices_all], align='center')
plt.yticks(range(10), [X.columns[i] for i in indices_all])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances across all Diseases')
plt.show()


# In[2]:


#Logistic Regrssion Final
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Split data into features and targets
X = df.iloc[:, :-4]
y = df.iloc[:, -4:]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE with adjusted k_neighbors
smote = SMOTE(random_state=42, k_neighbors=1)

X_resampled_list = []
y_resampled_list = []

for i in range(y.shape[1]):
    X_res, y_res = smote.fit_resample(X_scaled, y.iloc[:, i])
    X_resampled_list.append(X_res)
    y_resampled_list.append(y_res)

# Split data into training and testing sets
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

for X_res, y_res in zip(X_resampled_list, y_resampled_list):
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    X_train_list.append(X_train)
    X_test_list.append(X_test)
    y_train_list.append(y_train)
    y_test_list.append(y_test)

# Train and evaluate Logistic Regression for each disease
for i, disease_name in enumerate(['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice']):
    logreg_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    logreg_model.fit(X_train_list[i], y_train_list[i])

    y_pred = logreg_model.predict(X_test_list[i])
    y_prob = logreg_model.predict_proba(X_test_list[i])[:, 1]

    accuracy = accuracy_score(y_test_list[i], y_pred)
    roc_auc = roc_auc_score(y_test_list[i], y_prob)
    fpr, tpr, _ = roc_curve(y_test_list[i], y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test_list[i], y_pred).ravel()
    specificity = tn / (tn + fp)

    print(f"Evaluation for {disease_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print("Classification Report:")
    print(classification_report(y_test_list[i], y_pred))
    print(f"Confusion Matrix for {disease_name}:")
    print(confusion_matrix(y_test_list[i], y_pred))
    print("\n")

    # Visualize feature importances
    feature_importances = np.abs(logreg_model.coef_[0])
    feature_names = X.columns
    sorted_idx = np.argsort(feature_importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances[sorted_idx], y=feature_names[sorted_idx])
    plt.title(f'Feature Importances for {disease_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    # Save the plot as an image file
    plt.savefig(f'{disease_name}_feature_importances.png')
    plt.show()

    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {disease_name}')
    plt.legend(loc='lower right')
    
    # Save the plot as an image file
    plt.savefig(f'{disease_name}_roc_curve.png')
    plt.show()

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(confusion_matrix(y_test_list[i], y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {disease_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the plot as an image file
    plt.savefig(f'{disease_name}_confusion_matrix.png')
    plt.show()


# In[40]:


#Random Forest Final
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz

# Set the path to Graphviz executables
os.environ["PATH"] += os.pathsep + 'D:\\windows_10_cmake_Release_Graphviz-11.0.0-win64\\Graphviz-11.0.0-win64\\bin'

# Load your dataset
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'
data = pd.read_excel(file_path)

# Check column names to identify any issues
print("Columns in the dataset:", data.columns)

# Update target columns based on actual column names in your dataset
target_columns = ['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice']

# Check if target columns are present
for column in target_columns:
    if column not in data.columns:
        print(f"Error: Column '{column}' not found in the dataset.")

# Ensure 'Age' column exists before creating 'Age Category'
if 'Age' in data.columns:
    data['Age Category'] = data['Age'].apply(lambda x: 'below 45' if x < 45 else 'above 45')

    # Function to plot feature importance
    def plot_feature_importance(model, feature_names, title):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f'Feature Importance for {title}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()

    # Function to visualize a single tree from the Random Forest
    def visualize_tree(model, feature_names, target, tree_index=0):
        estimator = model.estimators_[tree_index]

        dot_data = export_graphviz(estimator, out_file=None, 
                                   feature_names=feature_names,  
                                   class_names=[str(class_name) for class_name in model.classes_],  
                                   filled=True, rounded=True,  
                                   special_characters=True)  
        graph = graphviz.Source(dot_data)  
        graph.render(f"random_forest_tree_{target.replace(' ', '_')}")
        graph.view()

    # Separate features (X) and targets (y)
    X = data.drop(columns=target_columns + ['Age Category'])  # Exclude non-numeric column 'Age Category'
    feature_names = X.columns

    # Train and evaluate models for each target
    for target in target_columns:
        y = data[target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest model
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(rf_model, X, y, cv=5)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Evaluation for {target} using Random Forest:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Mean CV Score: {cv_scores.mean():.2f}")
        print("Classification Report:")
        print(classification_rep)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\n")

        # Plot feature importance
        plot_feature_importance(rf_model, feature_names, target)

        # Visualize a single tree from the Random Forest
        visualize_tree(rf_model, feature_names, target, tree_index=0)
else:
    print("Error: 'Age' column not found in the dataset.")



# In[38]:


import graphviz
from sklearn.tree import export_graphviz

def visualize_tree(model, feature_names, target, tree_index=0):
    """
    Visualizes a single decision tree from a Random Forest model using Graphviz.
    
    Parameters:
    - model: The trained Random Forest model
    - feature_names: List of feature names (column names of the input data)
    - target: Name of the target variable (class label)
    - tree_index: Index of the tree in the Random Forest to visualize (default is 0)
    """
    # Retrieve the decision tree from the ensemble
    estimator = model.estimators_[tree_index]

    # Export the decision tree as a Graphviz DOT format string
    dot_data = export_graphviz(estimator, out_file=None,
                               feature_names=feature_names,
                               class_names=[str(class_name) for class_name in model.classes_],
                               filled=True, rounded=True,
                               special_characters=True)

    # Create a Graphviz Source object from the DOT data
    graph = graphviz.Source(dot_data)

    # Render the decision tree
    # Save the rendered tree as a PDF file
    output_file = f"random_forest_tree_{target.replace(' ', '_')}.pdf"
    graph.render(output_file, view=True)

    # Optionally, save the rendered tree as PNG
    # graph.render(f"random_forest_tree_{target.replace(' ', '_')}.png", format='png')

    # Display the rendered tree
    # graph.view()

# Example usage:
# visualize_tree(rf_model, feature_names, 'TargetClass')


# In[34]:


#Hepatitis extended
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data from Excel (replace with your actual file path)
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'
df = pd.read_excel(file_path)

# Separate Features and Target
X = df.iloc[:, :-4]  # Features (all columns except the last four)
y = df.iloc[:, -4:]  # Target (last four columns)

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'XGBoost': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB()
}

# Dictionary to store cross-validation scores
cv_scores = {}

# Train and evaluate models for each disease separately
for disease in y_train.columns:
    print(f"Evaluating models for disease: {disease}\n")
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train[disease])
        
        # Cross-validation
        cv_score = cross_val_score(model, X, y[disease], cv=5, scoring='accuracy').mean()
        cv_scores[model_name] = cv_score

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test[disease], y_pred)

        # Print evaluation metrics
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Mean Cross-Validation Score: {cv_score:.3f}")
        print(f"Classification Report for {model_name} - {disease}:")
        print(classification_report(y_test[disease], y_pred))
        print(f"Confusion Matrix for {model_name} - {disease}:")
        print(confusion_matrix(y_test[disease], y_pred))
        print("\n------------------------------------------\n")

    print(f"================================================================================\n")

# Create a DataFrame for mean cross-validation scores
cv_scores_df = pd.DataFrame(list(cv_scores.items()), columns=['Model', 'Mean CV Score'])

# Plotting
plt.figure(figsize=(8, 10))
sns.barplot(x='Model', y='Mean CV Score', data=cv_scores_df, palette='viridis')
plt.title('Mean Cross-Validation Score for Different Models')
plt.xlabel('Model')
plt.ylabel('Mean CV Score')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for clarity
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data from Excel (replace with your actual file path)
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'
df = pd.read_excel(file_path)

# Separate Features and Target
X = df.iloc[:, :-4]  # Features (all columns except the last four)
y = df.iloc[:, -4:]  # Target (last four columns)

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'XGBoost': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB()
}

# Dictionary to store cross-validation scores
cv_scores = {}

# Train and evaluate models for each disease separately
for disease in y_train.columns:
    print(f"Evaluating models for disease: {disease}\n")
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train[disease])
        
        # Cross-validation
        cv_score = cross_val_score(model, X, y[disease], cv=5, scoring='accuracy').mean()
        cv_scores[model_name] = cv_score

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test[disease], y_pred)

        # Print evaluation metrics
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Mean Cross-Validation Score: {cv_score:.3f}")
        print(f"Classification Report for {model_name} - {disease}:")
        print(classification_report(y_test[disease], y_pred))
        print(f"Confusion Matrix for {model_name} - {disease}:")
        print(confusion_matrix(y_test[disease], y_pred))
        print("\n------------------------------------------\n")

    print(f"================================================================================\n")

# Create a DataFrame for mean cross-validation scores
cv_scores_df = pd.DataFrame(list(cv_scores.items()), columns=['Model', 'Mean CV Score'])

# Plotting
plt.figure(figsize=(8, 10))
sns.barplot(x='Model', y='Mean CV Score', data=cv_scores_df, palette='viridis')
plt.title('Mean Cross-Validation Score for Different Models')
plt.xlabel('Model')
plt.ylabel('Mean CV Score')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for clarity
plt.xticks(rotation=45, fontsize=14)  # Rotate x-axis labels for better visibility and increase font size
plt.tight_layout()
plt.show()


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Separate features and target variables
X = df.iloc[:, :-4]  # All columns except the last four
y_jaundice = df.iloc[:, -4]  # Fouth last column
y_hemolytic = df.iloc[:, -3]  # Third last column
y_hepatic = df.iloc[:, -2]  # Second last column
y_obstructive = df.iloc[:, -1]  # last column


# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X, y, disease_name):
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_res, y_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results


# Train and evaluate models for each disease
results_jaundice = train_and_evaluate(X, y_jaundice, 'Jaundice')
results_hemolytic = train_and_evaluate(X, y_hemolytic, 'Hemolytic Jaundice')
results_hepatic = train_and_evaluate(X, y_hepatic, 'Hepatic Jaundice')
results_obstructive = train_and_evaluate(X, y_obstructive, 'Obstructive Jaundice')

# Plotting the results
def plot_results(results, disease_name, metric):
    labels = list(results.keys())
    scores = [results[model][metric] for model in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars = ax.bar(x, scores, width)

    ax.set_xlabel('Models')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.bar_label(bars, padding=3)

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Plot accuracy, precision, and recall for each disease
for disease_name, results in zip(['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice'],
                                 [results_jaundice, results_hemolytic, results_hepatic, results_obstructive]):
    plot_results(results, disease_name, 'accuracy')
    plot_results(results, disease_name, 'precision')
    plot_results(results, disease_name, 'recall')

# Plotting misclassification and correct classification
def plot_classification_summary(results, disease_name):
    labels = list(results.keys())
    metrics = ['Correctly Classified', 'Misclassified No Disease', 'Misclassified Disease']
    
    summary = {label: {metric: 0 for metric in metrics} for label in labels}
    
    for label in labels:
        y_test = results[label]['y_test']
        y_pred = results[label]['y_pred']
        confusion = confusion_matrix(y_test, y_pred)
        
        summary[label]['Correctly Classified'] = np.sum(np.diag(confusion))
        summary[label]['Misclassified No Disease'] = confusion[0, 1]
        summary[label]['Misclassified Disease'] = confusion[1, 0]

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [summary[label][metric] for label in labels]
        ax.bar(x + i*width, values, width, label=metric)

    ax.set_xlabel('Models')
    ax.set_ylabel('Counts')
    ax.set_title(f'Classification Summary for {disease_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Plot classification summaries for each disease
for disease_name, results in zip(['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice'],
                                 [results_jaundice, results_hemolytic, results_hepatic, results_obstructive]):
    plot_classification_summary(results, disease_name)

# Plotting ROC curves
def plot_roc_curves(results, disease_name):
    fig, ax = plt.subplots()

    for name, result in results.items():
        y_test = result['y_test']
        X_test = result['X_test']
        y_pred_proba = result['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic for {disease_name}')
    ax.legend(loc='lower right')

    plt.show()

# Plot ROC curves for each disease
for disease_name, results in zip(['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice'],
                                 [results_jaundice, results_hemolytic, results_hepatic, results_obstructive]):
    plot_roc_curves(results, disease_name)


# In[18]:


def plot_classification_summary(results, disease_name):
    labels = list(results.keys())
    metrics = ['Correctly Classified', 'Misclassified No Disease', 'Misclassified Disease']
    
    summary = {label: {metric: 0 for metric in metrics} for label in labels}
    
    for label in labels:
        y_test = results[label]['y_test']
        y_pred = results[label]['y_pred']
        confusion = confusion_matrix(y_test, y_pred)
        
        summary[label]['Correctly Classified'] = np.sum(np.diag(confusion))
        summary[label]['Misclassified No Disease'] = confusion[0, 1]
        summary[label]['Misclassified Disease'] = confusion[1, 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [summary[label][metric] for label in labels]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel('Models')
    ax.set_ylabel('Counts')
    ax.set_title(f'Classification Summary for {disease_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.xticks(rotation=45)
    plt.show()

# Plot classification summaries for each disease
for disease_name, results in zip(['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice'],
                                 [results_jaundice, results_hemolytic, results_hepatic, results_obstructive]):
    plot_classification_summary(results, disease_name)


# In[7]:


import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Separate features and target variables
X = df.iloc[:, :-4]  # All columns except the last four
y_jaundice = df.iloc[:, -4]  # Fouth last column
y_hemolytic = df.iloc[:, -3]  # Third last column
y_hepatic = df.iloc[:, -2]  # Second last column
y_obstructive = df.iloc[:, -1]  # last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_jaundice, X_test_jaundice, y_train_jaundice, y_test_jaundice = train_test_split(X, y_jaundice, test_size=0.2, random_state=42, stratify=y_jaundice)
X_train_hemolytic, X_test_hemolytic, y_train_hemolytic, y_test_hemolytic = train_test_split(X, y_hemolytic, test_size=0.2, random_state=42, stratify=y_hemolytic)
X_train_hepatic, X_test_hepatic, y_train_hepatic, y_test_hepatic = train_test_split(X, y_hepatic, test_size=0.2, random_state=42, stratify=y_hepatic)
X_train_obstructive, X_test_obstructive, y_train_obstructive, y_test_obstructive = train_test_split(X, y_obstructive, test_size=0.2, random_state=42, stratify=y_obstructive)

# Print train-test splits
print(f"Train-test split for Jaundice: X_train shape: {X_train_jaundice.shape}, X_test shape: {X_test_jaundice.shape}, y_train support: {len(y_train_jaundice)}, y_test support: {len(y_test_jaundice)}")
print(f"Train-test split for Hemolytic Disorders: X_train shape: {X_train_hemolytic.shape}, X_test shape: {X_test_hemolytic.shape}, y_train support: {len(y_train_hemolytic)}, y_test support: {len(y_test_hemolytic)}")
print(f"Train-test split for Hepatic Diseases: X_train shape: {X_train_hepatic.shape}, X_test shape: {X_test_hepatic.shape}, y_train support: {len(y_train_hepatic)}, y_test support: {len(y_test_hepatic)}")
print(f"Train-test split for Obstructive Disorders: X_train shape: {X_train_obstructive.shape}, X_test shape: {X_test_obstructive.shape}, y_train support: {len(y_train_obstructive)}, y_test support: {len(y_test_obstructive)}")

# SMOTE for handling class imbalance after the split
smote = SMOTE(random_state=42)
X_train_jaundice_res, y_train_jaundice_res = smote.fit_resample(X_train_jaundice, y_train_jaundice)
X_train_hemolytic_res, y_train_hemolytic_res = smote.fit_resample(X_train_hemolytic, y_train_hemolytic)
X_train_hepatic_res, y_train_hepatic_res = smote.fit_resample(X_train_hepatic, y_train_hepatic)
X_train_obstructive_res, y_train_obstructive_res = smote.fit_resample(X_train_obstructive, y_train_obstructive)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Separate features and target variables
X = df.iloc[:, :-4]  # All columns except the last four
y_jaundice = df.iloc[:, -4]  # Fourth last column
y_hemolytic = df.iloc[:, -3]  # Third last column
y_hepatic = df.iloc[:, -2]  # Second last column
y_obstructive = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_jaundice, X_test_jaundice, y_train_jaundice, y_test_jaundice = train_test_split(X, y_jaundice, test_size=0.2, random_state=42, stratify=y_jaundice)
X_train_hemolytic, X_test_hemolytic, y_train_hemolytic, y_test_hemolytic = train_test_split(X, y_hemolytic, test_size=0.2, random_state=42, stratify=y_hemolytic)
X_train_hepatic, X_test_hepatic, y_train_hepatic, y_test_hepatic = train_test_split(X, y_hepatic, test_size=0.2, random_state=42, stratify=y_hepatic)
X_train_obstructive, X_test_obstructive, y_train_obstructive, y_test_obstructive = train_test_split(X, y_obstructive, test_size=0.2, random_state=42, stratify=y_obstructive)

# Print train-test splits
print(f"Train-test split for Jaundice: X_train shape: {X_train_jaundice.shape}, X_test shape: {X_test_jaundice.shape}, y_train support: {len(y_train_jaundice)}, y_test support: {len(y_test_jaundice)}")
print(f"Train-test split for Hemolytic Jaundice: X_train shape: {X_train_hemolytic.shape}, X_test shape: {X_test_hemolytic.shape}, y_train support: {len(y_train_hemolytic)}, y_test support: {len(y_test_hemolytic)}")
print(f"Train-test split for Hepatic Jaundice: X_train shape: {X_train_hepatic.shape}, X_test shape: {X_test_hepatic.shape}, y_train support: {len(y_train_hepatic)}, y_test support: {len(y_test_hepatic)}")
print(f"Train-test split for Obstructive Jaundice: X_train shape: {X_train_obstructive.shape}, X_test shape: {X_test_obstructive.shape}, y_train support: {len(y_train_obstructive)}, y_test support: {len(y_test_obstructive)}")

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train, y_train, X_test, y_test, disease_name):
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-score: {f1:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_jaundice = train_and_evaluate(X_train_jaundice, y_train_jaundice, X_test_jaundice, y_test_jaundice, 'Jaundice')
results_hemolytic = train_and_evaluate(X_train_hemolytic, y_train_hemolytic, X_test_hemolytic, y_test_hemolytic, 'Hemolytic Jaundice')
results_hepatic = train_and_evaluate(X_train_hepatic, y_train_hepatic, X_test_hepatic, y_test_hepatic, 'Hepatic Jaundice')
results_obstructive = train_and_evaluate(X_train_obstructive, y_train_obstructive, X_test_obstructive, y_test_obstructive, 'Obstructive Jaundice')

# Plotting the results
def plot_results(results, disease_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = list(results.keys())
    scores = {metric: [results[model][metric] for model in labels] for metric in metrics}

    x = np.arange(len(labels))
    width = 0.2
    offset = -0.3

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + offset + i*width, scores[metric], width, label=metric.capitalize())

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(bars, padding=3)

    fig.tight_layout(rect=[0, 0, 0.85, 1])
  
    plt.xticks(rotation=45)
    plt.show()

# Plot evaluation metrics for each disease
plot_results(results_jaundice, 'Jaundice')
plot_results(results_hemolytic, 'Hemolytic Jaundice')
plot_results(results_hepatic, 'Hepatic Jaundice')
plot_results(results_obstructive, 'Obstructive Jaundice')


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Separate features and target variables
X = df.iloc[:, :-4]  # All columns except the last four
y_jaundice = df.iloc[:, -4]  # Fourth last column
y_hemolytic = df.iloc[:, -3]  # Third last column
y_hepatic = df.iloc[:, -2]  # Second last column
y_obstructive = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_jaundice, X_test_jaundice, y_train_jaundice, y_test_jaundice = train_test_split(X, y_jaundice, test_size=0.2, random_state=42, stratify=y_jaundice)
X_train_hemolytic, X_test_hemolytic, y_train_hemolytic, y_test_hemolytic = train_test_split(X, y_hemolytic, test_size=0.2, random_state=42, stratify=y_hemolytic)
X_train_hepatic, X_test_hepatic, y_train_hepatic, y_test_hepatic = train_test_split(X, y_hepatic, test_size=0.2, random_state=42, stratify=y_hepatic)
X_train_obstructive, X_test_obstructive, y_train_obstructive, y_test_obstructive = train_test_split(X, y_obstructive, test_size=0.2, random_state=42, stratify=y_obstructive)

# Print train-test splits
print(f"Train-test split for Jaundice: X_train shape: {X_train_jaundice.shape}, X_test shape: {X_test_jaundice.shape}, y_train support: {len(y_train_jaundice)}, y_test support: {len(y_test_jaundice)}")
print(f"Train-test split for Hemolytic Jaundice: X_train shape: {X_train_hemolytic.shape}, X_test shape: {X_test_hemolytic.shape}, y_train support: {len(y_train_hemolytic)}, y_test support: {len(y_test_hemolytic)}")
print(f"Train-test split for Hepatic Jaundice: X_train shape: {X_train_hepatic.shape}, X_test shape: {X_test_hepatic.shape}, y_train support: {len(y_train_hepatic)}, y_test support: {len(y_test_hepatic)}")
print(f"Train-test split for Obstructive Jaundice: X_train shape: {X_train_obstructive.shape}, X_test shape: {X_test_obstructive.shape}, y_train support: {len(y_train_obstructive)}, y_test support: {len(y_test_obstructive)}")

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train, y_train, X_test, y_test, disease_name):
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-score: {f1:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

        # Plot ROC curve
        plot_roc_curve(model, X_test, y_test, name, disease_name)

    return results

# Plotting ROC curve
def plot_roc_curve(model, X_test, y_test, model_name, disease_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} - {disease_name}')
    plt.legend(loc='lower right')
    plt.show()

# Train and evaluate models for each disease
results_jaundice = train_and_evaluate(X_train_jaundice, y_train_jaundice, X_test_jaundice, y_test_jaundice, 'Jaundice')
results_hemolytic = train_and_evaluate(X_train_hemolytic, y_train_hemolytic, X_test_hemolytic, y_test_hemolytic, 'Hemolytic Jaundice')
results_hepatic = train_and_evaluate(X_train_hepatic, y_train_hepatic, X_test_hepatic, y_test_hepatic, 'Hepatic Jaundice')
results_obstructive = train_and_evaluate(X_train_obstructive, y_train_obstructive, X_test_obstructive, y_test_obstructive, 'Obstructive Jaundice')

# Plotting the results
def plot_evaluation_metrics(results, disease_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = list(results.keys())
    scores = {metric: [results[model][metric] for model in labels] for metric in metrics}

    x = np.arange(len(labels))
    width = 0.2
    offset = -0.3

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + offset + i*width, scores[metric], width, label=metric.capitalize())

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(bars, padding=3)

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Plot evaluation metrics for each disease
plot_evaluation_metrics(results_jaundice, 'Jaundice')
plot_evaluation_metrics(results_hemolytic, 'Hemolytic Jaundice')
plot_evaluation_metrics(results_hepatic, 'Hepatic Jaundice')
plot_evaluation_metrics(results_obstructive, 'Obstructive Jaundice')


# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Separate features and target variables
X = df.iloc[:, :-4]  # All columns except the last four
y_jaundice = df.iloc[:, -4]  # Fourth last column
y_hemolytic = df.iloc[:, -3]  # Third last column
y_hepatic = df.iloc[:, -2]  # Second last column
y_obstructive = df.iloc[:, -1]  # last column

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train, y_train, X_test, y_test, disease_name):
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_res, y_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model,
            'X_test': X_test
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    # Plot ROC curves for all models
    plot_roc_curves(results, disease_name)

    return results


# Separate into train-test splits with stratification
X_train_jaundice, X_test_jaundice, y_train_jaundice, y_test_jaundice = train_test_split(X, y_jaundice, test_size=0.2, random_state=42, stratify=y_jaundice)
X_train_hemolytic, X_test_hemolytic, y_train_hemolytic, y_test_hemolytic = train_test_split(X, y_hemolytic, test_size=0.2, random_state=42, stratify=y_hemolytic)
X_train_hepatic, X_test_hepatic, y_train_hepatic, y_test_hepatic = train_test_split(X, y_hepatic, test_size=0.2, random_state=42, stratify=y_hepatic)
X_train_obstructive, X_test_obstructive, y_train_obstructive, y_test_obstructive = train_test_split(X, y_obstructive, test_size=0.2, random_state=42, stratify=y_obstructive)

# Train and evaluate models for each disease
results_jaundice = train_and_evaluate(X_train_jaundice, y_train_jaundice, X_test_jaundice, y_test_jaundice, 'Jaundice')
results_hemolytic = train_and_evaluate(X_train_hemolytic, y_train_hemolytic, X_test_hemolytic, y_test_hemolytic, 'Hemolytic Jaundice')
results_hepatic = train_and_evaluate(X_train_hepatic, y_train_hepatic, X_test_hepatic, y_test_hepatic, 'Hepatic Jaundice')
results_obstructive = train_and_evaluate(X_train_obstructive, y_train_obstructive, X_test_obstructive, y_test_obstructive, 'Obstructive Jaundice')


# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Load and preprocess data
data = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Check column names in the DataFrame
print("Columns in the DataFrame:")
print(data.columns)

# Assuming these are the target variables (diseases), verify exact names
target_columns = ['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice']

# Verify each target column exists in the DataFrame
for col in target_columns:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in the DataFrame. Please check your column names or adjust your code accordingly.")

# Separate features (X) and targets (y)
X = data.drop(columns=target_columns)
y = data[target_columns]

# Step 2: Apply SMOTE to each disease column separately
smote = SMOTE(random_state=42)
data_resampled = []

for i in range(y.shape[1]):
    X_resampled_i, y_resampled_i = smote.fit_resample(X, y.iloc[:, i])
    data_resampled.append((X_resampled_i, y_resampled_i, target_columns[i]))

# Step 3: Split resampled data into training and testing sets, train Decision Tree models, and evaluate
dt_models = []

for X_resampled_i, y_resampled_i, target_col in data_resampled:
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_resampled_i, y_resampled_i, train_size=68, test_size=17, random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_i, y_train_i)
    dt_models.append((dt, target_col))  # Store model and corresponding disease name
    
    # Evaluate the model
    y_pred_i = dt.predict(X_test_i)
    
    print(f"Classification Report for {target_col}:")
    print(classification_report(y_test_i, y_pred_i))
    
    cm = confusion_matrix(y_test_i, y_pred_i)
    print(f"\nConfusion Matrix for {target_col}:")
    print(cm)
    print("\n")
    
    # Plot the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"Confusion Matrix for {target_col}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Visualize feature importances
    importances = dt.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X.columns[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances for {target_col}")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), names, rotation=90)
    plt.show()

    # Visualize Decision Tree
    plt.figure(figsize=(15, 10))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['0', '1'])
    plt.title(f"Decision Tree for {target_col}")
    plt.show()

# Step 4: Top Feature Importances
plt.figure(figsize=(10, 6))
importances_all = np.mean([dt.feature_importances_ for dt, _ in dt_models], axis=0)
indices_all = np.argsort(importances_all)[::-1][:10]  # Top 10 features
plt.barh(range(10), importances_all[indices_all], align='center')
plt.yticks(range(10), [X.columns[i] for i in indices_all])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances across all Diseases')
plt.show()


# In[21]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz

# Set the path to Graphviz executables
os.environ["PATH"] += os.pathsep + 'D:\\windows_10_cmake_Release_Graphviz-11.0.0-win64\\Graphviz-11.0.0-win64\\bin'

# Load your dataset
file_path = r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx'
data = pd.read_excel(file_path)

# Check column names to identify any issues
print("Columns in the dataset:", data.columns)

# Update target columns based on actual column names in your dataset
target_columns = ['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice']

# Check if target columns are present
for column in target_columns:
    if column not in data.columns:
        print(f"Error: Column '{column}' not found in the dataset.")

# Ensure 'Age' column exists before creating 'Age Category'
if 'Age' in data.columns:
    data['Age Category'] = data['Age'].apply(lambda x: 'below 45' if x < 45 else 'above 45')

    # Function to plot feature importance
    def plot_feature_importance(model, feature_names, title):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
        plt.title(f'Feature Importance for {title}')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.show()

    # Function to visualize a single tree from the Random Forest
    def visualize_tree(model, feature_names, target, tree_index=0):
        estimator = model.estimators_[tree_index]

        dot_data = export_graphviz(estimator, out_file=None, 
                                   feature_names=feature_names,  
                                   class_names=[str(class_name) for class_name in model.classes_],  
                                   filled=True, rounded=True,  
                                   special_characters=True)  
        graph = graphviz.Source(dot_data)  
        graph.render(f"random_forest_tree_{target.replace(' ', '_')}")
        graph.view()

    # Separate features (X) and targets (y)
    X = data.drop(columns=target_columns + ['Age Category'])  # Exclude non-numeric column 'Age Category'
    feature_names = X.columns

    # Train and evaluate models for each target
    for target in target_columns:
        y = data[target]

        # Split the data into training (68 samples) and testing (17 samples) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=68, test_size=17, random_state=42)

        # Train Random Forest model
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Evaluation for {target} using Random Forest:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:")
        print(classification_rep)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\n")

        # Plot the Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f"Confusion Matrix for {target}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Plot feature importance
        plot_feature_importance(rf_model, feature_names, target)

        # Visualize a single tree from the Random Forest
        visualize_tree(rf_model, feature_names, target, tree_index=0)
else:
    print("Error: 'Age' column not found in the dataset.")


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Separate features and target variables
X = df.iloc[:, :-4]  # All columns except the last four
y_jaundice = df.iloc[:, -4]  # Fourth last column
y_hemolytic = df.iloc[:, -3]  # Third last column
y_hepatic = df.iloc[:, -2]  # Second last column
y_obstructive = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_jaundice, X_test_jaundice, y_train_jaundice, y_test_jaundice = train_test_split(X, y_jaundice, test_size=0.2, random_state=42, stratify=y_jaundice)
X_train_hemolytic, X_test_hemolytic, y_train_hemolytic, y_test_hemolytic = train_test_split(X, y_hemolytic, test_size=0.2, random_state=42, stratify=y_hemolytic)
X_train_hepatic, X_test_hepatic, y_train_hepatic, y_test_hepatic = train_test_split(X, y_hepatic, test_size=0.2, random_state=42, stratify=y_hepatic)
X_train_obstructive, X_test_obstructive, y_train_obstructive, y_test_obstructive = train_test_split(X, y_obstructive, test_size=0.2, random_state=42, stratify=y_obstructive)

# Print train-test splits
print(f"Train-test split for Jaundice: X_train shape: {X_train_jaundice.shape}, X_test shape: {X_test_jaundice.shape}, y_train support: {len(y_train_jaundice)}, y_test support: {len(y_test_jaundice)}")
print(f"Train-test split for Hemolytic Jaundice: X_train shape: {X_train_hemolytic.shape}, X_test shape: {X_test_hemolytic.shape}, y_train support: {len(y_train_hemolytic)}, y_test support: {len(y_test_hemolytic)}")
print(f"Train-test split for Hepatic Jaundice: X_train shape: {X_train_hepatic.shape}, X_test shape: {X_test_hepatic.shape}, y_train support: {len(y_train_hepatic)}, y_test support: {len(y_test_hepatic)}")
print(f"Train-test split for Obstructive Jaundice: X_train shape: {X_train_obstructive.shape}, X_test shape: {X_test_obstructive.shape}, y_train support: {len(y_train_obstructive)}, y_test support: {len(y_test_obstructive)}")

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train, y_train, X_test, y_test, disease_name):
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1-score: {f1:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_jaundice = train_and_evaluate(X_train_jaundice, y_train_jaundice, X_test_jaundice, y_test_jaundice, 'Jaundice')
results_hemolytic = train_and_evaluate(X_train_hemolytic, y_train_hemolytic, X_test_hemolytic, y_test_hemolytic, 'Hemolytic Jaundice')
results_hepatic = train_and_evaluate(X_train_hepatic, y_train_hepatic, X_test_hepatic, y_test_hepatic, 'Hepatic Jaundice')
results_obstructive = train_and_evaluate(X_train_obstructive, y_train_obstructive, X_test_obstructive, y_test_obstructive, 'Obstructive Jaundice')

# Plotting the results
def plot_results(results, disease_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = list(results.keys())
    scores = {metric: [results[model][metric] for model in labels] for metric in metrics}

    x = np.arange(len(labels))
    width = 0.2
    offset = -0.3

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + offset + i*width, scores[metric], width, label=metric.capitalize())

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(bars, padding=3)

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.xticks(rotation=45)
    plt.show()

# Plot evaluation metrics for each disease
plot_results(results_jaundice, 'Jaundice')
plot_results(results_hemolytic, 'Hemolytic Jaundice')
plot_results(results_hepatic, 'Hepatic Jaundice')
plot_results(results_obstructive, 'Obstructive Jaundice')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {title}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrices for each disease and model
for disease_name, results in zip(['Jaundice', 'Hemolytic Jaundice', 'Hepatic Jaundice', 'Obstructive Jaundice'], [results_jaundice, results_hemolytic, results_hepatic, results_obstructive]):
    for model_name in results.keys():
        cm = confusion_matrix(results[model_name]['y_test'], results[model_name]['y_pred'])
        plot_confusion_matrix(cm, ['No', 'Yes'], f"{disease_name} using {model_name}")


# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Separate features and target variables
X = df.iloc[:, :-4]  # All columns except the last four
y_jaundice = df.iloc[:, -4]  # Fourth last column
y_hemolytic = df.iloc[:, -3]  # Third last column
y_hepatic = df.iloc[:, -2]  # Second last column
y_obstructive = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_jaundice, X_test_jaundice, y_train_jaundice, y_test_jaundice = train_test_split(X, y_jaundice, test_size=0.2, random_state=42, stratify=y_jaundice)
X_train_hemolytic, X_test_hemolytic, y_train_hemolytic, y_test_hemolytic = train_test_split(X, y_hemolytic, test_size=0.2, random_state=42, stratify=y_hemolytic)
X_train_hepatic, X_test_hepatic, y_train_hepatic, y_test_hepatic = train_test_split(X, y_hepatic, test_size=0.2, random_state=42, stratify=y_hepatic)
X_train_obstructive, X_test_obstructive, y_train_obstructive, y_test_obstructive = train_test_split(X, y_obstructive, test_size=0.2, random_state=42, stratify=y_obstructive)

# Print train-test splits
print(f"Train-test split for Jaundice: X_train shape: {X_train_jaundice.shape}, X_test shape: {X_test_jaundice.shape}, y_train support: {len(y_train_jaundice)}, y_test support: {len(y_test_jaundice)}")
print(f"Train-test split for Hemolytic Jaundice: X_train shape: {X_train_hemolytic.shape}, X_test shape: {X_test_hemolytic.shape}, y_train support: {len(y_train_hemolytic)}, y_test support: {len(y_test_hemolytic)}")
print(f"Train-test split for Hepatic Jaundice: X_train shape: {X_train_hepatic.shape}, X_test shape: {X_test_hepatic.shape}, y_train support: {len(y_train_hepatic)}, y_test support: {len(y_test_hepatic)}")
print(f"Train-test split for Obstructive Jaundice: X_train shape: {X_train_obstructive.shape}, X_test shape: {X_test_obstructive.shape}, y_train support: {len(y_train_obstructive)}, y_test support: {len(y_test_obstructive)}")

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train, y_train, X_test, y_test, disease_name):
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
    mean_cv_score = np.mean(cv_scores)
    
    print(f'Evaluation for {disease_name} using Logistic Regression:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    print(f'Mean CV Score: {mean_cv_score:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_cv_score': mean_cv_score,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

# Train and evaluate models for each disease
results_jaundice = train_and_evaluate(X_train_jaundice, y_train_jaundice, X_test_jaundice, y_test_jaundice, 'Jaundice')
results_hemolytic = train_and_evaluate(X_train_hemolytic, y_train_hemolytic, X_test_hemolytic, y_test_hemolytic, 'Hemolytic Jaundice')
results_hepatic = train_and_evaluate(X_train_hepatic, y_train_hepatic, X_test_hepatic, y_test_hepatic, 'Hepatic Jaundice')
results_obstructive = train_and_evaluate(X_train_obstructive, y_train_obstructive, X_test_obstructive, y_test_obstructive, 'Obstructive Jaundice')

# Plotting the results
def plot_results(results, disease_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = ['Logistic Regression']
    scores = {metric: [results[metric]] for metric in metrics}

    x = np.arange(len(labels))
    width = 0.2
    offset = -0.3

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + offset + i*width, scores[metric], width, label=metric.capitalize())

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(bars, padding=3)

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.xticks(rotation=45)
    plt.show()

# Plot evaluation metrics for each disease
plot_results(results_jaundice, 'Jaundice')
plot_results(results_hemolytic, 'Hemolytic Jaundice')
plot_results(results_hepatic, 'Hepatic Jaundice')
plot_results(results_obstructive, 'Obstructive Jaundice')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {title}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrices for each disease
plot_confusion_matrix(confusion_matrix(results_jaundice['y_test'], results_jaundice['y_pred']), ['No', 'Yes'], 'Jaundice')
plot_confusion_matrix(confusion_matrix(results_hemolytic['y_test'], results_hemolytic['y_pred']), ['No', 'Yes'], 'Hemolytic Jaundice')
plot_confusion_matrix(confusion_matrix(results_hepatic['y_test'], results_hepatic['y_pred']), ['No', 'Yes'], 'Hepatic Jaundice')
plot_confusion_matrix(confusion_matrix(results_obstructive['y_test'], results_obstructive['y_pred']), ['No', 'Yes'], 'Obstructive Jaundice')

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_prob, disease_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {disease_name}')
    plt.legend(loc='lower right')
    plt.show()

# Plot ROC curves for each disease
plot_roc_curve(results_jaundice['y_test'], results_jaundice['y_pred_prob'], 'Jaundice')
plot_roc_curve(results_hemolytic['y_test'], results_hemolytic['y_pred_prob'], 'Hemolytic Jaundice')
plot_roc_curve(results_hepatic['y_test'], results_hepatic['y_pred_prob'], 'Hepatic Jaundice')
plot_roc_curve(results_obstructive['y_test'], results_obstructive['y_pred_prob'], 'Obstructive Jaundice')


# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\male below 45 svm2.xlsx')

# Separate features and target variables
X = df.iloc[:, :-4]  # All columns except the last four
y_jaundice = df.iloc[:, -4]  # Fourth last column
y_hemolytic = df.iloc[:, -3]  # Third last column
y_hepatic = df.iloc[:, -2]  # Second last column
y_obstructive = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_jaundice, X_test_jaundice, y_train_jaundice, y_test_jaundice = train_test_split(X, y_jaundice, test_size=0.2, random_state=42, stratify=y_jaundice)
X_train_hemolytic, X_test_hemolytic, y_train_hemolytic, y_test_hemolytic = train_test_split(X, y_hemolytic, test_size=0.2, random_state=42, stratify=y_hemolytic)
X_train_hepatic, X_test_hepatic, y_train_hepatic, y_test_hepatic = train_test_split(X, y_hepatic, test_size=0.2, random_state=42, stratify=y_hepatic)
X_train_obstructive, X_test_obstructive, y_train_obstructive, y_test_obstructive = train_test_split(X, y_obstructive, test_size=0.2, random_state=42, stratify=y_obstructive)

# Print train-test splits
print(f"Train-test split for Jaundice: X_train shape: {X_train_jaundice.shape}, X_test shape: {X_test_jaundice.shape}, y_train support: {len(y_train_jaundice)}, y_test support: {len(y_test_jaundice)}")
print(f"Train-test split for Hemolytic Jaundice: X_train shape: {X_train_hemolytic.shape}, X_test shape: {X_test_hemolytic.shape}, y_train support: {len(y_train_hemolytic)}, y_test support: {len(y_test_hemolytic)}")
print(f"Train-test split for Hepatic Jaundice: X_train shape: {X_train_hepatic.shape}, X_test shape: {X_test_hepatic.shape}, y_train support: {len(y_train_hepatic)}, y_test support: {len(y_test_hepatic)}")
print(f"Train-test split for Obstructive Jaundice: X_train shape: {X_train_obstructive.shape}, X_test shape: {X_test_obstructive.shape}, y_train support: {len(y_train_obstructive)}, y_test support: {len(y_test_obstructive)}")

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Gaussian Naive Bayes model
model = GaussianNB()

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train, y_train, X_test, y_test, disease_name):
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
    mean_cv_score = np.mean(cv_scores)
    
    print(f'Evaluation for {disease_name} using Gaussian Naive Bayes:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    print(f'Mean CV Score: {mean_cv_score:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_cv_score': mean_cv_score,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

# Train and evaluate models for each disease
results_jaundice = train_and_evaluate(X_train_jaundice, y_train_jaundice, X_test_jaundice, y_test_jaundice, 'Jaundice')
results_hemolytic = train_and_evaluate(X_train_hemolytic, y_train_hemolytic, X_test_hemolytic, y_test_hemolytic, 'Hemolytic Jaundice')
results_hepatic = train_and_evaluate(X_train_hepatic, y_train_hepatic, X_test_hepatic, y_test_hepatic, 'Hepatic Jaundice')
results_obstructive = train_and_evaluate(X_train_obstructive, y_train_obstructive, X_test_obstructive, y_test_obstructive, 'Obstructive Jaundice')

# Plotting the results
def plot_results(results, disease_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = {metric: [results[metric]] for metric in metrics}

    labels = ['Gaussian Naive Bayes']
    x = np.arange(len(labels))
    width = 0.2  # Width of the bars
    offset = -0.3

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + offset + i*width, scores[metric], width, label=metric.capitalize())

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.xticks(rotation=45)
    plt.show()

# Plot evaluation metrics for each disease
plot_results(results_jaundice, 'Jaundice')
plot_results(results_hemolytic, 'Hemolytic Jaundice')
plot_results(results_hepatic, 'Hepatic Jaundice')
plot_results(results_obstructive, 'Obstructive Jaundice')

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_prob, disease_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {disease_name}')
    plt.legend(loc='lower right')
    plt.show()

# Plot ROC curves for each disease
plot_roc_curve(results_jaundice['y_test'], results_jaundice['y_pred_prob'], 'Jaundice')
plot_roc_curve(results_hemolytic['y_test'], results_hemolytic['y_pred_prob'], 'Hemolytic Jaundice')
plot_roc_curve(results_hepatic['y_test'], results_hepatic['y_pred_prob'], 'Hepatic Jaundice')
plot_roc_curve(results_obstructive['y_test'], results_obstructive['y_pred_prob'], 'Obstructive Jaundice')


# In[ ]:




