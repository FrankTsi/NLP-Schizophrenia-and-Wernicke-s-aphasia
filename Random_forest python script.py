import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Loading data
df = pd.read_csv('Word_sent_embedings.csv', sep = ',') #replace with your path
df.head()

# Replace all NaN values with zero
df.fillna(0, inplace=True)

#SSD VS WERNICKE COMPARISON
# Filter the dataset to include only two levels of Diagnosis: SSD vs Wernicke, SSD vs Healthy, Wernicke vs Healthy
df_bin = df[df['Diagnosis'].isin(['SSD', 'Wernicke'])] # Healthy vs Wernicke

# Split the dataset into features and target variable
X = df_bin.drop(columns=['ID', 'Diagnosis']) #, 'Diagnosis_Personal', 'Diagnosis_Picture'
y = df_bin['Diagnosis']
#Training and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Majority Baseline Model for Wernicke vs SSD
from sklearn.metrics import accuracy_score
# Majority Class Baseline Classifier
mb = DummyClassifier(strategy='most_frequent')
# Fit the model
mb.fit(X, y)
# Predict using the same data
y_pred = mb.predict(X)
# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("Majority Class Baseline accuracy:", accuracy)

# Define the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Perform k=5 fold cross-validation (Stratified KFold) for Healthy controls vs Wernicke
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store classification reports and metrics
classification_reports = []
precisions = []
recalls = []
f1_scores = []
accuracies = []

# Perform k-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    print("Fold:", fold)
    
    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the classifier
    rf.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rf.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports.append(report)
    
    # Aggregate precision, recall, F1-score, and accuracy
    precisions.append(report['macro avg']['precision'])
    recalls.append(report['macro avg']['recall'])
    f1_scores.append(report['macro avg']['f1-score'])
    accuracies.append(report['accuracy'])

# Compute mean scores for precision, recall, F1, and accuracy
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1 = np.mean(f1_scores)
mean_accuracy = np.mean(accuracies)

# Print mean scores for precision, recall, F1, and accuracy
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean F1-score:", mean_f1)
print("Mean Accuracy:", mean_accuracy)

# Print or store the classification reports
print("Classification Reports for Each Fold:")
for idx, report in enumerate(classification_reports, 1):
    print("Fold", idx, ":\n", classification_report(y_test, y_pred))


#Healthy_C VS WERNICKE COMPARISON
# Filter the dataset to include only two levels of Diagnosis: SSD vs Wernicke, SSD vs Healthy, Wernicke vs Healthy
df_bin = df[df['Diagnosis'].isin(['Healthy_C', 'Wernicke'])] # Healthy vs Wernicke

# Split the dataset into features and target
X = df_bin.drop(columns=['ID', 'Diagnosis'])
y = df_bin['Diagnosis']

#Training and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest classifier
rf1 = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform k=5 fold cross-validation (Stratified KFold) for Healthy controls vs Wernicke
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store classification reports and metrics
classification_reports = []
precisions = []
recalls = []
f1_scores = []
accuracies = []

# Perform k-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    print("Fold:", fold)
    
    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the classifier
    rf1.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rf1.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports.append(report)
    
    # Aggregate precision, recall, F1-score, and accuracy
    precisions.append(report['macro avg']['precision'])
    recalls.append(report['macro avg']['recall'])
    f1_scores.append(report['macro avg']['f1-score'])
    accuracies.append(report['accuracy'])

# Compute mean scores for precision, recall, F1, and accuracy
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1 = np.mean(f1_scores)
mean_accuracy = np.mean(accuracies)

# Print mean scores for precision, recall, F1, and accuracy
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean F1-score:", mean_f1)
print("Mean Accuracy:", mean_accuracy)

# Print or store the classification reports
print("Classification Reports for Each Fold:")
for idx, report in enumerate(classification_reports, 1):
    print("Fold", idx, ":\n", classification_report(y_test, y_pred))

#HEALTHY CONTROLS VS SSD COMPARISON
# Filter the dataset to include only two levels of Diagnosis: SSD vs Wernicke, SSD vs Healthy, Wernicke vs Healthy
df_bin = df[df['Diagnosis'].isin(['SSD', 'Healthy_C'])] # Healthy vs SSD

# Split the dataset into features and target
X = df_bin.drop(columns=['ID', 'Diagnosis'])
y = df_bin['Diagnosis']

#Training and Test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Baseline Model for  SSD vs Healthy control
from sklearn.metrics import accuracy_score
# Majority Class Baseline Classifier
mb = DummyClassifier(strategy='most_frequent')
# Fit the model
mb.fit(X, y)
# Predict using the same data
y_pred = mb.predict(X)
# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("Majority Class Baseline accuracy:", accuracy)

# Define the Random Forest classifier
rf2 = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform k=5 fold cross-validation (Stratified KFold) for Healthy controls vs Wernicke
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store classification reports and metrics
classification_reports = []
precisions = []
recalls = []
f1_scores = []
accuracies = []

# Perform k-fold cross-validation
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    print("Fold:", fold)
    
    # Split the data
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Train the classifier
    rf2.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = rf2.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_reports.append(report)
    
    # Aggregate precision, recall, F1-score, and accuracy
    precisions.append(report['macro avg']['precision'])
    recalls.append(report['macro avg']['recall'])
    f1_scores.append(report['macro avg']['f1-score'])
    accuracies.append(report['accuracy'])

# Compute mean scores for precision, recall, F1, and accuracy
mean_precision = np.mean(precisions)
mean_recall = np.mean(recalls)
mean_f1 = np.mean(f1_scores)
mean_accuracy = np.mean(accuracies)

# Print mean scores for precision, recall, F1, and accuracy
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean F1-score:", mean_f1)
print("Mean Accuracy:", mean_accuracy)

# Print or store the classification reports
print("Classification Reports for Each Fold:")
for idx, report in enumerate(classification_reports, 1):
    print("Fold", idx, ":\n", classification_report(y_test, y_pred))

#FEATURE IMPORTANCE
# Get feature importances for Wernicke vs Healthy Control
feature_importances_WHC = rf1.feature_importances_

# Get feature importances for Wernicke vs Patients with Disorder
feature_importances_SSDHC = rf2.feature_importances_

# Get feature names
feature_names = [col for col in df_bin.columns if col not in ['ID', 'Diagnosis']]

# Create DataFrames to hold feature names and importances
feature_importance_df_WHC = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances_WHC})
feature_importance_df_SSDHC = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances_SSDHC})

# Sort the DataFrames by importance in descending order
feature_importance_df_WHC = feature_importance_df_WHC.sort_values(by='Importance', ascending=False)
feature_importance_df_SSDHC = feature_importance_df_SSDHC.sort_values(by='Importance', ascending=False)

# Select top 10 features for Wernicke vs Healthy Control
feature_importance_df_top10_WHC = feature_importance_df_WHC.head(10)

# Select top 10 features for Wernicke vs Patients with Disorder
feature_importance_df_top10_SSDHC = feature_importance_df_SSDHC.head(10)

# Plot feature importances for top 10 features for Wernicke vs Healthy Control
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Importance', y='Feature', data=feature_importance_df_top10_WHC)
plt.title('Top 10 Feature Importance (Wernicke vs Healthy Control)')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Plot feature importances for top 10 features for Wernicke vs Patients with Disorder
plt.subplot(1, 2, 2)
sns.barplot(x='Importance', y='Feature', data=feature_importance_df_top10_SSDHC)
plt.title('Top 10 Feature Importance (SSD vs Healthy Control)')
plt.xlabel('Importance')
plt.ylabel('Feature')

plt.tight_layout()

# Save the plot 
plt.savefig('feature_importance_comparison.pdf', dpi=300)
plt.show()


