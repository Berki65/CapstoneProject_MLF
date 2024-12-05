import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Step 1: Load Data
train_data = pd.read_csv("PM_train.csv")
test_data = pd.read_csv("PM_test.csv")

# Step 2: Create RUL and Target Variable
train_data['RUL'] = train_data.groupby('id')['cycle'].transform(max) - train_data['cycle']
test_data['RUL'] = test_data.groupby('id')['cycle'].transform(max) - test_data['cycle']

# Define binary target (e.g., RUL <= 10 means maintenance required)
threshold = 10
train_data['Y'] = (train_data['RUL'] <= threshold).astype(int)
test_data['Y'] = (test_data['RUL'] <= threshold).astype(int)

# Step 3: Modify Sampling to Ensure Class Balance
# Select a wider range of data (last 50 cycles for each unit)
train_data = train_data.groupby('id').apply(lambda x: x.tail(50)).reset_index(drop=True)
test_data = test_data.groupby('id').apply(lambda x: x.tail(50)).reset_index(drop=True)

# Check class distribution
print("Class distribution in train_data:\n", train_data['Y'].value_counts())

# Step 4: Select Features
features = [col for col in train_data.columns if col not in ['id', 'cycle', 'RUL', 'Y']]
X_train = train_data[features]
y_train = train_data['Y']
X_test = test_data[features]
y_test = test_data['Y']

# Step 5: Preprocess Features (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Address Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Step 7: Train Logistic Regression Model with Balanced Classes
logreg = LogisticRegression(class_weight='balanced', random_state=42)
logreg.fit(X_train_balanced, y_train_balanced)

# Step 8: Make Predictions
y_pred = logreg.predict(X_test_scaled)
y_pred_prob = logreg.predict_proba(X_test_scaled)[:, 1]

# Step 9: Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("AUROC:", roc_auc_score(y_test, y_pred_prob))

# Step 10: Feature Coefficients (Optional for Insight)
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': logreg.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("Top Features:\n", coefficients.head(10))

# Step 10: Confusion Matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred)

# Extract TP, FP, FN, TN from the confusion matrix
TN_nn, FP_nn, FN_nn, TP_nn = conf_matrix_rf.ravel()

# Print the TP/FP breakdown
print(f"True Positives (TP): {TP_nn}")
print(f"False Positives (FP): {FP_nn}")
print(f"True Negatives (TN): {TN_nn}")
print(f"False Negatives (FN): {FN_nn}")

# Optionally, display the confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix_rf)