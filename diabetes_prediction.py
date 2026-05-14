# =============================================================
# SXRHE: SMOTE + XGBoost + Random Forest Hybrid Ensemble
# Early Diabetes Disease Prediction
# Author: Your Name
# Dataset: PIMA Indians Diabetes Database
# =============================================================

# ─────────────────────────────────────────
# SECTION 1: IMPORT LIBRARIES
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

print("=" * 60)
print("  SXRHE: Diabetes Prediction - Starting Analysis")
print("=" * 60)

# ─────────────────────────────────────────
# SECTION 2: LOAD DATASET
# ─────────────────────────────────────────
df = pd.read_csv('diabetes.csv')

print("\n[1] Dataset Loaded Successfully!")
print(f"    Shape: {df.shape}")
print(f"\n    First 5 rows:\n{df.head()}")
print(f"\n    Class Distribution:\n{df['Outcome'].value_counts()}")

# ─────────────────────────────────────────
# SECTION 3: DATA PREPROCESSING
# ─────────────────────────────────────────
print("\n[2] Preprocessing Data...")

# Replace biologically impossible zeros with median
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

print(f"    Missing zero values replaced with median.")

# Plot class distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='Outcome', data=df, palette='Set2')
plt.title('Class Distribution (0=No Diabetes, 1=Diabetes)')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=150)
plt.show()
print("    Saved: 01_class_distribution.png")

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=150)
plt.show()
print("    Saved: 02_correlation_heatmap.png")

# ─────────────────────────────────────────
# SECTION 4: FEATURE SELECTION (Mutual Information)
# ─────────────────────────────────────────
print("\n[3] Selecting Top Features using Mutual Information...")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values('MI Score', ascending=False)

print(f"\n    Feature Importance (Mutual Information):\n{mi_df}")

# Plot MI scores
plt.figure(figsize=(8, 5))
sns.barplot(x='MI Score', y='Feature', data=mi_df, palette='viridis')
plt.title('Feature Importance - Mutual Information')
plt.tight_layout()
plt.savefig('03_feature_importance_MI.png', dpi=150)
plt.show()
print("    Saved: 03_feature_importance_MI.png")

# Select top 6 features
top_features = mi_df['Feature'].head(6).tolist()
print(f"\n    Top 6 Features Selected: {top_features}")
X = X[top_features]

# ─────────────────────────────────────────
# SECTION 5: APPLY SMOTE (Handle Imbalance)
# ─────────────────────────────────────────
print("\n[4] Applying SMOTE to balance dataset...")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"    Before SMOTE: {dict(y.value_counts())}")
print(f"    After SMOTE:  {dict(pd.Series(y_resampled).value_counts())}")

# ─────────────────────────────────────────
# SECTION 6: TRAIN/TEST SPLIT + SCALING
# ─────────────────────────────────────────
print("\n[5] Splitting data (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"    Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────
# SECTION 7: DEFINE ALL MODELS
# ─────────────────────────────────────────
print("\n[6] Training Models...")

# Baseline models for comparison
baseline_models = {
    'Decision Tree':    DecisionTreeClassifier(random_state=42),
    'KNN':              KNeighborsClassifier(),
    'Naive Bayes':      GaussianNB(),
    'SVM':              SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest':    RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost':          XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
}

# PROPOSED MODEL: SXRHE Stacking Ensemble
base_learners = [
    ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
    ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
]
meta_learner = LogisticRegression(random_state=42)

sxrhe_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)

# ─────────────────────────────────────────
# SECTION 8: STRATIFIED K-FOLD CROSS VALIDATION
# ─────────────────────────────────────────
print("\n[7] Running Stratified 10-Fold Cross Validation...")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_results = {}
all_models = {**baseline_models, 'SXRHE (Proposed)': sxrhe_model}

for name, model in all_models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    cv_results[name] = cv_scores
    print(f"    {name:25s}: CV Accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────
# SECTION 9: TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────
print("\n[8] Training and Evaluating on Test Set...")

results = []

for name, model in all_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': round(acc * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall': round(rec * 100, 2),
        'F1-Score': round(f1 * 100, 2),
        'AUC-ROC': round(auc * 100, 2)
    })

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print(f"\n    Comparative Analysis:\n{results_df.to_string(index=False)}")

# Save comparison table
results_df.to_csv('comparative_analysis.csv', index=False)
print("\n    Saved: comparative_analysis.csv")

# ─────────────────────────────────────────
# SECTION 10: VISUALIZATIONS
# ─────────────────────────────────────────
print("\n[9] Generating Visualizations...")

# --- Comparative Bar Chart ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x = np.arange(len(results_df))
fig, ax = plt.subplots(figsize=(14, 6))
width = 0.15
for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results_df[metric], width, label=metric)
ax.set_xlabel('Models')
ax.set_ylabel('Score (%)')
ax.set_title('Comparative Analysis: All Models vs SXRHE (Proposed)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df['Model'], rotation=30, ha='right')
ax.legend()
ax.set_ylim(50, 105)
plt.tight_layout()
plt.savefig('04_comparative_analysis.png', dpi=150)
plt.show()
print("    Saved: 04_comparative_analysis.png")

# --- Confusion Matrix for SXRHE ---
sxrhe_model.fit(X_train, y_train)
y_pred_sxrhe = sxrhe_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_sxrhe)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix - SXRHE (Proposed Model)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('05_confusion_matrix_SXRHE.png', dpi=150)
plt.show()
print("    Saved: 05_confusion_matrix_SXRHE.png")

# --- ROC Curves for All Models ---
plt.figure(figsize=(10, 7))
for name, model in all_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    lw = 3 if name == 'SXRHE (Proposed)' else 1
    plt.plot(fpr, tpr, lw=lw, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - All Models')
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig('06_roc_curves.png', dpi=150)
plt.show()
print("    Saved: 06_roc_curves.png")

# --- Cross Validation Box Plot ---
plt.figure(figsize=(12, 6))
cv_data = [cv_results[name] * 100 for name in all_models.keys()]
plt.boxplot(cv_data, labels=all_models.keys(), patch_artist=True)
plt.xticks(rotation=30, ha='right')
plt.ylabel('CV Accuracy (%)')
plt.title('10-Fold Cross Validation Accuracy Distribution')
plt.tight_layout()
plt.savefig('07_cross_validation_boxplot.png', dpi=150)
plt.show()
print("    Saved: 07_cross_validation_boxplot.png")

# ─────────────────────────────────────────
# SECTION 11: SHAP EXPLAINABILITY
# ─────────────────────────────────────────
print("\n[10] Generating SHAP Explainability...")

# Use Random Forest for SHAP (compatible with all versions)
rf_shap = RandomForestClassifier(n_estimators=100, random_state=42)
rf_shap.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf_shap)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list — take class 1
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test,
                  feature_names=top_features,
                  show=False)
plt.title('SHAP Feature Importance Summary')
plt.tight_layout()
plt.savefig('08_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 08_shap_summary.png")

# SHAP Bar Plot
plt.figure()
shap.summary_plot(shap_values, X_test,
                  feature_names=top_features,
                  plot_type='bar',
                  show=False)
plt.title('SHAP Mean Feature Importance')
plt.tight_layout()
plt.savefig('09_shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Saved: 09_shap_bar.png")

# ─────────────────────────────────────────
# SECTION 12: FINAL REPORT
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n  Proposed Model: SXRHE (Stacking Ensemble)")
print(f"  Dataset: PIMA Indians Diabetes Database")
print(f"  Top Features Used: {top_features}")
print(f"\n  Performance Metrics:")
sxrhe_row = results_df[results_df['Model'] == 'SXRHE (Proposed)'].iloc[0]
for metric in metrics:
    print(f"    {metric:15s}: {sxrhe_row[metric]:.2f}%")

print(f"\n  Classification Report (SXRHE):")
print(classification_report(y_test, y_pred_sxrhe,
                             target_names=['No Diabetes', 'Diabetes']))

print("\n  All output images saved in your project folder!")
print("  Files: 01 to 09 PNG images + comparative_analysis.csv")
print("\n" + "=" * 60)
print("  DONE! Your project code ran successfully!")
print("=" * 60)
