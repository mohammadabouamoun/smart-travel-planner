#!/usr/bin/env python3
"""
ML Classifier for Travel Style Prediction – improved version
Selects best model via cross-validation, tunes all candidates, saves final model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score
import joblib
import os
from datetime import datetime

# ------------------------------
# 1. Load dataset
# ------------------------------
DATA_PATH = 'data/destinations.csv'
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run the generation script first.")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Total samples: {len(df)}")
print(df['style'].value_counts())

X = df.drop(['name', 'style'], axis=1)
y = df['style']

# Identify features: exclude binary columns from scaling
binary_cols = ['has_beach', 'english_friendly']   # 0/1 columns
numeric_cols = [col for col in X.select_dtypes(include=['int64', 'float64']).columns
                if col not in binary_cols]
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric (to scale): {numeric_cols}")
print(f"Categorical: {categorical_cols}")
print(f"Binary (pass through): {binary_cols}")

# ------------------------------
# 2. Stratified train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# ------------------------------
# 3. Preprocessing pipeline
# ------------------------------
# Numeric: scale; Categorical: one-hot; Binary: keep as-is (passthrough)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'  # binary columns pass through unscaled
)

# ------------------------------
# 4. Define base models
# ------------------------------
base_models = {
    'LogisticRegression': LogisticRegression(
    solver='lbfgs', max_iter=2000,
    class_weight='balanced', random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    )
}

# ------------------------------
# 5. Cross-validation strategy
# ------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------
# 6. Hyperparameter grids for each model
# ------------------------------
param_grids = {
    'LogisticRegression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['lbfgs'],
        'classifier__penalty': ['l2']   # multinomial supports only l2
    },
    'RandomForest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    },
    'GradientBoosting': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [3, 5]
    }
}

best_model_name = None
best_pipeline = None
best_cv_f1 = -1
cv_results = []

print("\n=== Tuning & Cross‑Validation ===")
for name, model in base_models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    grid = GridSearchCV(pipeline, param_grids[name], cv=cv,
                        scoring='f1_macro', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    
    cv_mean = grid.best_score_
    cv_std = grid.cv_results_['std_test_score'][grid.best_index_]
    print(f"{name}: best CV F1 = {cv_mean:.4f} ± {cv_std:.4f} with {grid.best_params_}")
    
    cv_results.append({
        'model': name,
        'best_cv_f1': cv_mean,
        'cv_std': cv_std,
        'best_params': grid.best_params_
    })
    
    # Track the overall best
    if cv_mean > best_cv_f1:
        best_cv_f1 = cv_mean
        best_model_name = name
        best_pipeline = grid.best_estimator_

print(f"\nBest model after tuning: {best_model_name} (CV F1 = {best_cv_f1:.4f})")

# ------------------------------
# 7. Final evaluation on test set (once!)
# ------------------------------
y_pred_test = best_pipeline.predict(X_test)
test_f1 = f1_score(y_test, y_pred_test, average='macro')
test_acc = (y_pred_test == y_test).mean()

print(f"Test Macro F1: {test_f1:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ------------------------------
# 8. Save results & model
# ------------------------------
# Save CV results table
cv_df = pd.DataFrame(cv_results)
cv_df['test_f1'] = test_f1
cv_df['test_accuracy'] = test_acc
cv_df['timestamp'] = datetime.now().isoformat()
cv_df.to_csv('results.csv', index=False)
print("Results saved to results.csv")

# Save best pipeline (preprocessor + classifier)
os.makedirs('backend/models', exist_ok=True)
joblib.dump(best_pipeline, 'backend/models/best_model.pkl')
print("Best model saved to backend/models/best_model.pkl")

# Per‑class report
print("\n=== Classification Report (Tuned Model on Test Set) ===")
print(classification_report(y_test, y_pred_test))

print("Step 2 complete. Model ready for integration.")