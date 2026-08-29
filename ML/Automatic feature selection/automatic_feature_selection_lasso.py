# Automatic Feature Selection Using Lasso Regression

# ## Overview
# This notebook demonstrates how Lasso (Least Absolute Shrinkage and Selection Operator) regression performs **automatic feature selection** by shrinking less important feature coefficients to exactly zero. This is a powerful technique for reducing model complexity and improving interpretability.

# ## The Lasso Regression Cost Function
# 
# Lasso minimizes the following objective function:
# 
# $$\text{Cost} = \frac{1}{2N_{training}} \sum_{i=1}^{N_{training}} (Y_{real}^{(i)} - Y_{predict}^{(i)})^2 + \alpha \sum_{j=1}^{N}|a_j|$$
# 
# **Where:**
# - **First term**: Mean squared error (prediction loss) - measures how well the model fits the data
# - **Second term**: L1 regularization penalty (sum of absolute values of coefficients)
# - **α (alpha)**: Regularization strength parameter that controls the trade-off between fitting the data and keeping coefficients small
# - **N**: Total number of features
# - **a_j**: The coefficient for feature j
# 
# ## How Lasso Performs Feature Selection
# 
# 1. **Coefficient Shrinkage**: The L1 penalty encourages the algorithm to minimize coefficients as much as possible
# 2. **Sparsity**: Unlike Ridge regression (L2), Lasso can shrink coefficients **exactly to zero** (not just close to zero)
# 3. **Automatic Selection**: Features with zero coefficients are effectively removed from the model
# 4. **Collinearity Handling**: If two features are collinear (highly correlated), Lasso will keep only one and set the other's coefficient to zero, automatically addressing multicollinearity
# 
# ## Key Advantages
# - ✅ **Automatic feature selection** without manual intervention
# - ✅ **Reduced model complexity** (fewer features → easier interpretation)
# - ✅ **Better generalization** (reduced overfitting)
# - ✅ **Computational efficiency** (fewer features to compute with)
# - ✅ **Handles multicollinearity** by selecting one feature from correlated sets
# 
# ## Important Note
# **Lasso works best on scaled/normalized data** because the L1 penalty treats all features equally. Without scaling, features with larger values would have an unfair advantage in the regularization process.

# ## Step 1: Import Required Libraries

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_diabetes # diabetes dataset
import warnings

warnings.filterwarnings('ignore')

# ## Step 2: Load and Prepare the Dataset
# 
# We'll use the **Diabetes dataset** from scikit-learn:
# - **442 samples** with 10 features each
# - Features represent various physiological measurements
# - Target: quantitative measure of disease progression
# - Feature names: age, sex, bmi, bp, s1, s2, s3, s4, s5, s6

X, y = load_diabetes(return_X_y=True)
features = load_diabetes()['feature_names']

print(f"Dataset shape: {X.shape}")
print(f"Number of features: {len(features)}")
print(f"Features: {list(features)}")

# ## Step 3: Split Data and Create Pipeline
# 
# We create a pipeline that:
# 1. **StandardScaler**: Normalizes features to have mean=0 and std=1 (required for Lasso to work properly)
# 2. **Lasso**: Applies Lasso regression for feature selection
# 
# Pipelines are useful because they ensure the same preprocessing is applied consistently to both training and test data.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.33, 
    random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso())
])

# ## Step 4: Hyperparameter Tuning with GridSearchCV
# 
# ### Understanding `model__alpha` - The Key Parameter
# 
# The **alpha (α)** parameter is the **regularization strength** that controls the amount of feature selection:
# 
# | Alpha Value | Effect | Use Case |
# |-------------|--------|----------|
# | **α = 0** | No regularization (equivalent to standard Linear Regression) | Fit all features without any penalty |
# | **0 < α < 1** | Mild regularization | Slight coefficient shrinkage, most features retained |
# | **α ≈ 1-3** | Strong regularization | Many coefficients set to zero (aggressive feature selection) |
# | **α >> 3** | Very strong regularization | Most/all coefficients become zero (very few or no features selected) |
# 
# ### How to Choose Alpha?
# We use **GridSearchCV with cross-validation** to automatically find the optimal alpha:
# - Test multiple alpha values (0.1 to 3.0 in steps of 0.1)
# - Use 5-fold cross-validation to evaluate each alpha
# - Select the alpha with the lowest mean squared error
# - This provides a data-driven approach rather than manual tuning

param_grid = {
    'model__alpha': np.arange(0.1, 3, 0.1)
}

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=1
)

print("GridSearchCV configured:")
print("  - Testing 29 different alpha values")
print("  - Using 5-fold cross-validation")
print("  - Objective: Minimize mean squared error")

# ## Step 5: Train the Model
# 
# This step performs cross-validation over all alpha values:
# 1. For each alpha value
# 2. Train the model 5 times (one for each fold)
# 3. Evaluate performance on held-out fold
# 4. Calculate average performance across all folds
# 5. Select alpha with best average performance

search.fit(X_train, y_train)
print("Model training completed!")

# ## Step 6: Analyze Results
# 
# ### Best Hyperparameters Found

print("Best Hyperparameters:")
print(search.best_params_)
print(f"\nOptimal Alpha Value: {search.best_params_['model__alpha']:.2f}")
print(f"Best Cross-Validation Score (neg_MSE): {search.best_score_:.4f}")
print(f"\nInterpretation: Alpha = 1.2 provides the best balance")
print("between prediction accuracy and feature selection.")

# ### Extracted Model Coefficients
# 
# Each coefficient represents the weight/importance of a feature in the model. Zero coefficients mean the feature was eliminated by Lasso.

coef = search.best_estimator_[1].coef_

print("Model Coefficients (Feature Weights):")
print("="*40)
for feature, coefficient in zip(features, coef):
    status = "[SELECTED]" if coefficient != 0 else "[ELIMINATED]"
    print(f"{feature:5s}: {coefficient:8.4f}  {status}")

# ### Features Selected by Lasso (Non-Zero Coefficients)
# 
# These are the features that Lasso determined to be important for predicting disease progression.

selected_features = np.array(features)[coef != 0]

print(f"Selected Features ({len(selected_features)} out of {len(features)}):")
print("="*40)
print(selected_features)
print(f"\nFeature Selection Rate: {len(selected_features)/len(features)*100:.1f}%")
print(f"\nThese {len(selected_features)} features are sufficient for predictions.")

# ### Features Eliminated by Lasso (Zero Coefficients)
# 
# These features were determined to be less important or redundant for predicting disease progression. Lasso automatically set their coefficients to exactly zero, removing them from the model.

eliminated_features = np.array(features)[coef == 0]

print(f"Eliminated Features ({len(eliminated_features)} out of {len(features)}):")
print("="*40)
print(eliminated_features)
print(f"\nFeature Elimination Rate: {len(eliminated_features)/len(features)*100:.1f}%")
print(f"\nReason for elimination:")
print("  - Low predictive power (don't help reduce MSE)")
print("  - Collinear with other features (redundant information)")
#   - Adding them increases the L1 penalty without improving fit

# ## Summary & Key Findings
# 
# ### Results:
# 1. **Optimal Alpha**: The grid search determined that **α = 1.2** provides the best balance between model fit and feature selection
# 2. **Features Selected**: **7 out of 10** features were retained in the model (70%)
# 3. **Features Eliminated**: **3 features** (s2, s4, s6) were automatically excluded due to their low predictive value or collinearity
# 4. **Model Simplification**: We reduced model complexity by 30% while maintaining good predictive performance
# 
# ### Advantages Demonstrated:
# - ✅ **Automatic feature selection** - No need to manually choose which features to keep
# - ✅ **Reduced complexity** - Simpler models are easier to interpret and understand
# - ✅ **Better generalization** - Smaller model is less prone to overfitting
# - ✅ **Computational efficiency** - Fewer features mean faster predictions
# 
# ### When to Use Lasso:
# - **High-dimensional datasets** with many features
# - **Need for interpretability** - Understand which features matter
# - **Suspected multicollinearity** - Correlated features in the data
# - **Want automatic feature selection** - Instead of manual feature engineering
# - **Prevent overfitting** - When model complexity needs to be controlled
# 
# ### Comparison with Similar Methods:
# - **Ridge Regression (L2)**: Shrinks coefficients but doesn't set them to zero
# - **Elastic Net**: Combines L1 and L2 penalties for balanced feature selection
# - **Manual Feature Selection**: Time-consuming and potentially biased


"""
output:

Dataset shape: (442, 10)
Number of features: 10
Features: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
Training set size: 296
Test set size: 146
GridSearchCV configured:
  - Testing 29 different alpha values
  - Using 5-fold cross-validation
  - Objective: Minimize mean squared error
Fitting 5 folds for each of 29 candidates, totalling 145 fits
Model training completed!
Best Hyperparameters:
{'model__alpha': np.float64(1.2000000000000002)}

Optimal Alpha Value: 1.20
Best Cross-Validation Score (neg_MSE): -3165.0478

Interpretation: Alpha = 1.2 provides the best balance
between prediction accuracy and feature selection.
Model Coefficients (Feature Weights):
========================================
age  :   0.1511  [SELECTED]
sex  :  -9.0050  [SELECTED]
bmi  :  26.9020  [SELECTED]
bp   :  18.0485  [SELECTED]
s1   :  -5.4186  [SELECTED]
s2   :  -0.0000  [ELIMINATED]
s3   : -12.2791  [SELECTED]
s4   :   0.0000  [ELIMINATED]
s5   :  19.4891  [SELECTED]
s6   :   0.0000  [ELIMINATED]
Selected Features (7 out of 10):
========================================
['age' 'sex' 'bmi' 'bp' 's1' 's3' 's5']

Feature Selection Rate: 70.0%

These 7 features are sufficient for predictions.
Eliminated Features (3 out of 10):
========================================
['s2' 's4' 's6']

Feature Elimination Rate: 30.0%

Reason for elimination:
  - Low predictive power (don't help reduce MSE)
  - Collinear with other features (redundant information)

"""
