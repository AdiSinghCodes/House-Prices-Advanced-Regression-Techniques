"""
House Prices: Advanced Regression Techniques
Module 2: Regression + Trees + Performance Metrics

This script covers:
1. Data Exploration & Visualization
2. Feature Engineering
3. Linear/Multivariate Regression (with Regularization)
4. Tree-based Models (CART, Random Forest, Gradient Boosting)
5. Performance Metrics Comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# Linear Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Tree Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("HOUSE PRICES: ADVANCED REGRESSION TECHNIQUES")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================
print("\n1. LOADING DATA...")

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

print("\nFirst few rows of training data:")
print(train_df.head())

print("\nBasic statistics of target variable (SalePrice):")
print(train_df['SalePrice'].describe())

print("\nData types summary:")
print(train_df.dtypes.value_counts())

print("\nMissing values in training set (top 20):")
missing = train_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing.head(20))

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. EXPLORATORY DATA ANALYSIS")
print("="*80)

# Target variable distribution
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.hist(train_df['SalePrice'], bins=50, edgecolor='black')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Distribution of Sale Price')

plt.subplot(1, 3, 2)
plt.hist(np.log1p(train_df['SalePrice']), bins=50, edgecolor='black')
plt.xlabel('Log(Sale Price)')
plt.ylabel('Frequency')
plt.title('Distribution of Log-Transformed Sale Price')

plt.subplot(1, 3, 3)
from scipy import stats
stats.probplot(train_df['SalePrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot of Sale Price')

plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
print("\nSaved: target_distribution.png")

# Correlation analysis
numeric_features = train_df.select_dtypes(include=[np.number]).columns
correlations = train_df[numeric_features].corr()['SalePrice'].sort_values(ascending=False)

print("\nTop 10 features correlated with SalePrice:")
print(correlations.head(11))  # 11 to include SalePrice itself

# Correlation heatmap for top features
plt.figure(figsize=(12, 10))
top_features = correlations.head(11).index
sns.heatmap(train_df[top_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1)
plt.title('Correlation Heatmap - Top Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: correlation_heatmap.png")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("3. FEATURE ENGINEERING")
print("="*80)

# Save test IDs for submission
test_ids = test_df['Id']

# Combine train and test for consistent feature engineering
y_train = train_df['SalePrice'].copy()
train_df = train_df.drop(['SalePrice'], axis=1)
combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

print(f"Combined dataset shape: {combined.shape}")

# Drop Id column
combined = combined.drop(['Id'], axis=1)

# Handle missing values
print("\nHandling missing values...")

# Features where missing means "None"
none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
             'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 
             'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

for col in none_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna('None')

# Features where missing means 0
zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
             'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
             'BsmtHalfBath', 'MasVnrArea']

for col in zero_cols:
    if col in combined.columns:
        combined[col] = combined[col].fillna(0)

# LotFrontage: fill with median by neighborhood
combined['LotFrontage'] = combined.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median()))

# Fill remaining numerical with median
num_cols = combined.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if combined[col].isnull().sum() > 0:
        combined[col] = combined[col].fillna(combined[col].median())

# Fill remaining categorical with mode
cat_cols = combined.select_dtypes(include=['object']).columns
for col in cat_cols:
    if combined[col].isnull().sum() > 0:
        combined[col] = combined[col].fillna(combined[col].mode()[0])

print(f"Missing values after imputation: {combined.isnull().sum().sum()}")

# Create new features
print("\nCreating new features...")

# Total square footage
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']

# Total bathrooms
combined['TotalBath'] = (combined['FullBath'] + 0.5 * combined['HalfBath'] + 
                         combined['BsmtFullBath'] + 0.5 * combined['BsmtHalfBath'])

# Total porch area
combined['TotalPorchSF'] = (combined['OpenPorchSF'] + combined['3SsnPorch'] + 
                            combined['EnclosedPorch'] + combined['ScreenPorch'] + 
                            combined['WoodDeckSF'])

# House age and remodel age
combined['HouseAge'] = 2026 - combined['YearBuilt']
combined['RemodAge'] = 2026 - combined['YearRemodAdd']

# Has pool, garage, basement, fireplace
combined['HasPool'] = (combined['PoolArea'] > 0).astype(int)
combined['HasGarage'] = (combined['GarageArea'] > 0).astype(int)
combined['HasBsmt'] = (combined['TotalBsmtSF'] > 0).astype(int)
combined['HasFireplace'] = (combined['Fireplaces'] > 0).astype(int)

# Quality * Area features
combined['OverallQual_GrLivArea'] = combined['OverallQual'] * combined['GrLivArea']
combined['OverallQual_TotalSF'] = combined['OverallQual'] * combined['TotalSF']

print(f"New features created. Current shape: {combined.shape}")

# Encode categorical variables
print("\nEncoding categorical variables...")

# Get categorical columns
categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()

# One-hot encoding
combined = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)

print(f"After encoding - Shape: {combined.shape}")

# Split back to train and test
X_train = combined[:len(train_df)].copy()
X_test = combined[len(train_df):].copy()

print(f"\nFinal X_train shape: {X_train.shape}")
print(f"Final X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")

# Log transform target variable (improves performance for house prices)
y_train_log = np.log1p(y_train)

# Split for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_log, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train_split.shape}")
print(f"Validation set: {X_val.shape}")

# ============================================================================
# 4. LINEAR REGRESSION MODELS
# ============================================================================
print("\n" + "="*80)
print("4. LINEAR REGRESSION MODELS")
print("="*80)

# Store results
results = {}

# Scale features for regularized models
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)

# 4.1 Linear Regression
print("\n4.1 Linear Regression")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train_split)
y_pred_lr = lr.predict(X_val_scaled)

rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
mae_lr = mean_absolute_error(y_val, y_pred_lr)
r2_lr = r2_score(y_val, y_pred_lr)

results['Linear Regression'] = {'RMSE': rmse_lr, 'MAE': mae_lr, 'R2': r2_lr}
print(f"RMSE: {rmse_lr:.4f}, MAE: {mae_lr:.4f}, R²: {r2_lr:.4f}")

# 4.2 Ridge Regression
print("\n4.2 Ridge Regression (L2 Regularization)")
ridge = Ridge(alpha=10.0, random_state=42)
ridge.fit(X_train_scaled, y_train_split)
y_pred_ridge = ridge.predict(X_val_scaled)

rmse_ridge = np.sqrt(mean_squared_error(y_val, y_pred_ridge))
mae_ridge = mean_absolute_error(y_val, y_pred_ridge)
r2_ridge = r2_score(y_val, y_pred_ridge)

results['Ridge'] = {'RMSE': rmse_ridge, 'MAE': mae_ridge, 'R2': r2_ridge}
print(f"RMSE: {rmse_ridge:.4f}, MAE: {mae_ridge:.4f}, R²: {r2_ridge:.4f}")

# 4.3 Lasso Regression
print("\n4.3 Lasso Regression (L1 Regularization)")
lasso = Lasso(alpha=0.001, random_state=42, max_iter=10000)
lasso.fit(X_train_scaled, y_train_split)
y_pred_lasso = lasso.predict(X_val_scaled)

rmse_lasso = np.sqrt(mean_squared_error(y_val, y_pred_lasso))
mae_lasso = mean_absolute_error(y_val, y_pred_lasso)
r2_lasso = r2_score(y_val, y_pred_lasso)

results['Lasso'] = {'RMSE': rmse_lasso, 'MAE': mae_lasso, 'R2': r2_lasso}
print(f"RMSE: {rmse_lasso:.4f}, MAE: {mae_lasso:.4f}, R²: {r2_lasso:.4f}")

# Feature importance from Lasso
lasso_coef = pd.Series(lasso.coef_, index=X_train.columns)
important_features = lasso_coef[lasso_coef != 0].sort_values(ascending=False)
print(f"\nNumber of features selected by Lasso: {len(important_features)}")

# 4.4 ElasticNet Regression
print("\n4.4 ElasticNet Regression (L1 + L2 Regularization)")
elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=10000)
elastic.fit(X_train_scaled, y_train_split)
y_pred_elastic = elastic.predict(X_val_scaled)

rmse_elastic = np.sqrt(mean_squared_error(y_val, y_pred_elastic))
mae_elastic = mean_absolute_error(y_val, y_pred_elastic)
r2_elastic = r2_score(y_val, y_pred_elastic)

results['ElasticNet'] = {'RMSE': rmse_elastic, 'MAE': mae_elastic, 'R2': r2_elastic}
print(f"RMSE: {rmse_elastic:.4f}, MAE: {mae_elastic:.4f}, R²: {r2_elastic:.4f}")

# ============================================================================
# 5. TREE-BASED MODELS
# ============================================================================
print("\n" + "="*80)
print("5. TREE-BASED MODELS")
print("="*80)

# Use unscaled data for tree models
# 5.1 Decision Tree (CART)
print("\n5.1 Decision Tree Regressor (CART)")
dt = DecisionTreeRegressor(max_depth=10, min_samples_split=20, 
                           min_samples_leaf=10, random_state=42)
dt.fit(X_train_split, y_train_split)
y_pred_dt = dt.predict(X_val)

rmse_dt = np.sqrt(mean_squared_error(y_val, y_pred_dt))
mae_dt = mean_absolute_error(y_val, y_pred_dt)
r2_dt = r2_score(y_val, y_pred_dt)

results['Decision Tree'] = {'RMSE': rmse_dt, 'MAE': mae_dt, 'R2': r2_dt}
print(f"RMSE: {rmse_dt:.4f}, MAE: {mae_dt:.4f}, R²: {r2_dt:.4f}")

# 5.2 Random Forest
print("\n5.2 Random Forest Regressor")
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=10,
                           min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train_split, y_train_split)
y_pred_rf = rf.predict(X_val)

rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
mae_rf = mean_absolute_error(y_val, y_pred_rf)
r2_rf = r2_score(y_val, y_pred_rf)

results['Random Forest'] = {'RMSE': rmse_rf, 'MAE': mae_rf, 'R2': r2_rf}
print(f"RMSE: {rmse_rf:.4f}, MAE: {mae_rf:.4f}, R²: {r2_rf:.4f}")

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(feature_importance.head(10))

# 5.3 Gradient Boosting
print("\n5.3 Gradient Boosting Regressor")
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                               min_samples_split=10, min_samples_leaf=5,
                               random_state=42)
gb.fit(X_train_split, y_train_split)
y_pred_gb = gb.predict(X_val)

rmse_gb = np.sqrt(mean_squared_error(y_val, y_pred_gb))
mae_gb = mean_absolute_error(y_val, y_pred_gb)
r2_gb = r2_score(y_val, y_pred_gb)

results['Gradient Boosting'] = {'RMSE': rmse_gb, 'MAE': mae_gb, 'R2': r2_gb}
print(f"RMSE: {rmse_gb:.4f}, MAE: {mae_gb:.4f}, R²: {r2_gb:.4f}")

# 5.4 XGBoost
print("\n5.4 XGBoost Regressor")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                   random_state=42, n_jobs=-1)
xgb.fit(X_train_split, y_train_split)
y_pred_xgb = xgb.predict(X_val)

rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
r2_xgb = r2_score(y_val, y_pred_xgb)

results['XGBoost'] = {'RMSE': rmse_xgb, 'MAE': mae_xgb, 'R2': r2_xgb}
print(f"RMSE: {rmse_xgb:.4f}, MAE: {mae_xgb:.4f}, R²: {r2_xgb:.4f}")

# ============================================================================
# 6. PERFORMANCE METRICS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("6. PERFORMANCE METRICS COMPARISON")
print("="*80)

# Create results dataframe
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('RMSE')

print("\nModel Performance Summary (sorted by RMSE):")
print(results_df.to_string())

# Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE comparison
axes[0].barh(results_df.index, results_df['RMSE'], color='steelblue')
axes[0].set_xlabel('RMSE (Lower is Better)')
axes[0].set_title('Root Mean Squared Error Comparison')
axes[0].invert_yaxis()

# MAE comparison
axes[1].barh(results_df.index, results_df['MAE'], color='coral')
axes[1].set_xlabel('MAE (Lower is Better)')
axes[1].set_title('Mean Absolute Error Comparison')
axes[1].invert_yaxis()

# R² comparison
axes[2].barh(results_df.index, results_df['R2'], color='green')
axes[2].set_xlabel('R² Score (Higher is Better)')
axes[2].set_title('R² Score Comparison')
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: model_comparison.png")

# ============================================================================
# 7. FINAL MODEL AND PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("7. FINAL MODEL AND PREDICTIONS")
print("="*80)

# Select best model based on RMSE
best_model_name = results_df.index[0]
print(f"\nBest Model: {best_model_name}")

# Train final model on full training data
print("\nTraining final model on full training set...")

if best_model_name == 'XGBoost':
    final_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                               random_state=42, n_jobs=-1)
elif best_model_name == 'Gradient Boosting':
    final_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, 
                                           max_depth=5, random_state=42)
elif best_model_name == 'Random Forest':
    final_model = RandomForestRegressor(n_estimators=200, max_depth=15, 
                                       random_state=42, n_jobs=-1)
else:
    # Use Ridge as default for linear models
    final_model = Ridge(alpha=10.0, random_state=42)
    X_train_final = scaler.fit_transform(X_train)
    X_test_final = scaler.transform(X_test)
    final_model.fit(X_train_final, y_train_log)
    y_pred_test = final_model.predict(X_test_final)

# For tree-based models, no scaling needed
if best_model_name in ['XGBoost', 'Gradient Boosting', 'Random Forest', 'Decision Tree']:
    final_model.fit(X_train, y_train_log)
    y_pred_test = final_model.predict(X_test)

# Transform predictions back to original scale
y_pred_test_original = np.expm1(y_pred_test)

# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred_test_original
})

submission.to_csv('submission.csv', index=False)
print(f"\nSubmission file created: submission.csv")
print(f"Shape: {submission.shape}")
print("\nFirst few predictions:")
print(submission.head(10))

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("1. target_distribution.png - Target variable analysis")
print("2. correlation_heatmap.png - Feature correlation analysis")
print("3. model_comparison.png - Model performance comparison")
print("4. submission.csv - Final predictions for Kaggle submission")
print("\nKey Insights:")
print(f"- Best performing model: {best_model_name}")
print(f"- Best validation RMSE: {results_df['RMSE'].iloc[0]:.4f}")
print(f"- Best validation R²: {results_df['R2'].iloc[0]:.4f}")
print(f"- Total features engineered: {X_train.shape[1]}")
print("\nNext steps:")
print("1. Review the visualizations to understand model performance")
print("2. Upload submission.csv to Kaggle")
print("3. Consider hyperparameter tuning for further improvement")
print("4. Experiment with feature selection and ensemble methods")
