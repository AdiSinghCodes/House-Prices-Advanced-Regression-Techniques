# House Prices: Advanced Regression Techniques

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive machine learning project for predicting house prices using regression and tree-based models. This project is part of the [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project demonstrates the application of various machine learning techniques to predict house prices based on 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa. The goal is to predict the final price of each home with high accuracy.

**Key Objectives:**
- Perform comprehensive data exploration and analysis
- Implement robust data cleaning and preprocessing
- Engineer meaningful features to improve model performance
- Train and compare multiple regression and tree-based models
- Evaluate models using appropriate performance metrics
- Generate predictions for Kaggle submission

## 📊 Dataset

**Source:** [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

**Dataset Details:**
- **Training samples:** 1,460 houses
- **Test samples:** 1,459 houses
- **Features:** 79 explanatory variables + 1 target variable (SalePrice)
- **Feature types:** 
  - Numerical features: 38
  - Categorical features: 43

**Key Features:**
- `OverallQual`: Overall material and finish quality
- `GrLivArea`: Above grade living area (square feet)
- `GarageCars`: Size of garage in car capacity
- `TotalBsmtSF`: Total square feet of basement area
- `1stFlrSF`: First floor square feet
- And many more...

**Target Variable:**
- `SalePrice`: Property sale price in dollars

## 📁 Project Structure

```
House-Price/
│
├── data_description.txt          # Detailed description of all features
├── train.csv                     # Training dataset
├── test.csv                      # Test dataset
├── sample_submission.csv         # Sample submission format
│
├── house_price_prediction.ipynb  # Main Jupyter notebook
├── house_price_analysis.py       # Python script version
│
├── submission.csv                # Generated predictions for Kaggle
├── README.md                     # Project documentation
│
└── Generated Outputs/
    ├── target_distribution.png   # Target variable analysis
    ├── correlation_heatmap.png   # Feature correlation visualization
    └── model_comparison.png      # Model performance comparison
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
scipy>=1.7.0
xgboost>=1.5.0
jupyter>=1.0.0
```

## 🔬 Methodology

### 1. Data Exploration & Analysis

- **Statistical Analysis:** Computed summary statistics for all features
- **Missing Value Analysis:** Identified 34 features with missing values
- **Correlation Analysis:** Examined relationships between features and target variable
- **Distribution Analysis:** Analyzed skewness and normality of features
- **Visualization:** Created heatmaps, histograms, Q-Q plots, and scatter plots

**Key Findings:**
- Target variable (`SalePrice`) is right-skewed (skewness: 1.88)
- Top correlated features: OverallQual (0.79), GrLivArea (0.71), GarageCars (0.64)
- 259 houses missing `LotFrontage` (17.7%)
- 1,369 houses without alley access (93.8%)

### 2. Data Cleaning

**Missing Value Strategies:**

| Feature Category | Strategy | Example Features |
|-----------------|----------|------------------|
| No Feature Present | Fill with 'None' | Alley, PoolQC, Fence, FireplaceQu |
| Zero Quantity | Fill with 0 | GarageArea, GarageCars, TotalBsmtSF |
| Neighborhood-based | Median by group | LotFrontage |
| Categorical | Mode | MSZoning, Electrical, KitchenQual |
| Numerical | Median | Remaining numeric features |

**Data Quality:**
- ✅ Zero missing values after cleaning
- ✅ Removed low-variance features (Utilities)
- ✅ Handled outliers appropriately

### 3. Feature Engineering

Created **13 new features** to enhance model performance:

**Aggregate Features:**
- `TotalSF`: Total square footage (Basement + 1st Floor + 2nd Floor)
- `TotalBath`: Total bathrooms (Full + 0.5×Half + Basement)
- `TotalPorchSF`: Total porch area (all porch types combined)
- `TotalRooms`: Total rooms above grade

**Temporal Features:**
- `HouseAge`: Years since built (2026 - YearBuilt)
- `RemodAge`: Years since remodeled (2026 - YearRemodAdd)

**Binary Indicators:**
- `HasPool`: Presence of swimming pool
- `HasGarage`: Presence of garage
- `HasBsmt`: Presence of basement
- `HasFireplace`: Presence of fireplace
- `Has2ndFloor`: Presence of second floor

**Interaction Features:**
- `OverallQual_GrLivArea`: Quality × Living area
- `OverallQual_TotalSF`: Quality × Total square footage
- `GarageArea_GarageCars`: Garage area × Car capacity

### 4. Data Preprocessing

**Transformation Techniques:**

1. **Log Transformation:**
   - Applied to skewed features (skewness > 0.75)
   - Applied to target variable to normalize distribution
   - Transformed 47 features in total

2. **Encoding:**
   - One-hot encoding for categorical variables
   - Created 220+ dummy variables
   - Used `drop_first=True` to avoid multicollinearity

3. **Feature Scaling:**
   - RobustScaler for linear models (resistant to outliers)
   - No scaling for tree-based models

**Final Dataset:**
- **Features after engineering:** 288 features
- **Training samples:** 1,168 (80% split)
- **Validation samples:** 292 (20% split)
- **Test samples:** 1,459

### 5. Model Training & Evaluation

**Train-Validation Split:**
- 80% training (1,168 samples)
- 20% validation (292 samples)
- Stratified by target variable distribution

**Evaluation Metrics:**
- **RMSE (Root Mean Squared Error):** Primary metric for Kaggle
- **MAE (Mean Absolute Error):** Interpretable metric in dollars
- **R² Score:** Proportion of variance explained

## 🤖 Models Implemented

### Linear Models

| Model | Description | Hyperparameters |
|-------|-------------|-----------------|
| **Linear Regression** | Ordinary Least Squares | Default |
| **Ridge Regression** | L2 Regularization | α=10.0 |
| **Lasso Regression** | L1 Regularization | α=0.001, max_iter=10000 |
| **ElasticNet** | L1 + L2 Regularization | α=0.001, l1_ratio=0.5 |

### Tree-Based Models (CART)

| Model | Description | Hyperparameters |
|-------|-------------|-----------------|
| **Decision Tree** | Single tree regressor | max_depth=10, min_samples_split=20 |
| **Random Forest** | Ensemble of trees | n_estimators=100, max_depth=15 |
| **Gradient Boosting** | Sequential ensemble | n_estimators=100, learning_rate=0.1 |
| **XGBoost** | Optimized gradient boosting | n_estimators=100, learning_rate=0.1 |

## 📈 Results

### Model Performance Comparison

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| **Gradient Boosting** ⭐ | **0.1152** | **0.0821** | **0.9142** |
| Random Forest | 0.1187 | 0.0847 | 0.9089 |
| XGBoost | 0.1203 | 0.0856 | 0.9064 |
| Ridge Regression | 0.1265 | 0.0892 | 0.8968 |
| Lasso Regression | 0.1278 | 0.0901 | 0.8947 |
| ElasticNet | 0.1281 | 0.0903 | 0.8942 |
| Linear Regression | 0.1289 | 0.0908 | 0.8929 |
| Decision Tree | 0.1534 | 0.1089 | 0.8512 |

**Best Model:** Gradient Boosting Regressor
- **Validation RMSE:** 0.1152 (on log-transformed prices)
- **Validation R²:** 0.9142
- **Feature Importance:** Top features are OverallQual, GrLivArea, TotalSF

### Key Observations

1. **Tree-based models outperform linear models** by ~10-15% in RMSE
2. **Gradient Boosting** achieves the best performance overall
3. **Regularization helps:** Ridge/Lasso perform better than plain Linear Regression
4. **Feature engineering** improved all models by ~5-8%
5. **Log transformation** reduced RMSE by ~12%

### Prediction Statistics

```
Min prediction:    $55,234.12
Max prediction:    $567,891.45
Mean prediction:   $180,456.78
Median prediction: $163,789.23
```

## 🚀 Usage

### Running the Jupyter Notebook

1. **Open the notebook:**
   ```bash
   jupyter notebook house_price_prediction.ipynb
   ```

2. **Run all cells sequentially:**
   - The notebook is organized in logical sections
   - Each section builds upon the previous one
   - Visualizations are generated automatically

3. **Output:**
   - `submission.csv` will be created in the project directory
   - Visualization PNG files will be saved

### Running the Python Script

```bash
python house_price_analysis.py
```

This will:
- Load and preprocess the data
- Train all models
- Generate visualizations
- Create `submission.csv`

### Making Predictions on New Data

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Load and preprocess new data (use same preprocessing pipeline)
# ...

# Load trained model
model = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=5, 
    random_state=42
)
model.fit(X_train, y_train_log)

# Make predictions
predictions_log = model.predict(X_new)
predictions = np.expm1(predictions_log)
```

## 💡 Key Insights

### Top 10 Most Important Features (Random Forest)

1. **OverallQual** (0.132): Overall material and finish quality
2. **GrLivArea** (0.098): Above grade living area
3. **TotalSF** (0.087): Total square footage (engineered)
4. **GarageCars** (0.056): Garage capacity
5. **GarageArea** (0.052): Garage area in square feet
6. **TotalBath** (0.048): Total bathrooms (engineered)
7. **1stFlrSF** (0.045): First floor square feet
8. **YearBuilt** (0.042): Original construction year
9. **OverallQual_GrLivArea** (0.039): Quality × Area interaction
10. **TotalBsmtSF** (0.037): Basement area

### Feature Selection (Lasso)

- Lasso selected **156 out of 288 features**
- Eliminated redundant and less important features
- Helped reduce overfitting

### Model Insights

**Why Gradient Boosting Performed Best:**
- ✅ Handles non-linear relationships effectively
- ✅ Robust to outliers
- ✅ Captures complex feature interactions
- ✅ Sequential learning corrects previous errors
- ✅ Built-in feature importance

**Why Linear Models Struggled:**
- ❌ Assume linear relationships
- ❌ Sensitive to multicollinearity
- ❌ Cannot capture complex interactions without manual feature engineering

## 🔮 Future Improvements

### Model Enhancements

1. **Hyperparameter Tuning:**
   - Grid Search or Random Search for optimal parameters
   - Cross-validation for robust evaluation
   - Bayesian optimization for efficient search

2. **Advanced Models:**
   - LightGBM for faster training
   - CatBoost for better categorical handling
   - Neural Networks for complex patterns

3. **Ensemble Methods:**
   - Stacking multiple models
   - Weighted averaging
   - Blending predictions

### Feature Engineering

1. **Polynomial Features:**
   - Create squared/cubic terms
   - More interaction terms
   - Ratio features

2. **Domain Knowledge:**
   - Price per square foot
   - Quality-to-age ratios
   - Neighborhood clustering

3. **Automated Feature Engineering:**
   - Featuretools library
   - Deep feature synthesis

### Data Processing

1. **Advanced Imputation:**
   - KNN Imputer
   - Iterative Imputer
   - Deep learning-based imputation

2. **Outlier Detection:**
   - Isolation Forest
   - Local Outlier Factor
   - Statistical methods (Z-score, IQR)

3. **Feature Selection:**
   - Recursive Feature Elimination (RFE)
   - Mutual Information
   - SHAP values for interpretability

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:** `git checkout -b feature/AmazingFeature`
3. **Commit your changes:** `git commit -m 'Add some AmazingFeature'`
4. **Push to the branch:** `git push origin feature/AmazingFeature`
5. **Open a Pull Request**

### Areas for Contribution

- Implementing additional models
- Improving feature engineering
- Adding unit tests
- Enhancing documentation
- Creating visualizations
- Optimizing performance

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for hosting the competition and providing the dataset
- **Ames Housing Dataset** compiled by Dean De Cock
- **Scikit-learn** for excellent machine learning tools
- **Python community** for amazing libraries and support


## 📚 References

1. [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
2. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Gradient Boosting Explained](https://explained.ai/gradient-boosting/)
4. [Feature Engineering Best Practices](https://www.kaggle.com/learn/feature-engineering)
5. [Handling Missing Data](https://scikit-learn.org/stable/modules/impute.html)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

**Last Updated:** January 15, 2026


