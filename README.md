# Laptop Price Predictor - Machine Learning Project

A comprehensive machine learning project that predicts laptop prices using various regression algorithms. This project demonstrates end-to-end data science workflow including data preprocessing, feature engineering, and model development.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Project Workflow](#project-workflow)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
7. [Model Comparison](#model-comparison)
8. [Overfitting Reduction Techniques](#overfitting-reduction-techniques)
9. [Output & Deployment](#output--deployment)
10. [Libraries Used](#libraries-used)

---

## 🎯 Project Overview

This project builds a predictive model to estimate laptop prices based on hardware specifications and features. The goal is to develop an accurate regression model that can predict laptop prices using various machine learning algorithms and ensemble techniques.

---

## 📊 Dataset Information

### Dataset Shape
- **Total Records:** 1,303 laptops
- **Total Features (Initial):** 13 columns
- **Target Variable:** Price

### Initial Dataset Columns
- Unnamed: 0 (Index column - removed)
- Company
- TypeName
- Inches
- ScreenResolution
- Cpu
- Ram
- Memory
- Gpu
- OpSys
- Weight
- Price

### Data Quality Checks

| Check | Result |
|-------|--------|
| Duplicate Records | 0 duplicates found |
| Missing Values | No missing values |
| Data Types | Mixed (strings, integers, floats) |

---

## 🔄 Project Workflow

```
Data Loading
    ↓
Exploratory Data Analysis (EDA)
    ↓
Data Preprocessing & Cleaning
    ↓
Feature Engineering
    ↓
Train-Test Split
    ↓
Model Training & Evaluation
    ↓
Model Comparison & Selection
    ↓
Model Export & Deployment
```

---

## 🛠️ Data Preprocessing

### Step 1: Remove Unnecessary Columns
```python
df.drop(columns=['Unnamed: 0'], inplace=True)
```
- Removed the auto-generated index column that was not needed for analysis

### Step 2: Remove Unit Suffixes from Numeric Columns
```python
df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')
```
- **Ram:** Removed 'GB' suffix (e.g., "8GB" → "8")
- **Weight:** Removed 'kg' suffix (e.g., "2.3kg" → "2.3")

### Step 3: Convert Data Types
```python
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')
```
- **Ram:** Converted to integer (int32)
- **Weight:** Converted to float (float32)

### Summary of Preprocessing
- ✅ Removed 1 unnecessary column
- ✅ Standardized unit measurements
- ✅ Converted string values to numeric types
- ✅ No missing values to handle
- ✅ No duplicate records to remove

---

## 🔧 Feature Engineering

### 1. Screen Resolution Analysis & Feature Extraction

#### Touchscreen Feature
```python
df['Touchscreen'] = df['ScreenResolution'].apply(
    lambda x: 1 if 'Touchscreen' in x else 0
)
```
- Binary feature: 1 if laptop has touchscreen, 0 otherwise

#### IPS Display Feature
```python
df['Ips'] = df['ScreenResolution'].apply(
    lambda x: 1 if 'IPS' in x else 0
)
```
- Binary feature: 1 if display is IPS type, 0 otherwise

#### Screen Resolution Parsing
```python
new = df['ScreenResolution'].str.split('x', n=1, expand=True)
df['X_res'] = new[0]  # Horizontal resolution
df['Y_res'] = new[1]  # Vertical resolution
```
- Split resolution into X and Y components
- Example: "1920x1080" → X_res=1920, Y_res=1080

#### Resolution Cleaning
- Removed non-numeric characters (commas, text)
- Extracted numeric values using regex: `r'(\d+\.?\d+)'`
- Converted to integer type

#### Pixels Per Inch (PPI) Calculation
```python
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype('float')
```
- **Formula:** PPI = $\sqrt{X_{res}^2 + Y_{res}^2} / \text{Inches}$
- Measures screen pixel density
- Higher PPI indicates sharper display

#### Screen Features Cleanup
- Dropped original ScreenResolution column
- Removed intermediate resolution columns (Inches, X_res, Y_res)
- Retained engineered features: Touchscreen, Ips, ppi

---

### 2. CPU (Processor) Feature Engineering

#### CPU Name Extraction
```python
df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
```
- Extracted first 3 words from CPU string
- Example: "Intel Core i7 8th Gen" → "Intel Core i7"

#### CPU Brand Categorization
```python
def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'

df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
```
- **Categories:** Intel Core i7, Intel Core i5, Intel Core i3, Other Intel Processor, AMD Processor
- Captures processor tier which significantly affects price

#### CPU Features Cleanup
- Dropped original Cpu column
- Dropped intermediate Cpu Name column
- Retained categorical feature: Cpu brand

---

### 3. RAM Feature Engineering
- Analyzed RAM distribution and correlation with price
- Converted to numeric format during preprocessing
- Retained as-is due to strong price correlation

---

### 4. Memory Storage Feature Engineering

#### Complex Memory Parsing
Multiple storage configurations parsed:
- Single storage: "512GB SSD"
- Combined storage: "256GB SSD + 1TB HDD"
- Multiple types: HDD, SSD, Hybrid, Flash Storage

#### Storage Type Detection
```python
df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)
```
- Created 8 binary features (Layer1 and Layer2 for each type)

#### Storage Capacity Calculation
```python
df['HDD'] = (df['first'] * df['Layer1HDD'] + df['second'] * df['Layer2HDD'])
df['SSD'] = (df['first'] * df['Layer1SSD'] + df['second'] * df['Layer2SSD'])
df['Hybrid'] = (df['first'] * df['Layer1Hybrid'] + df['second'] * df['Layer2Hybrid'])
df['Flash_Storage'] = (df['first'] * df['Layer1Flash_Storage'] + df['second'] * df['Layer2Flash_Storage'])
```
- Combined storage from primary and secondary layers
- Each storage type gets separate feature column (in GB)

#### Memory Features Cleanup
- Dropped intermediate columns (Layer1HDD, Layer1SSD, etc.)
- Dropped weakly correlated features: Hybrid, Flash_Storage
- Retained: HDD, SSD (most relevant storage types)

---

### 5. GPU (Graphics Card) Feature Engineering

#### GPU Brand Extraction
```python
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
```
- Extracted GPU manufacturer (NVIDIA, AMD, Intel, ARM)

#### GPU Brand Filtering
```python
df = df[df['Gpu brand'] != 'ARM']
```
- Removed ARM processors (integrated graphics, not discrete GPUs)
- Only retained discrete GPU options

#### GPU Features Used
- **Categories:** NVIDIA, AMD, Intel (integrated graphics)
- GPU type affects performance and price

---

### 6. Operating System Feature Engineering

#### OS Categorization
```python
def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)
```
- **Categories:** Windows, Mac, Others/No OS/Linux
- Simplified OS feature by grouping similar versions
- Mac OS typically commands premium prices

#### OS Features Cleanup
- Dropped original OpSys column
- Retained categorical feature: os

---

### 7. Weight Feature
- Converted to float32 during preprocessing
- Analyzed correlation with price
- Highly correlated with laptop size and performance

---

## 📈 Exploratory Data Analysis

### Visualizations Created

| Feature | Analysis |
|---------|----------|
| Price | Distribution plot (distplot) - understands price spread |
| Company | Bar plot - identifies price variations by manufacturer |
| TypeName | Price by laptop type (Ultrabook, Notebook, Gaming, etc.) |
| Inches | Scatter plot vs Price - screen size impact |
| RAM | Bar plot - RAM impact on pricing |
| Touchscreen | Bar plot - touchscreen feature impact |
| IPS | Bar plot - IPS display impact on price |
| CPU Brand | Bar plot with median estimation |
| GPU Brand | Bar plot with median aggregation |
| Operating System | Bar plot - OS impact on price |
| Weight | Scatter plot and distribution - weight vs price |
| Correlation Matrix | Heatmap - feature relationships with price |

### Log Transformation
```python
y = np.log(df['Price'])
```
- Applied natural logarithm to price for normalization
- Reduces skewness in price distribution
- Improves model performance

---

## 📊 Final Dataset Features

After feature engineering, the final dataset contains:

| Feature | Type | Description |
|---------|------|-------------|
| Company | Categorical | Laptop manufacturer |
| TypeName | Categorical | Laptop type (Ultrabook, Notebook, Gaming, etc.) |
| Ram | Numeric | RAM in GB |
| Touchscreen | Binary | 1 if has touchscreen, 0 otherwise |
| Ips | Binary | 1 if IPS display, 0 otherwise |
| ppi | Numeric | Pixels per inch (screen quality) |
| Cpu brand | Categorical | Processor type (i7, i5, i3, etc.) |
| HDD | Numeric | HDD storage in GB |
| SSD | Numeric | SSD storage in GB |
| Gpu brand | Categorical | GPU manufacturer |
| os | Categorical | Operating system type |
| Weight | Numeric | Weight in kg |
| Price | Numeric | Target variable (log-transformed) |

---

## 🔀 Data Splitting

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
```

- **Training Set:** 85% (1,107 samples)
- **Test Set:** 15% (196 samples)
- **Random State:** 2 (for reproducibility)
- **Target:** Log-transformed Price

---

## 🤖 Model Development

### Preprocessing Pipeline

```python
step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse=False, drop='first'), [0,1,7,10,11])
], remainder='passthrough')
```

**One-Hot Encoded Categorical Features:**
- Index 0: Company
- Index 1: TypeName
- Index 7: Cpu brand
- Index 10: Gpu brand
- Index 11: os

**Passed Through (Numeric):**
- Ram, Touchscreen, Ips, ppi, HDD, SSD, Weight

---

### Models Implemented

#### 1. Linear Regression
```python
step2 = LinearRegression()
```
- **Use Case:** Baseline model for comparison
- **Characteristics:** Fast, interpretable, assumes linear relationships
- **Expected Performance:** Good for simple patterns

#### 2. Ridge Regression
```python
step2 = Ridge(alpha=10)
```
- **Use Case:** Prevents overfitting through L2 regularization
- **Alpha Value:** 10 (balances bias-variance tradeoff)
- **Advantage:** Better generalization than linear regression

#### 3. Lasso Regression
```python
step2 = Lasso(alpha=0.001)
```
- **Use Case:** Feature selection through L1 regularization
- **Alpha Value:** 0.001 (very low for minimal feature elimination)
- **Advantage:** Automatically performs feature selection

#### 4. K-Nearest Neighbors (KNN)
```python
step2 = KNeighborsRegressor(n_neighbors=3)
```
- **Use Case:** Non-parametric approach based on local neighborhoods
- **n_neighbors:** 3 (uses 3 nearest neighbors)
- **Advantage:** Can capture non-linear patterns

#### 5. Decision Tree Regressor
```python
step2 = DecisionTreeRegressor(max_depth=8)
```
- **Use Case:** Tree-based model with interpretability
- **max_depth:** 8 (controls complexity)
- **Advantage:** Captures non-linear relationships, easy to interpret

#### 6. Support Vector Regression (SVM)
```python
step2 = SVR(kernel='rbf', C=10000, epsilon=0.1)
```
- **Kernel:** RBF (Radial Basis Function) - non-linear
- **C:** 10000 (regularization parameter - high for tight fit)
- **Epsilon:** 0.1 (tolerance for errors)
- **Advantage:** Effective in high-dimensional spaces

#### 7. Random Forest Regressor
```python
step2 = RandomForestRegressor(
    n_estimators=100,
    random_state=3,
    max_samples=0.5,
    max_features=0.75,
    max_depth=15
)
```
- **n_estimators:** 100 trees
- **max_samples:** 0.5 (50% of samples per tree)
- **max_features:** 0.75 (75% of features per split)
- **max_depth:** 15 (tree depth control)
- **Advantage:** Ensemble method reduces overfitting

#### 8. Extra Trees (Extremely Randomized Trees)
```python
step2 = ExtraTreesRegressor(
    n_estimators=100,
    random_state=3,
    max_samples=0.5,
    max_features=0.75,
    max_depth=15
)
```
- **Similar to Random Forest** but with random split thresholds
- **Advantage:** Faster training, good for high-dimensional data

#### 9. AdaBoost Regressor
```python
step2 = AdaBoostRegressor(n_estimators=15, learning_rate=1.0)
```
- **n_estimators:** 15 boosting iterations
- **learning_rate:** 1.0 (full contribution of each model)
- **Advantage:** Sequentially improves weak learners

#### 10. Gradient Boosting Regressor
```python
step2 = GradientBoostingRegressor(n_estimators=500)
```
- **n_estimators:** 500 boosting stages
- **Advantage:** Sequential error correction, often achieves best results

#### 11. XGBoost Regressor
```python
step2 = XGBRegressor(n_estimators=45, max_depth=5, learning_rate=0.5)
```
- **n_estimators:** 45 boosting rounds
- **max_depth:** 5 (shallower trees, prevent overfitting)
- **learning_rate:** 0.5 (moderate learning rate)
- **Advantage:** Fast, efficient, regularized boosting

---

## 🚀 Overfitting Reduction Techniques

### 1. Regularization

**Ridge Regression (L2 Regularization)**
- Adds penalty term: $\lambda \sum \beta_i^2$
- Reduces coefficient magnitude
- Prevents extreme parameter values

**Lasso Regression (L1 Regularization)**
- Adds penalty term: $\lambda \sum |\beta_i|$
- Performs automatic feature selection
- Some coefficients become exactly zero

### 2. Tree-Based Constraints

**Decision Tree:**
- `max_depth=8` - Limits tree depth
- Prevents memorization of training data

**Random Forest:**
- `max_samples=0.5` - Bootstrap with 50% of samples
- `max_features=0.75` - Use 75% of features at each split
- Ensemble averaging reduces overfitting
- Multiple random trees with averaging

**Extra Trees:**
- Random split thresholds (vs optimal splits)
- Additional randomization reduces overfitting

**AdaBoost:**
- Limited weak learners (n_estimators=15)
- Focuses on misclassified samples
- Sequential learning prevents convergence

### 3. Ensemble Methods

**Voting Regressor**
```python
step2 = VotingRegressor([
    ('rf', RandomForestRegressor(...)),
    ('gbdt', GradientBoostingRegressor(...)),
    ('xgb', XGBRegressor(...)),
    ('et', ExtraTreesRegressor(...))
], weights=[5,1,1,1])
```
- Combines 4 diverse models
- Weighted averaging (RF gets weight=5)
- Reduces variance through ensemble

**Stacking Regressor**
```python
estimators = [
    ('rf', RandomForestRegressor(...)),
    ('gbdt', GradientBoostingRegressor(...)),
    ('xgb', XGBRegressor(...))
]
step2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=100))
```
- Meta-learner approach:
  - Base models: RF, GBDT, XGB
  - Final estimator: Ridge regression
- Learns optimal combination of base models
- Captures complex relationships

### 4. Hyperparameter Tuning

- **Learning rates:** Controlled in boosting models
- **Tree depth:** Limited to prevent memorization
- **Alpha values:** Regularization strength
- **Number of estimators:** Balanced for training time

### 5. Data Splitting

- **Train-Test Split:** 85-15 ensures model validation on unseen data
- **Random State:** Reproducible splits
- **Sufficient Test Size:** 196 samples for reliable evaluation

---

## 📊 Model Comparison & Results

### Evaluation Metrics

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **R² Score** | Coefficient of determination | Proportion of variance explained (0-1, higher better) |
| **MAE** | Mean Absolute Error | Average absolute prediction error (lower better) |

### Model Performance Summary

All models are evaluated using:
- **R² Score:** How well the model explains price variance
- **Mean Absolute Error (MAE):** Average prediction error in actual price units

**Ranking Strategy:**
1. **R² Score** (primary): Higher R² indicates better variance explanation
2. **MAE** (secondary): Lower MAE indicates more accurate predictions
3. **Ensemble models** typically outperform individual models

---

## 📦 Output & Model Export

### Model Serialization

```python
import pickle

pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))
```

**Exported Files:**
1. **df.pkl** - Preprocessed dataframe with all features
2. **pipe.pkl** - Trained scikit-learn pipeline (best model)

**Pipeline Contents:**
- ColumnTransformer for categorical encoding
- Final trained regression model
- Ready for production predictions

### Deployment Workflow

```python
# Load pipeline in production
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Make predictions on new data
new_predictions = pipe.predict(X_new)

# Convert log predictions back to actual price
predicted_prices = np.exp(new_predictions)
```

---

## 📚 Libraries Used

### Core Data Science
- **NumPy:** Numerical computations
- **Pandas:** Data manipulation and analysis
- **Matplotlib:** Static visualizations
- **Seaborn:** Statistical data visualization

### Machine Learning (scikit-learn)
- **Linear Models:** LinearRegression, Ridge, Lasso
- **Neighbors:** KNeighborsRegressor
- **Trees:** DecisionTreeRegressor
- **Ensemble:** RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
- **SVM:** SVR
- **Preprocessing:** ColumnTransformer, OneHotEncoder, Pipeline
- **Model Selection:** train_test_split
- **Metrics:** r2_score, mean_absolute_error

### Gradient Boosting
- **XGBoost:** XGBRegressor (advanced boosting)

---

## 🎯 Key Insights

### Feature Importance (Expected)
1. **CPU Brand** - High correlation with price
2. **RAM** - Strong pricing indicator
3. **GPU Brand** - Significant for performance laptops
4. **SSD Storage** - Premium feature
5. **Operating System** - Mac OS commands premium
6. **Weight** - Indicates portability level
7. **PPI** - Screen quality metric
8. **Company** - Brand premium/discount
9. **Touchscreen** - Modern feature premium
10. **TypeName** - Gaming/Ultrabook categories differ in price

### Model Selection Recommendations
- **Best Performance:** Gradient Boosting or Stacking Regressor
- **Fastest Training:** Linear Regression or KNN
- **Most Interpretable:** Decision Tree or Linear Regression
- **Production Choice:** Voting/Stacking Ensemble (balanced performance)

---

## 🔄 Workflow Summary

```
1. Data Loading & Exploration
   ├─ Load dataset (1,303 records)
   ├─ Verify shape and data quality
   └─ Check for missing values and duplicates

2. Data Preprocessing
   ├─ Remove unnecessary columns
   ├─ Remove unit suffixes from text
   └─ Convert to appropriate data types

3. Feature Engineering
   ├─ Screen Resolution → Touchscreen, IPS, PPI
   ├─ CPU → CPU Brand categorization
   ├─ Memory → HDD, SSD, Hybrid capacities
   ├─ GPU → GPU Brand extraction
   ├─ OS → OS categorization
   └─ Weight → Numeric conversion

4. Exploratory Data Analysis
   ├─ Distribution plots
   ├─ Categorical relationships
   ├─ Scatter plots for continuous variables
   └─ Correlation heatmap

5. Data Transformation
   ├─ Apply log transformation to target (Price)
   └─ Split into training (85%) and test (15%)

6. Model Development & Training
   ├─ Create preprocessing pipeline (One-Hot Encoding)
   ├─ Train 11 different models
   ├─ Evaluate each model
   └─ Compare metrics (R² and MAE)

7. Overfitting Reduction
   ├─ Regularization (Ridge, Lasso)
   ├─ Tree constraints (max_depth, max_samples)
   ├─ Ensemble methods (Voting, Stacking)
   └─ Hyperparameter tuning

8. Model Export & Deployment
   ├─ Serialize best pipeline
   ├─ Save to pickle file
   └─ Ready for production inference
```

---

## 🚀 How to Use

### Training
```python
# Run all cells in sequence to train models
# Results are printed after each model execution
```

### Prediction
```python
# Load the saved pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Make predictions (example)
new_laptop_features = X_test.iloc[[0]]  # Single laptop
predicted_log_price = pipe.predict(new_laptop_features)
predicted_price = np.exp(predicted_log_price)  # Convert from log scale
```

---

## 📝 Notes

- All categorical variables undergo One-Hot Encoding with `drop='first'` to avoid multicollinearity
- Log transformation applied to Price for better model performance
- Random state set to 2 for reproducible results
- Models evaluated on held-out test set (15% of data)
- Ensemble models provide robust predictions by combining diverse models

---

**Project Created:** April 2026  
**Total Laptops Analyzed:** 1,303  
**Final Features:** 13  
**Models Trained:** 11  
**Best Strategy:** Stacking/Voting Ensemble
