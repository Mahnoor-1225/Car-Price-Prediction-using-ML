import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Data Exploration

# Load the dataset
file_path = r"C:\Users\mahno\OneDrive\Desktop\internship\CAR DETAILS FROM CAR DEKHO (1).csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())

# Information about the dataset
print(data.info())

# Data visualization
# Histograms for numerical features
data.hist(bins=30, figsize=(15, 10))
plt.show()

# Box plots to identify outliers
plt.figure(figsize=(15, 10))
sns.boxplot(data=data)
plt.show()

# Correlation matrix for numerical columns only
numerical_data = data.select_dtypes(include=[np.number])
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Scatter plots to visualize relationships with target variable
plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='selling_price', data=data)
plt.show()

# Step 2: Feature Engineering

# Handling missing values
data = data.dropna()

# Encoding categorical variables
categorical_features = ['name', 'fuel', 'seller_type', 'transmission', 'owner']
numerical_features = ['year', 'km_driven']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Step 3: Model Selection and Training

# Splitting the dataset
X = data.drop('selling_price', axis=1)
y = data['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# Training models
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"{name}:")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred)}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred))}\n")

# Step 4: Model Evaluation

# Cross-validation with Grid Search for Random Forest
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20]
}

grid_search = GridSearchCV(estimator=Pipeline(steps=[('preprocessor', preprocessor),
                                                     ('model', RandomForestRegressor(random_state=42))]),
                           param_grid=param_grid,
                           cv=5,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
print("Best parameters for Random Forest: ", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print(f"Best Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
