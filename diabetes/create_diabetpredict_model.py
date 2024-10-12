import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load the dataset from the 'data' folder
data_path = '../data/diabetes_data.csv'
df = pd.read_csv(data_path)

# Dropping the 'insulin' column
df_no_insulin = df.drop(columns=['insulin'])

# Separating features and labels
X = df_no_insulin.iloc[:, :-1]
y = df_no_insulin.iloc[:, -1]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Standardizing the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
scaler_path = '../models/scaler.pkl'
joblib.dump(scaler, scaler_path)

# Define the hyperparameter grid for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],        # Type of regularization
    'solver': ['liblinear']         # Solver compatible with l1 regularization
}

# Setting up the grid search with cross-validation
grid_search = GridSearchCV(LogisticRegression(random_state=2, max_iter=1000), param_grid, cv=5, scoring='accuracy')

# Fitting the grid search on the standardized data
grid_search.fit(X_train_scaled, y_train)

# Best parameters and model
best_model = grid_search.best_estimator_

# Save the best model to a file
model_path = '../models/best_logistic_regression_model.pkl'
joblib.dump(best_model, model_path)

# Making predictions on the test set using the best model
y_pred_best = best_model.predict(X_test_scaled)

# Evaluating the model
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best)

# Output results
print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Accuracy: {accuracy_best}")
print(f"Precision: {precision_best}")

print("Model and scaler have been saved successfully!")
