# Importing the necessary libraries for testing the model
import joblib
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

from diabetes.create_diabetpredict_model import X_test, y_test

# Path to the saved model and scaler
model_path = '../models/best_logistic_regression_model.pkl'
scaler_path = '../models/scaler.pkl'

# Load the saved model and scaler
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)

# Simulating test data (using the already split X_test and y_test from training)
# Assume X_test and y_test are already available

# Standardize the test data using the loaded scaler
X_test_scaled = loaded_scaler.transform(X_test)

# Make predictions using the loaded model
y_pred_loaded = loaded_model.predict(X_test_scaled)

# Evaluate the loaded model
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
precision_loaded = precision_score(y_test, y_pred_loaded)
confusion_matrix_loaded = confusion_matrix(y_test, y_pred_loaded)

# Output the evaluation results
print(f"Accuracy (Loaded Model): {accuracy_loaded}")
print(f"Precision (Loaded Model): {precision_loaded}")
print("Confusion Matrix (Loaded Model):")
print(confusion_matrix_loaded)
