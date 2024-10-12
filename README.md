# Chronic Disease Prediction System

## Overview

This project is a **Chronic Disease Prediction System** that uses machine learning to assess the likelihood of a person being at risk of various chronic diseases, such as **diabetes**, **hypertension**, and **cardiovascular diseases**. The model takes inputs such as personal medical history, family history, biometric data, and lifestyle factors to provide an estimated risk percentage.

### Features:
- **Prediction for multiple chronic diseases**: Currently supports the prediction of diabetes but is designed to expand to other diseases.
- **User-friendly interaction**: The user provides their health data step by step through clear and interactive questions.
- **Probability-based risk assessment**: Displays a detailed probability percentage of the user's risk, with adjustments based on their input.
- **Adjustable confidence levels**: Risk levels are dynamically adjusted using a step-wise method to provide more realistic interpretations.

## Technologies Used
- **Python 3.x**
- **Scikit-learn**: For model training and prediction.
- **Pandas & Numpy**: For data handling and manipulation.
- **Joblib**: For saving and loading machine learning models.

## Getting Started

### Prerequisites

To run this project locally, ensure you have Python 3.x installed and the following packages:

    
      pip install numpy pandas scikit-learn joblib
## Installing
Clone the repository to your local machine:

     
     git clone https://github.com/your-username/chronic-disease-prediction.git
    cd chronic-disease-prediction
Load your saved machine learning model and scaler inside the models/ directory. If these models are not available, train and save them using a chronic disease dataset.

## Run the prediction script:


     python main.py
### How It Works
Upon running the script, users will be prompted to provide health-related inputs such as:

Number of pregnancies (for female users),
### Glucose level
### Blood pressure
### Skin fold thickness
### BMI (calculated using height and weight)
### Family history of chronic diseases
### Age.
After collecting the inputs, the model will estimate the user's risk of developing chronic diseases based on the data provided. It will return a probability percentage that indicates the likelihood of the user being at risk.

## Adjusting Predictions
The system employs an adjustment mechanism to provide realistic probability estimates:

For high initial probabilities (≥ 80%), a small increase is added.
For lower probabilities, a larger adjustment is made to reflect real-world risk factors.
Example Output
bash
Copy code
How many times have you been pregnant? (if not applicable, enter 0): 2
What is your fasting glucose level? (mg/dL): 105
What is your diastolic blood pressure? (mmHg): 85
...
After adjustment, there is a 67.32% chance that you are at risk of chronic disease.
The model predicts that you are at risk of chronic disease.
Customization
The project is designed to be expanded for other chronic diseases beyond diabetes. To support a new disease:

Collect relevant data for that disease (e.g., cholesterol levels for heart disease).
Update the input questions to reflect the disease's specific risk factors.
Train a new machine learning model or retrain the existing model to predict the new chronic disease.
Model Training
For users who want to retrain the model for another disease or update the existing one:

Use a chronic disease dataset in CSV format.
Train the model using scikit-learn with the features specific to your disease.
Save the model and the scaler in the models/ directory using joblib.
python
Copy code
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Assuming data is loaded into X_train, y_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, 'models/chronic_disease_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
Future Improvements
Support for other diseases: Expand prediction to other chronic diseases such as hypertension and cardiovascular diseases.
Web-based Interface: Integrate with a web front-end to improve user accessibility.
Mobile App: Build a mobile app for easy input and prediction.
Contributing
We welcome contributions! Feel free to submit a pull request or report an issue.

Fork the repository.
Create your feature branch (git checkout -b feature/new-disease).
Commit your changes (git commit -m 'Add new disease prediction').
Push to the branch (git push origin feature/new-disease).
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.


This `README.md` provides a clear overview of the project, instructions for setup, 