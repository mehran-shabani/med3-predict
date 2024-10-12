import joblib
import pandas as pd

# Load the saved model and scaler
model_path = '../models/best_logistic_regression_model.pkl'
scaler_path = '../models/scaler.pkl'
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)


# Function to calculate BMI
def calculate_bmi():
    weight = float(input("Enter your weight in kilograms: "))
    height = float(input("Enter your height in meters: "))
    bmi = weight / (height ** 2)
    return bmi


# Function to calculate Diabetes Pedigree Function (DPF)
def calculate_dpf():
    # Ask user about family diabetes history with more details
    num_parents = int(input("How many of your parents have diabetes? (0, 1, or 2): "))
    num_siblings = int(input("How many of your siblings have diabetes?: "))
    num_grandparents = int(input("How many of your grandparents have diabetes?: "))
    num_uncles_aunts = int(input("How many of your uncles/aunts have diabetes?: "))

    # Assign weights based on family relation
    parent_weight = 0.5
    sibling_weight = 0.3
    grandparent_weight = 0.2
    uncle_aunt_weight = 0.1

    # Calculate the DPF
    dpf = (num_parents * parent_weight) + (num_siblings * sibling_weight) + \
          (num_grandparents * grandparent_weight) + (num_uncles_aunts * uncle_aunt_weight)

    return dpf


# Function to get user input for other features
def get_user_input():
    n_pregnant = float(input("Number of Pregnancies: "))
    glucose = float(input("Glucose Level: "))
    tension = float(input("Blood Pressure (Diastolic): "))
    thickness = float(input("Skin Thickness: "))

    # Calculate BMI using weight and height
    bmi = calculate_bmi()

    age = float(input("Age: "))

    # Calculate DPF using family history
    dpf = calculate_dpf()

    # Create a pandas DataFrame with the inputs and feature names
    feature_names = ['n_pregnant', 'glucose', 'tension', 'thickness', 'bmi', 'pedigree', 'age']
    user_data = pd.DataFrame([[n_pregnant, glucose, tension, thickness, bmi, dpf, age]], columns=feature_names)

    # Standardize the input data using the saved scaler
    user_data_scaled = loaded_scaler.transform(user_data)

    return user_data_scaled


# Function to predict diabetes using the loaded model
def predict_diabetes():
    user_data_scaled = get_user_input()

    # Make the prediction
    prediction = loaded_model.predict(user_data_scaled)

    # Output the result
    if prediction == 1:
        print("The model predicts that you are at risk of diabetes.")
    else:
        print("The model predicts that you are not at risk of diabetes.")


# Call the function to predict diabetes
predict_diabetes()
