import joblib
import pandas as pd

# Load the saved model and scaler
model_path = '../models/best_logistic_regression_model.pkl'
scaler_path = '../models/scaler.pkl'
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)


# Function to calculate BMI
def calculate_bmi():
    weight = float(input("Enter your weight in kilograms (e.g., 70): "))
    height = float(input("Enter your height in meters (e.g., 1.75): "))
    bmi = weight / (height ** 2)
    return bmi


# Function to calculate Diabetes Pedigree Function (DPF)
def calculate_dpf():
    # Ask user about family diabetes history with more details
    print("Let's assess your family's history of diabetes:")
    num_parents = int(input("How many of your parents have diabetes? (0, 1, or 2): "))
    num_siblings = int(input("How many of your siblings have diabetes? (enter a number): "))
    num_grandparents = int(input("How many of your grandparents have diabetes? (enter a number): "))
    num_uncles_aunts = int(input("How many of your uncles or aunts have diabetes? (enter a number): "))

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
    n_pregnant = float(input("How many times have you been pregnant? (if not applicable, enter 0): "))
    glucose = float(input("What is your fasting glucose level? (mg/dL): "))
    tension = float(input("What is your diastolic blood pressure? (mmHg): "))

    # Asking for skin thickness with more explanation
    print("\nSkin Thickness refers to the thickness of skin folds measured on your triceps.")
    thickness = float(input("Enter your skin thickness in millimeters (e.g., 20): "))

    # Calculate BMI using weight and height
    bmi = calculate_bmi()

    age = float(input("What is your age in years?: "))

    # Calculate DPF using family history
    dpf = calculate_dpf()

    # Create a pandas DataFrame with the inputs and feature names
    feature_names = ['n_pregnant', 'glucose', 'tension', 'thickness', 'bmi', 'pedigree', 'age']
    user_data = pd.DataFrame([[n_pregnant, glucose, tension, thickness, bmi, dpf, age]], columns=feature_names)

    # Standardize the input data using the saved scaler
    user_data_scaled = loaded_scaler.transform(user_data)

    return user_data_scaled


def adjust_probability(probability):
    if probability >= 0.9:
        probability += 0.05  # Add 5% for high probabilities
    elif probability >= 0.6:
        probability += 0.10  # Add 10% for medium-high probabilities
    elif probability >= 0.3:
        probability += 0.15  # Add 15% for medium-low probabilities
    else:
        probability += 0.20  # Add 20% for low probabilities

    # Ensure the probability doesn't exceed 100%
    return min(probability, 1.0)


# Function to predict diabetes using the loaded model and show probability
def predict_diabetes():
    user_data_scaled = get_user_input()

    # Get the probability of being diabetic (class 1)
    probability = loaded_model.predict_proba(user_data_scaled)[0][1]

    # Adjust the probability based on the initial value
    adjusted_probability = adjust_probability(probability)

    # Make sure the adjusted probability doesn't exceed 100%
    adjusted_probability = min(adjusted_probability, 1.0)

    # Convert to percentage
    percentage = adjusted_probability * 100

    # Interpret the prediction
    if adjusted_probability >= 0.5:
        print(f"\nAfter adjustment, there is a {percentage:.2f}% chance that you are at risk of diabetes.")
        print("The model predicts that you are at risk of diabetes.")
    else:
        print(f"\nAfter adjustment, there is only a {percentage:.2f}% chance that you are at risk of diabetes.")
        print("The model predicts that you are not at risk of diabetes.")


# Call the function to predict diabetes
predict_diabetes()
