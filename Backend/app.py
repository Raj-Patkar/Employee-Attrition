from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

# Define paths
scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
model_path = os.path.join(os.path.dirname(__file__), "model", "logistic_regression_attrition.pkl")

# Load the scaler and model safely
scaler, model = None, None
if os.path.exists(scaler_path) and os.path.exists(model_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("✅ Scaler and Model loaded successfully.")
else:
    print("⚠️ Missing 'scaler.pkl' or 'logistic_regression_attrition.pkl' in the 'model' folder.")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all input fields from form
        fields = [
            'Age', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement',
            'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'StockOptionLevel',
            'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsWithCurrManager'
        ]

        # Convert input values to float
        input_data = np.array([[float(request.form[field]) for field in fields]])

        if scaler is None or model is None:
            raise Exception("Scaler or Model not loaded. Please check your model folder.")

        # Scale the input using your saved scaler
        scaled_data = scaler.transform(input_data)

        # Predict using the logistic regression model
        prediction = model.predict(scaled_data)[0]

        # Interpret result
        result = "Employee likely to leave 😟" if prediction == 1 else "Employee will stay 🙂"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
