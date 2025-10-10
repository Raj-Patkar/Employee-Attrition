from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "attrition_model.pkl")
#model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        monthly_income = float(request.form['monthly_income'])
        overtime = int(request.form['overtime'])
        job_satisfaction = int(request.form['job_satisfaction'])
        years_at_company = int(request.form['years_at_company'])

        # Format input
        input_data = np.array([[age, monthly_income, overtime, job_satisfaction, years_at_company]])

        # Predict
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "Employee likely to leave 😟"
        else:
            result = "Employee will stay 🙂"

        return render_template("result.html", prediction=result)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
