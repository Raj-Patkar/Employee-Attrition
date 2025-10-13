from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")

# Define paths
model_path = os.path.join(os.path.dirname(__file__), "model", "logistic_regression_attrition.pkl")
scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Feature columns expected by the model
feature_columns = [
    'Age', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'StockOptionLevel',
    'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsWithCurrManager', 'OverTime_Yes',
    'JobRole_Human Resources', 'JobRole_Laboratory Technician',
    'JobRole_Manager', 'JobRole_Manufacturing Director',
    'JobRole_Research Director', 'JobRole_Research Scientist',
    'JobRole_Sales Executive', 'JobRole_Sales Representative',
    'MaritalStatus_Married', 'MaritalStatus_Single',
    'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
    'EducationField_Life Sciences', 'EducationField_Marketing',
    'EducationField_Medical', 'EducationField_Other',
    'EducationField_Technical Degree'
]

def predict_attrition(
    Age, DistanceFromHome, EnvironmentSatisfaction, JobInvolvement, JobLevel,
    JobSatisfaction, MonthlyIncome, StockOptionLevel, TotalWorkingYears,
    WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
    YearsWithCurrManager, OverTime, JobRole, MaritalStatus,
    BusinessTravel, EducationField
):
    # Base input structure
    input_data = {col: 0 for col in feature_columns}

    # Fill numeric fields
    input_data.update({
        'Age': Age,
        'DistanceFromHome': DistanceFromHome,
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'JobInvolvement': JobInvolvement,
        'JobLevel': JobLevel,
        'JobSatisfaction': JobSatisfaction,
        'MonthlyIncome': MonthlyIncome,
        'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': TotalWorkingYears,
        'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole,
        'YearsWithCurrManager': YearsWithCurrManager,
    })

    # OverTime
    if OverTime == "Yes":
        input_data['OverTime_Yes'] = 1

    # JobRole
    jobrole_col = f"JobRole_{JobRole}"
    if jobrole_col in input_data:
        input_data[jobrole_col] = 1

    # MaritalStatus
    if MaritalStatus != "Divorced":  # We only have Married and Single columns
        marital_col = f"MaritalStatus_{MaritalStatus}"
        input_data[marital_col] = 1

    # BusinessTravel
    travel_col = f"BusinessTravel_{BusinessTravel}"
    if travel_col in input_data:
        input_data[travel_col] = 1

    # EducationField
    edu_col = f"EducationField_{EducationField}"
    if edu_col in input_data:
        input_data[edu_col] = 1

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    df_input = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)

    # Make prediction
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    result = "Yes (Attrition Likely)" if prediction == 1 else "No (Attrition Unlikely)"
    return result, probability

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Age = float(request.form['Age'])
        DistanceFromHome = float(request.form['DistanceFromHome'])
        EnvironmentSatisfaction = float(request.form['EnvironmentSatisfaction'])
        JobInvolvement = float(request.form['JobInvolvement'])
        JobLevel = float(request.form['JobLevel'])
        JobSatisfaction = float(request.form['JobSatisfaction'])
        MonthlyIncome = float(request.form['MonthlyIncome'])
        StockOptionLevel = float(request.form['StockOptionLevel'])
        TotalWorkingYears = float(request.form['TotalWorkingYears'])
        WorkLifeBalance = float(request.form['WorkLifeBalance'])
        YearsAtCompany = float(request.form['YearsAtCompany'])
        YearsInCurrentRole = float(request.form['YearsInCurrentRole'])
        YearsWithCurrManager = float(request.form['YearsWithCurrManager'])
        OverTime = request.form['OverTime']
        JobRole = request.form['JobRole']
        MaritalStatus = request.form['MaritalStatus']
        BusinessTravel = request.form['BusinessTravel']
        EducationField = request.form['EducationField']

        result, probability = predict_attrition(
            Age, DistanceFromHome, EnvironmentSatisfaction, JobInvolvement, JobLevel,
            JobSatisfaction, MonthlyIncome, StockOptionLevel, TotalWorkingYears,
            WorkLifeBalance, YearsAtCompany, YearsInCurrentRole,
            YearsWithCurrManager, OverTime, JobRole, MaritalStatus,
            BusinessTravel, EducationField
        )

        return render_template("result.html", prediction=result, probability=f"{probability:.2f}")

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}", probability="")

if __name__ == "__main__":
    app.run(debug=True)
