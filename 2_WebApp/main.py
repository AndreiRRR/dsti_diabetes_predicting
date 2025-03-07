import pickle as pkl
from urllib import request

import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, app, render_template

app = Flask(__name__)

with open("Diabetes_Model.pkl", "rb") as f:
    model = pkl.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')

    elif request.method == "POST":
        try:

            PatientID = request.form.get('PatientID')
            Pregnancies = request.form.get('Pregnancies')
            PlasmaGlucose = request.form.get('PlasmaGlucose')
            DiastolicBloodPressure = request.form.get('DiastolicBloodPressure')
            TricepsThickness = request.form.get('TricepsThickness')
            SerumInsulin = request.form.get('SerumInsulin')
            BMI = request.form.get('BMI')
            DiabetesPedigree = request.form.get('DiabetesPedigree')
            Age = request.form.get('Age')

            input_data = pd.DataFrame(
                {'PatientID':int(PatientID),
                'Pregnancies': int(Pregnancies),
                'PlasmaGlucose': int(PlasmaGlucose),
                'DiastolicBloodPressure': int(DiastolicBloodPressure),
                'TricepsThickness': int(TricepsThickness),
                'SerumInsulin': int(SerumInsulin),
                'BMI': float(BMI),
                'DiabetesPedigree': float(DiabetesPedigree),
                'Age': int(Age)}
                , index=[0])

            # Call the model for prediction
            prediction = model.predict(input_data)

            # Response in function of prediction result
            response = "Prediction : DIABETES" if prediction[0] == 1 else "Prediction : NO DIABETES"

            # Send back index.html with parameters : Prediction Result and Form Values
            return render_template('index.html',
                                   message=response,
                                   PatientID=PatientID,
                                   Pregnancies=Pregnancies,
                                   PlasmaGlucose=PlasmaGlucose,
                                   DiastolicBloodPressure=DiastolicBloodPressure,
                                   TricepsThickness=TricepsThickness,
                                   SerumInsulin=SerumInsulin,
                                   BMI=BMI,
                                   DiabetesPedigree=DiabetesPedigree,
                                   Age=Age,
                                   )

        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
