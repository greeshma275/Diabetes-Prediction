
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# Data preprocessing
scaler = StandardScaler()
x = scaler.fit_transform(diabetes_dataset.drop(columns=['Outcome']))
y = diabetes_dataset['Outcome']

# Train model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

def predict_diabetes(input_data):
    input_df = pd.DataFrame([input_data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                                                   'SkinThickness', 'Insulin', 'BMI', 
                                                   'DiabetesPedigreeFunction', 'Age'])
    std_data = scaler.transform(input_df)  # Now StandardScaler gets proper column names
    prediction = classifier.predict(std_data)
    return "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict function called!")  # Debugging line
    input_data = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['blood_pressure']),
        float(request.form['skin_thickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['diabetes_pedigree']),
        float(request.form['age'])
    ]
    
    print("Input Data Received:", input_data)  # Debugging line
    
    result = predict_diabetes(input_data)
    
    print("Prediction Result:", result)  # Debugging line
    
    return render_template('index.html', prediction=result)



if __name__ == '__main__':
    app.run(debug=True)
