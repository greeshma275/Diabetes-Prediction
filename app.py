from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from datetime import datetime
import json
import hashlib
import os
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'diabetes_prediction_secret_key'

# Store predictions in memory (in production, use a database)
predictions_history = []

# Store users in memory (in production, use a database)
users = {}

# Check if users.json exists, if not create it
if os.path.exists('users.json'):
    with open('users.json', 'r') as f:
        users = json.load(f)
else:
    # Create empty users file
    with open('users.json', 'w') as f:
        json.dump({}, f)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def is_logged_in():
    """Check if user is logged in"""
    return 'user_id' in session

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

# Prediction function
def predict_diabetes(input_data):
    input_df = pd.DataFrame([input_data], columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                                                   'SkinThickness', 'Insulin', 'BMI', 
                                                   'DiabetesPedigreeFunction', 'Age'])
    std_data = scaler.transform(input_df)
    prediction = classifier.predict(std_data)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_logged_in():
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def home():
    return redirect(url_for('landing'))

@app.route('/landing')
@login_required
def landing():
    return render_template('landing.html', username=session.get('username'))

@app.route('/about')
@login_required
def about():
    return render_template('about.html', username=session.get('username'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('index.html', predictions=predictions_history, username=session.get('username'))
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
    result = predict_diabetes(input_data)
    prediction_record = {
        'id': len(predictions_history) + 1,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': session.get('user_id'),
        'input_data': {
            'pregnancies': input_data[0],
            'glucose': input_data[1],
            'blood_pressure': input_data[2],
            'skin_thickness': input_data[3],
            'insulin': input_data[4],
            'bmi': input_data[5],
            'diabetes_pedigree': input_data[6],
            'age': input_data[7]
        },
        'result': result,
        'is_diabetic': 'diabetic' in result.lower()
    }
    predictions_history.append(prediction_record)
    return render_template('index.html', prediction=result, predictions=predictions_history, username=session.get('username'), input_data=prediction_record['input_data'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('auth.html', mode='register')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('auth.html', mode='register')
        
        if username in users:
            flash('Username already exists!', 'error')
            return render_template('auth.html', mode='register')
        
        # Create new user
        users[username] = {
            'email': email,
            'password': hash_password(password),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to file
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=2)
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth.html', mode='register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('Username and password are required!', 'error')
            return render_template('auth.html', mode='login')
        
        if username in users and users[username]['password'] == hash_password(password):
            session['user_id'] = username
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password!', 'error')
            return render_template('auth.html', mode='login')
    
    return render_template('auth.html', mode='login')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/history')
def history():
    return jsonify(predictions_history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global predictions_history
    predictions_history = []
    return jsonify({'message': 'History cleared successfully'})


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
