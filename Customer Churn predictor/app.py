from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model (and scaler if needed)
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map categorical inputs
        geography_map = {'France': 0, 'Germany': 1, 'Spain': 2}
        gender_map = {'Male': 0, 'Female': 1}

        # Step 1: Build input as a DataFrame (to match training format)
        input_data = pd.DataFrame([{
            'CreditScore':       float(request.form['credit_score']),
            'Geography':         geography_map[request.form['geography']],
            'Gender':            gender_map[request.form['gender']],
            'Age':               int(request.form['age']),
            'Tenure':            int(request.form['tenure']),
            'Balance':           float(request.form['balance']),
            'NumOfProducts':     int(request.form['num_of_products']),
            'HasCrCard':         int(request.form['has_cr_card']),
            'IsActiveMember':    int(request.form['is_active_member']),
            'EstimatedSalary':   float(request.form['estimated_salary']),
        }])

        # Step 2: Scale only the 3 columns — exactly as done during training
        input_data[['CreditScore', 'Balance', 'EstimatedSalary']] = scaler.transform(
            input_data[['CreditScore', 'Balance', 'EstimatedSalary']]
        )

        print("After scaling:\n", input_data)
        print("Prediction:", model.predict(input_data))
        print("Probability:", model.predict_proba(input_data))

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Churn probability

        result = {
            'prediction': 'Will Churn' if prediction == 1 else 'Will NOT Churn',
            'probability': f'{probability * 100:.2f}%'
        }
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)