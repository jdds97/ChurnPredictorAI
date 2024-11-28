from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario y crear un DataFrame con la misma estructura que los datos de entrenamiento
        data = {
            'gender': [request.form['gender']],
            'Partner': [request.form['Partner']],
            'Dependents': [request.form['Dependents']],
            'tenure': [float(request.form['tenure'])],
            'PhoneService': [request.form['PhoneService']],
            'MultipleLines': [request.form['MultipleLines']],
            'InternetService': [request.form['InternetService']],
            'OnlineSecurity': [request.form['OnlineSecurity']],
            'OnlineBackup': [request.form['OnlineBackup']],
            'DeviceProtection': [request.form['DeviceProtection']],
            'TechSupport': [request.form['TechSupport']],
            'StreamingTV': [request.form['StreamingTV']],
            'StreamingMovies': [request.form['StreamingMovies']],
            'Contract': [request.form['Contract']],
            'PaperlessBilling': [request.form['PaperlessBilling']],
            'PaymentMethod': [request.form['PaymentMethod']],
            'MonthlyCharges': [float(request.form['MonthlyCharges'])],
            'TotalCharges': [float(request.form['TotalCharges'])]
        }
        
        # Crear DataFrame
        df = pd.DataFrame(data)
        
        # Hacer predicción
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]
        
        result = {
            'prediction': 'Alto riesgo de abandono' if prediction[0] == 1 else 'Bajo riesgo de abandono',
            'probability': f'{probability * 100:.1f}%'
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)