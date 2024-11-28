import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib

def entrenar_modelo():
    # Cargar datos
    data = pd.read_csv('datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Preprocesamiento básico
    data.drop(columns=['customerID'], inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Definir columnas de características
    características_numéricas = ['tenure', 'MonthlyCharges', 'TotalCharges']
    características_categóricas = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Crear preprocesador
    preprocesador = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), características_numéricas),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), características_categóricas)
        ])
    
    # Crear pipeline
    modelo = Pipeline([
        ('preprocesador', preprocesador),
        ('clasificador', LogisticRegression(random_state=42))
    ])
    
    # Dividir datos
    x = data.drop('Churn', axis=1)
    y = data['Churn']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar modelo
    modelo.fit(x_train, y_train)
    
    # Guardar modelo
    joblib.dump(modelo, 'model.pkl')
    
    # Imprimir métricas
    print(f"Precisión en entrenamiento: {modelo.score(x_train, y_train):.3f}")
    print(f"Precisión en prueba: {modelo.score(x_test, y_test):.3f}")

if __name__ == "__main__":
    entrenar_modelo()