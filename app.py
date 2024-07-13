from flask import Flask, request, render_template, redirect, url_for, send_file, flash
import pandas as pd
import io
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'supersecretkey'

data = None  # Declarar la variable global 'data'

# Ruta principal que muestra el formulario
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/multimedia')
def multimedia():
    return render_template('multimedia.html')

# Ruta para manejar la carga del archivo y procesar los datos
@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    global data  # Declarar que vamos a usar la variable global 'data'
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        try:
            if file.filename.endswith('.csv'):
                data = pd.read_csv(io.BytesIO(file.read()))
            elif file.filename.endswith('.xlsx'):
                data = pd.read_excel(io.BytesIO(file.read()))
            else:
                flash('Invalid file format')
                return redirect(request.url)
            
            predictions, mse, mae, r2, plot_url = make_predictions(data)
            
            data['Predictions'] = predictions
            
            predictions_html = data.head().to_html(classes='table table-striped')
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
            
            return render_template('predictions.html', tables=[predictions_html], metrics=metrics, plot_url=plot_url)
        except Exception as e:
            flash(f'Ocurrió un error al procesar el archivo: {e}')
            return redirect(request.url)
    return render_template('predictions.html')

@app.route('/download_predictions')
def download_predictions():
    global data  # Declarar que vamos a usar la variable global 'data'
    
    predictions_csv = io.BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)
    return send_file(predictions_csv, mimetype='text/csv', as_attachment=True, download_name='predictions.csv')

def make_predictions(data):
    model = load_model('best_model.h5')

    # Preprocessing data
    data['day_of_week'] = pd.to_datetime(data['Date']).dt.dayofweek
    data['day_of_month'] = pd.to_datetime(data['Date']).dt.day
    data['month'] = pd.to_datetime(data['Date']).dt.month
    data['quarter'] = pd.to_datetime(data['Date']).dt.quarter
    data['year'] = pd.to_datetime(data['Date']).dt.year

    features = data[['Volume', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year']]
    close_prices = data['Close*'].values.reshape(-1, 1)

    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)

    scaler_close = MinMaxScaler()
    close_prices_scaled = scaler_close.fit_transform(close_prices)

    predictions_scaled = model.predict(features_scaled)
    predictions = scaler_close.inverse_transform(predictions_scaled)

    mse = np.mean((predictions - close_prices) ** 2)
    mae = np.mean(np.abs(predictions - close_prices))
    r2 = 1 - (np.sum((predictions - close_prices) ** 2) / np.sum((close_prices - np.mean(close_prices)) ** 2))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(close_prices, label='Valores reales')
    plt.plot(predictions, label='Predicciones')
    plt.title('Predicciones vs Valores Reales')
    plt.xlabel('Índice')
    plt.ylabel('Precio de Cierre')
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return predictions.flatten(), mse, mae, r2, plot_url

if __name__ == "__main__":
    app.run(debug=True)
