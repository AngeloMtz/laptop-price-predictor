from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar logging para ver errores en Render
logging.basicConfig(level=logging.DEBUG)

# ─── Cargar modelo y scaler al iniciar la app ───────────────────────────────
model  = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
app.logger.debug('Modelo y scaler cargados correctamente.')

# Orden exacto de columnas que espera el modelo (mismo orden del entrenamiento)
COLUMNAS = [
    'Processor_Speed', 'RAM_Size', 'Storage_Capacity',
    'Screen_Size', 'Weight',
    'Brand_Acer', 'Brand_Asus', 'Brand_Dell', 'Brand_HP', 'Brand_Lenovo'
]

MARCAS = ['Acer', 'Asus', 'Dell', 'HP', 'Lenovo']

# ─── Ruta principal: muestra el formulario ──────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', marcas=MARCAS)

# ─── Ruta de predicción: recibe POST con los datos del formulario ────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Leer valores del formulario
        processor_speed    = float(request.form['processor_speed'])
        ram_size           = int(request.form['ram_size'])
        storage_capacity   = int(request.form['storage_capacity'])
        screen_size        = float(request.form['screen_size'])
        weight             = float(request.form['weight'])
        brand              = request.form['brand']  # ej. "Asus"

        # 2. Construir One-Hot Encoding manual para Brand
        brand_cols = {f'Brand_{m}': 1 if brand == m else 0 for m in MARCAS}

        # 3. Construir DataFrame con el orden correcto de columnas
        datos = {
            'Processor_Speed'  : [processor_speed],
            'RAM_Size'         : [ram_size],
            'Storage_Capacity' : [storage_capacity],
            'Screen_Size'      : [screen_size],
            'Weight'           : [weight],
            **{k: [v] for k, v in brand_cols.items()}
        }
        df_input = pd.DataFrame(datos, columns=COLUMNAS)
        app.logger.debug(f'Input DataFrame:\n{df_input}')

        # 4. Escalar con el mismo scaler del entrenamiento
        df_scaled = scaler.transform(df_input)

        # 5. Predecir
        precio = model.predict(df_scaled)[0]
        precio_fmt = f"${precio:,.2f}"
        app.logger.debug(f'Predicción: {precio_fmt}')

        return render_template('index.html',
                               marcas=MARCAS,
                               prediccion=precio_fmt,
                               valores=request.form)

    except Exception as e:
        app.logger.error(f'Error en predicción: {str(e)}')
        return render_template('index.html',
                               marcas=MARCAS,
                               error=str(e),
                               valores=request.form)

if __name__ == '__main__':
    app.run(debug=True)
