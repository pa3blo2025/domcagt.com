from flask import Flask, request, jsonify
import joblib  # o la librería que uses para cargar tu modelo

app = Flask(__name__)

# Carga tu modelo entrenado
modelo = joblib.load('modelo_entrenado.pkl')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json(force=True)
    # Asegúrate de que los datos recibidos tengan el formato correcto
    prediccion = modelo.predict([datos['caracteristicas']])
    # Devuelve el resultado como un JSON
    return jsonify({'prediccion': prediccion.tolist()})

if __name__ == '__main__':
    app.run(debug=True)