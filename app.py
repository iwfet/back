from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import StandardScaler, LabelEncoder
import requests
from flask_cors import CORS
from tensorflow import keras


estados_brasil = {"PR": {"nome": "Parana", "lat": -31.7, "lon": -60.5},}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///consulta.db'
db = SQLAlchemy(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
loaded_model = keras.models.load_model('meu_modelo.h5')




class Consulta(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cultivo = db.Column(db.String(50))
    estado = db.Column(db.String(50))
    temperatura = db.Column(db.Float)
    umidade = db.Column(db.Float)
    previsao = db.Column(db.Float)

def buscarClima(estado):
    api_key = '48ef3483de242b79bb4c4c7298dca55a'
    lat = estados_brasil[estado]['lat']
    lon = estados_brasil[estado]['lon']
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}'
    response = requests.get(url)
    clima = response.json()
    return (kelvin_para_celsius(clima['main']['temp'] ), clima['main']['humidity'])

def saveDB(cultivo, estado, temperatura, umidade, previsao):
    nova_consulta = Consulta(cultivo=cultivo, estado=estado, temperatura=temperatura, umidade=umidade, previsao=previsao)
    db.session.add(nova_consulta)
    db.session.commit()

def kelvin_para_celsius(temperatura_kelvin):
    temperatura_celsius = temperatura_kelvin - 273.15
    return temperatura_celsius

@app.route('/previsao', methods=['POST'])
def previsao():
    data = request.get_json()
    estado = data['estado']
    cultivo = data['cultivo']
    temperatura, umidade = buscarClima(estado)
    encoder_estado = LabelEncoder()
    encoder_cultivo = LabelEncoder()
    
    estado_encoded = encoder_estado.fit_transform([estado])
    cultivo_encoded = encoder_cultivo.fit_transform([cultivo])
    
    dados_input = [[estado_encoded[0], cultivo_encoded[0], temperatura, umidade]]

    scaler = StandardScaler()
    X_input_scaled = scaler.fit_transform(dados_input)
    predictions = loaded_model.predict(X_input_scaled).flatten()

    saveDB(cultivo, estado, temperatura, umidade, predictions[0])

    return jsonify({'previsao': format(float(predictions[0]), '.2f'), "temperatura":temperatura,"umidade":umidade })

@app.route('/ultimas_consultas', methods=['GET'])
def ultimas_consultas():
    consultas = Consulta.query.order_by(Consulta.id.desc()).limit(10).all()
    resultado = []
    for consulta in consultas:
        resultado.append({
            "id": consulta.id,
            "cultivo": consulta.cultivo,
            "estado": consulta.estado,
            "temperatura": consulta.temperatura,
            "umidade": consulta.umidade,
            "previsao": consulta.previsao
        })
    return jsonify({"consultas": resultado})
    
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
