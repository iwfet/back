from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask_sqlalchemy import SQLAlchemy
import requests
from flask_cors import CORS


estados_brasil = {
    "RO": {"nome": "Acre", "lat": 3.7759917, "lon": -60.2530222},
}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///consulta.db'
db = SQLAlchemy(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

model = tf.keras.models.load_model('modelo_rede_neural.h5')

encoder_estado = LabelEncoder()
encoder_cultivo = LabelEncoder()

scaler = StandardScaler()

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
    return (clima['main']['temp'] - 32) * 5/9, clima['main']['humidity']


def saveDB(cultivo, estado, temperatura, umidade, previsao):
    nova_consulta = Consulta(cultivo=cultivo, estado=estado, temperatura=temperatura, umidade=umidade, previsao=previsao)
    db.session.add(nova_consulta)
    db.session.commit()


@app.route('/previsao', methods=['POST'])
def previsao():
    data = request.get_json()
    estado = data['estado']
    cultivo = data['cultivo']
    
    temp, humidity = buscarClima(estado)

    if not hasattr(previsao, 'encoder_estado_fitted'):  
        states_list = list(estados_brasil.keys()) 
        encoder_estado.fit(states_list)  
        setattr(previsao, 'encoder_estado_fitted', True)  

    estado_encoded = encoder_estado.transform([estado])

    if not hasattr(previsao, 'encoder_cultivo_fitted'):  
        cultivos_list = list(set(df['Cultivo']))  
        encoder_cultivo.fit(cultivos_list)  
        setattr(previsao, 'encoder_cultivo_fitted', True) 

    cultivo_encoded = encoder_cultivo.transform([cultivo])

    X = np.concatenate([estado_encoded, np.array([[temp, humidity]]), np.expand_dims(cultivo_encoded, axis=1)], axis=1)
    X = scaler.transform(X)

  
    previsao = model.predict(X)

    saveDB(cultivo, estado, temp, humidity, previsao)

    return jsonify({'previsao': float(previsao), "temp":temp,"humidity":humidity })

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
