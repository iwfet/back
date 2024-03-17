import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model

modelo_carregado = load_model('meu_modelo.h5')

exemplo_teste = pd.DataFrame({
    'Estado': ['PR','PR','PR','PR'],
    'Temperatura': [27,34,10,40],
    'Umidade': [22,27,80,10],
    'Semente': ['Arroz Integral','Arroz Integral','Arroz Integral','Arroz Integral']
})

encoder = LabelEncoder()
exemplo_teste['Semente'] = encoder.fit_transform(exemplo_teste['Semente'])
onehot = OneHotEncoder(sparse=False)
estado_encoded = onehot.fit_transform(exemplo_teste[['Estado']])
df_teste = pd.DataFrame(estado_encoded, columns=[f'Estado_{state}' for state in encoder.classes_])
df_teste['Temperatura'] = exemplo_teste['Temperatura']
df_teste['Umidade'] = exemplo_teste['Umidade']
df_teste['Semente'] = exemplo_teste['Semente']


scaler = StandardScaler()
X_teste_scaled = scaler.fit_transform(df_teste)


previsao = modelo_carregado.predict(X_teste_scaled)


print(previsao)