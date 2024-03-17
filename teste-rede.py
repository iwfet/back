from sklearn.preprocessing import StandardScaler,LabelEncoder
from tensorflow import keras

# Carregar o modelo treinado
loaded_model = keras.models.load_model('meu_modelo.h5')

# Supondo que você tenha os valores de estado, cultivo, temperatura e umidade disponíveis
estado = 'SP'  # Exemplo de estado
cultivo = 'Arroz Integral'  # Exemplo de cultivo
temperatura = 90  # Exemplo de temperatura
umidade = 0  # Exemplo de umidade

# Aplicar Label Encoding aos valores de estado e cultivo
encoder_estado = LabelEncoder()
encoder_cultivo = LabelEncoder()

estado_encoded = encoder_estado.fit_transform([estado])
cultivo_encoded = encoder_cultivo.fit_transform([cultivo])

# Criar uma lista com os valores de entrada codificados
dados_input = [[estado_encoded[0], cultivo_encoded[0], temperatura, umidade]]

# Normalizar os dados de entrada usando o mesmo scaler utilizado para treinar o modelo
scaler = StandardScaler()
X_input_scaled = scaler.fit_transform(dados_input)

# Fazer previsões usando o modelo carregado
predictions = loaded_model.predict(X_input_scaled).flatten()

# Exibir a previsão
print("Previsão para a combinação estado-cultivo:")
print(predictions[0])

# Aqui você pode fazer qualquer outra manipulação ou análise da previsão conforme necessário
