import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 0. Leitura dos dados
df = pd.read_csv('../euromilhoes.csv')  # Substitua 'dados.csv' pelo nome do seu arquivo de dados
df.drop(['SomaEstrelas', 'SomaNumeros', 'ÉImparNumeros?'], axis=1, inplace=True)

# 1. Aplicação das redes Neuronais
# 1.1. Separar os dados de entrada (features) e de saída (target)
X = df.drop(['Semana', 'Mes', 'Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5'], axis=1)  # Colunas com os números
y = df[['Estrela1', 'Estrela2']]  # Colunas com os números a serem previstos

# 1.2. Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # random_state=42

# 1.3. Normalizar os dados
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1.4. Construir o modelo da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.2),  # Adiciona Dropout para evitar overfitting
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Adiciona Dropout para evitar overfitting
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Adiciona Dropout para evitar overfitting
    tf.keras.layers.Dense(y_train.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))  # Camada de saída com regularização L2
])

# 1.5. Compilar o modelo com um learning rate menor
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])

# 1.6. Treinar o modelo com mais épocas
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# 1.7. Avaliar o modelo no conjunto de teste
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Acurácia do modelo no conjunto de teste: {accuracy:.2f}')

# Fazer previsões com o modelo
predictions = model.predict(X_test_scaled)
previsao_inteira = np.round(predictions).astype(int)
previsao_inteira = np.clip(previsao_inteira, 1, 12)
previsao_inteira = np.sort(previsao_inteira)
print(f'Previsão dos números do Euromilhões: {previsao_inteira}')

# Dados para previsão
SerSexta = 0  # Exemplo: 0 representa não ser sexta-feira
mes = 7  # Exemplo: 7 representa julho
ano = 2023  # Exemplo: ano desejado
premio = 49000000  # Exemplo: Prêmio total disponível
dia = 25  # Exemplo: Dia que ocorre o jogo
Impar = 0  # Exemplo: A Soma dos números será impar? (Sim -> 1, Não -> 0)

# Criar DataFrame de previsão
dados_previsao = pd.DataFrame({
    'Estrela1': [0],  # Valor inicial irrelevante, será substituído durante a previsão
    'Estrela2': [0],
    'Ano': [ano],
    'Premio': [premio],
    'MesNumerico': [mes],
    'Dia': [dia],
    'ÉSexta?': [SerSexta],
    'ÉImparEstrelas?': [Impar],
})

# Normalizar os dados de previsão
dados_previsao_scaled = scaler.transform(dados_previsao)

# Realizar a previsão
previsao = model.predict(dados_previsao_scaled)

# Arredondar os valores previstos para inteiros e ajustar para o intervalo 1-12
previsao_inteira = np.round(previsao).astype(int)
previsao_inteira = np.clip(previsao_inteira, 1, 12)
previsao_inteira = np.sort(previsao_inteira)

print(f'Previsão dos números do Euromilhões: {previsao_inteira}')
