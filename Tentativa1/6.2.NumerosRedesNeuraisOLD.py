import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 0. Leitura dos dados
df = pd.read_csv('../euromilhoes.csv')  # Substitua 'dados.csv' pelo nome do seu arquivo de dados
df.drop(['SomaEstrelas', 'SomaNumeros', 'ÉImparNumeros?', 'ÉImparEstrelas?'], axis=1, inplace=True)

# 1. Aplicação das redes Neuronais
# 1.1. Separar os dados de entrada (features) e de saída (target)
X = df.drop(['Semana', 'Mes', 'Estrela1', 'Estrela2'], axis=1)  # Colunas com os números
y = df[['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5']]  # Colunas com os números a serem previstos

# 1.2. Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # random_state=42

# 1.3. Normalizar os dados
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1.4. Construir o modelo da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1])
])

# 1.5. Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# 1.6. Treinar o modelo
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# 1.7. Avaliar o modelo no conjunto de teste
loss = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'Erro médio quadrático (MSE) no conjunto de teste: {loss}')

# 1.8. Fazer previsões com o modelo
predictions = model.predict(X_test_scaled)
print(f'Exemplo de previsões: {predictions[0]}')


# 2. Tentativa de previsão
# 2.1. Dados para previsão

SerSexta = 0  # Exemplo: 0 representa não ser sexta-feira
mes = 8  # Exemplo: 7 representa julho
ano = 2023  # Exemplo: ano desejado
premio = 71000000  # Exemplo: Prêmio total disponível
dia = 22  # Exemplo: Dia que ocorre o jogo

# Criar DataFrame de previsão, ter cuidado que tem de ter a mesma ordem que X
dados_previsao = pd.DataFrame({
    'Numero1': [0],  # Valor inicial irrelevante, será substituído durante a previsão
    'Numero2': [0],
    'Numero3': [0],
    'Numero4': [0],
    'Numero5': [0],
    'Ano': [ano],
    'Premio': [premio],
    'MesNumerico': [mes],
    'Dia': [dia],
    'ÉSexta?': [SerSexta]
})

# Normalizar os dados de previsão
dados_previsao_scaled = scaler.transform(dados_previsao)

# Realizar a previsão
previsao = model.predict(dados_previsao_scaled)

# Arredondar os valores previstos para inteiros
previsao_inteira = np.round(previsao).astype(int)
previsao_inteira = np.squeeze(previsao_inteira)

# Verificar e remover valores repetidos
previsao_unica = np.unique(previsao_inteira)

# Se houver repetição, substituir valores repetidos
if len(previsao_unica) < len(previsao_inteira):
    numeros_unicos = set()
    previsao_corrigida = previsao_inteira.copy()
    for i, valor in enumerate(previsao_inteira):
        if valor == 0 or valor in numeros_unicos:
            novo_valor = valor
            while novo_valor == 0 or novo_valor in numeros_unicos:
                novo_valor = np.random.randint(1, 50)  # Intervalo de números do Euromilhões
            previsao_corrigida[i] = novo_valor
            numeros_unicos.add(novo_valor)
        else:
            numeros_unicos.add(valor)
    previsao_inteira = previsao_corrigida

previsao_inteira = np.sort(previsao_inteira)
print(f'Previsão dos números do Euromilhões: {previsao_inteira}')
