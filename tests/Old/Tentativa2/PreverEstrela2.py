import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


# 0. Leitura dos dados
df = pd.read_csv('../euromilhoes.csv')  # Substitua 'seu_arquivo.csv' pelo nome do seu arquivo de dados

# 1. Preparar os dados
X = df.drop(["Estrela2", 'Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5', 'Estrela1',
             'SomaNumeros', 'ÉImparNumeros?', 'SomaEstrelas', 'ÉImparEstrelas?', 'Mes', 'Semana'], axis=1)  # Features
y = df['Estrela2']  # Alvo

# 2. Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Criar o modelo Random Forest
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# 4. Treinar o modelo
model.fit(X_train, y_train)

# 5. Fazer previsões
predictions = model.predict(X_test)

# 6. Avaliar o modelo
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)

print(f'Erro Quadrático Médio: {mse}')
print(f'Erro Médio Absoluto: {mae}')
print(f'Raiz do Erro Quadrático Médio: {rmse}')
print(f'Coeficiente de Determinação (R²): {r2}')
print(f'Erro Percentual Absoluto Médio: {mape}')

# Impacto de cada uma das variáveis
# fig, ax = plt.subplots()
# plt.barh(X.columns, model.feature_importances_)

# 7. Realizar previsão única
# Dados para previsão
SerSexta = 1  # Exemplo: 0 representa não ser sexta-feira
mes = 2  # Exemplo: 7 representa julho
ano = 2004  # Exemplo: ano desejado
premio = 15000000  # Exemplo: Prêmio total disponível
dia = 13  # Exemplo: Dia que ocorre o jogo

# Criar DataFrame de previsão
dados_previsao = pd.DataFrame({
    'Ano': [ano],
    'Premio': [premio],
    'MesNumerico': [mes],
    'Dia': [dia],
    'ÉSexta?': [SerSexta]
})
previsao = model.predict(dados_previsao)
previsao_inteira = np.round(previsao).astype(int)
previsao_inteira = np.clip(previsao_inteira, 1, 12)

print(f'Previsão: {previsao_inteira}')
