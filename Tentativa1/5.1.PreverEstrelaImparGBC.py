import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../euromilhoes.csv")

# Separar os dados em features (X) e target (y)
X = df.drop(
    ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5', 'Estrela1', 'Estrela2', 'Mes', 'Semana', 'SomaNumeros',
     'ÉImparNumeros?', 'SomaEstrelas',
     'ÉImparEstrelas?'], axis=1)  # Features
y = df['ÉImparEstrelas?']  # Target

# Dividir o conjunto de dados em conjuntos de treino e teste (80% treinamento, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)  # 5, 7

# Criar o modelo Gradient Boosting Classifier
model = GradientBoostingClassifier()

# Treinar o modelo com os dados de treino
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print('Matriz de Confusão:')
print(conf_matrix)

# Exibir um relatório de classificação com métricas adicionais
report = classification_report(y_test, y_pred)
print('Relatório de Classificação:')
print(report)

# teste - - -
y_pred = model.predict(X_test)

probs = model.predict_proba(X_test)[:, 1]
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Recall: ", metrics.recall_score(y_test, y_pred, zero_division=1, pos_label=1))
print("Precision:", metrics.precision_score(y_test, y_pred, zero_division=1, pos_label=1))
auc = roc_auc_score(y_test, probs)
print("roc_auc_score:", auc)

# Confusion Matrix
classes = np.unique(y_test)
fig, ax = plt.subplots()
cm = metrics.confusion_matrix(y_test, y_pred, labels=classes[::1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Pastel1', cbar=False)
ax.set(xlabel="Previsto", ylabel="Verdadeiro", title="Confusion matrix")
ax.xaxis.set_label_position("top")  # Posiciona os rótulos dos valores previstos em cima
ax.xaxis.set_ticks_position("top")  # Posiciona os ticks dos valores previstos em cima
ax.yaxis.set_ticks_position("left")  # Posiciona os ticks das classes à esquerda
plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=classes[::1], rotation=0)
plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=classes[::1],
           rotation=0)  # Inverte a ordem dos rótulos das classes
plt.show()
del fig, ax

# Impacto de cada uma das variáveis
fig, ax = plt.subplots()
plt.barh(X.columns, model.feature_importances_)

# Final - Previsão real
caso_a_prever = pd.DataFrame({
    'Ano': [2023],  # Exemplo: ano desejado
    'Premio': [39000000],  # Exemplo: Previsão do 1.º Prémio com Jackpot do jogo anterior
    'MesNumerico': [7],  # Exemplo: 7 representa julho
    'Dia': [25],  # Exemplo: Dia que ocorre o jogo
    'ÉSexta?': [0]  # Exemplo: 0 representa não ser sexta-feira
})

# Normalizar os dados do caso a prever (se necessário)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
caso_a_prever_scaled = scaler.transform(caso_a_prever)

# Fazer a previsão
previsao = model.predict(caso_a_prever_scaled)
if 0 in previsao:
    resultado = 'Par'
else:
    resultado = 'Impar'

print(f'Previsão da soma das estrelas: {resultado} -> {previsao}')
