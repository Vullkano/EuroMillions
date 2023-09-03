import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("euromilhoes.csv")


# 1. BoxPlot dos diferentes números

fig, ax = plt.subplots(figsize=(10, 6))  # Figura única com os vários BoxPlot
sns.boxplot(data=df.iloc[:, :5], orient='v', ax=ax)  # BoxPlot Vertical

# Configurações do eixo x
ax.set_xticklabels(df.columns[:5])
ax.set_xlabel('Colunas')

# Configuração do título
ax.set_title('Boxplots dos diferentes números')

# Ajuste dos subplots
plt.subplots_adjust(bottom=0.25)

# Exibir o plot
plt.show()
del fig, ax


# 2. BoxPlot das diferentes estrelas

fig, ax = plt.subplots(figsize=(10, 6))  # Figura única com os vários BoxPlot
sns.boxplot(data=df.iloc[:, 5:7], orient='v', ax=ax)  # BoxPlot Vertical

# Configurações do eixo x
ax.set_xticklabels(df.columns[5:7])
ax.set_xlabel('Colunas')

# Configuração do título
ax.set_title('Boxplots das diferentes estrelas')

# Ajuste dos subplots
plt.subplots_adjust(bottom=0.25)

# Exibir o plot
plt.show()
del fig, ax


# 3. Distribuição da soma dos Números #

fig, ax = plt.subplots(figsize=(10, 6))  # Figura única com os vários BoxPlot
sns.histplot(data=df, x="SomaNumeros", kde=True, color="black")  # Extras: bins

# Configurações do histograma
plt.xlabel('Valores')
plt.ylabel('Soma')
plt.title('Histograma da Soma dos números')

# Exibir o plot
plt.show()
del fig, ax


# 4. Distribuição da soma das Estrelas #

# Criação da nova coluna
df["SomaEstrelas"] = df['Estrela1'] + df['Estrela2']

fig, ax = plt.subplots(figsize=(10, 6))  # Figura única com os vários BoxPlot
sns.histplot(data=df, x="SomaEstrelas", kde=True, color="black")  # Extras: bins

# Configurações do histograma
plt.xlabel('Valores')
plt.ylabel('Soma')
plt.title('Histograma da Soma das Estrelas')

# Exibir o plot
plt.show()
