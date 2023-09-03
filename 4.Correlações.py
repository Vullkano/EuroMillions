import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

df = pd.read_csv("euromilhoes.csv")

cols_to_drop = ['Mes', 'Semana']
df.drop(cols_to_drop, axis=1, inplace=True)

VarNum = ['Numero1', 'Numero2', 'Numero3', 'Numero4', 'Numero5', 'Estrela1',
          'Estrela2', 'Ano', 'Premio', 'Dia', 'SomaNumeros', 'SomaEstrelas']

VarCat = [i for i in df.columns if i not in VarNum]

# Visualização das Correlações
# 1. Correlação de Pearson

dfCorr = df[VarNum]
# plot correlation matrix
fig, ax = plt.subplots(figsize=(10, 10))
for col in dfCorr.columns:
    if dfCorr[col].dtype == "O":
        dfCorr[col] = dfCorr[col].factorize(sort=True)[0]
corr_matrix = dfCorr.corr(method="pearson")
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.5f', cmap="YlGnBu", cbar=True, linewidths=0.5, ax=ax)
plt.title("Correlação de Pearson para as features")
# plt.savefig("Pearson.svg", format="svg")
del dfCorr


# 2. Correlação de VCramer

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    cramer_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return cramer_v


# Converter colunas categóricas para o tipo 'category' (se necessário)
for col in VarCat:
    df[col] = df[col].astype('category')

# Calcular a matriz de correlação
corr_matrix = pd.DataFrame(index=VarCat, columns=VarCat)
for col1 in VarCat:
    for col2 in VarCat:
        corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])

# Remover colunas com valores ausentes na matriz de correlação
corr_matrix = corr_matrix.dropna(axis=1, how='any')

# Plotar o heatmap
plt.figure(figsize=(10, 8))
# rocket, coolwarm
sns.heatmap(corr_matrix.astype(float), annot=True, cmap='bone_r')
plt.title('Matriz de Correlação - Coeficiente de Cramer')
# plt.savefig("Cramer.svg", format="svg")
plt.show()


# 3. Correlação de ETA

def eta_coefficient(y, x):  # O X tem de ser numérico
    # Verificar se y é numérico
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("A variável y deve ser numérica.")

    categories, counts = np.unique(x, return_counts=True)
    l = len(categories)
    m = np.empty(l)
    qual = np.asarray(x)
    for k in range(l):
        m[k] = np.mean(y[qual == categories[k]])
    numerator = np.sum(counts * (m - np.mean(y)) ** 2)
    denominator = np.sum((y - np.mean(y)) ** 2)
    eta = np.sqrt(numerator / denominator)
    return eta


for j in VarCat:
    for i in VarNum:
        eta1 = eta_coefficient(df[str(i)], df[str(j)])
        print(f"Coeficiente de Eta entre o {j} e {i}: {eta1}")
