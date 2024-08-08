import pandas as pd
import re

# 0. Leitura do csv
df = pd.read_csv("PremioEuro.csv")

# Premio
ColunasPremio = [
    'Premio5+2',
    'Premio5+1',
    'Premio5+0',
    'Premio4+2',
    'Premio4+1',
    'Premio3+2',
    'Premio4+0',
    'Premio2+2',
    'Premio3+1',
    'Premio3+0',
    'Premio1+2',
    'Premio2+1',
    'Premio2+0',
]


# Função para extrair o valor numérico
def extrair_valor_moeda(texto):
    if pd.isnull(texto):
        return None
    valor = re.sub(r'[^\d.]', '', texto)  # Remover caracteres não numéricos, exceto o ponto decimal
    return float(valor)


for i in ColunasPremio:
    df[i] = df[i].apply(extrair_valor_moeda)
del i, ColunasPremio


# 2. Datas

# Dividir a coluna DataBruta em partes usando vírgulas e espaços
df[['DiaSemana', 'Resto']] = df['DataBruta'].str.split(', ', expand=True)

# Pegar nas diferentes partes da string
df['Dia'] = df['Resto'].str[0:3]
df['Mes'] = df['Resto'].str[3:6]
df['Ano'] = df['Resto'].str[6:].replace(r'\D', '', regex=True)

# Apagar a data Bruta
df.drop(['DataBruta', 'Resto'], axis=1, inplace=True)
