import re

import pandas as pd
from IPython.display import display

# 0. Leitura do csv
df = pd.read_csv("euromilhoes.csv")


# 1. Ajustar os prémios
def converterpremio(valor):
    padrao = r'\b(\d+)\b'
    numero = re.findall(padrao, valor)
    numero = int(numero[0])
    numero *= 1000000
    return numero


if "PremioFeio" in df.columns:
    df["Premio"] = df["PremioFeio"].apply(converterpremio)
    df.drop("PremioFeio", axis=1, inplace=True)

# 2. Arranjar as Datas
if "DiaSemana" in df.columns:
    # 2.1 Mês
    df["Mes"] = df["DiaSemana"].str[-3:]
    # 2.1.1 Associar um número a um mês diferente
    meses_numeros = {
        'jan': 1,
        'fev': 2,
        'mar': 3,
        'abr': 4,
        'mai': 5,
        'jun': 6,
        'jul': 7,
        'ago': 8,
        'set': 9,
        'out': 10,
        'nov': 11,
        'dez': 12
    }
    df['MesNumerico'] = df['Mes'].map(meses_numeros)
    # 2.2 Dia
    df['Dia'] = df['DiaSemana'].str[4:6]
    # 2.3 Dia da Semana
    df["Semana"] = df["DiaSemana"].str[:2]
    df.drop("DiaSemana", axis=1, inplace=True)
# 2.3.1 Criação da variável dummy que nos dá a info se é, ou não, sexta
df['ÉSexta?'] = df['Semana'].apply(lambda x: 1 if x == 'sx' else 0)

# 3 Criação da variável que possui a soma dos números
df["SomaNumeros"] = df['Numero1'] + df['Numero2'] + df['Numero3'] + df['Numero4'] + df['Numero5']
# 3.1 Criação de uma variável dummy que verifica se a soma dos números é ímpar ou não
df['ÉImparNumeros?'] = df['SomaNumeros'].apply(lambda x: 1 if x % 2 != 0 else 0)

# 4 Criação da variável que possui a soma dos números
df["SomaEstrelas"] = df['Estrela1'] + df['Estrela2']
# 4.1 Criação de uma variável dummy que verifica se a soma das estrelas é ímpar ou não
df['ÉImparEstrelas?'] = df['SomaEstrelas'].apply(lambda x: 1 if x % 2 != 0 else 0)

# Extra - Ordenar o DataFrame corretamente com o tempo
df.sort_values(by=["Ano", "MesNumerico", "Dia"], ascending=True, inplace=True)  # Colocar por ordem Crescente a data
df.reset_index(drop=True, inplace=True)  # Resetar o Índice do DataFrame

display(df)
df.to_csv("euromilhoes.csv", index=False)
