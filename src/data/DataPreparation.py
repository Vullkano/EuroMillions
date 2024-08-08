import re

import pandas as pd
from IPython.display import display
from pathlib import Path
import os

# 0. Leitura do csv
p = Path.cwd()
p = p.parent.parent

data_folder = p / "data" / "raw"
nome_ficheiro = "euromilhoesTeste.csv"

caminho_completo = os.path.join(data_folder, nome_ficheiro)
df = pd.read_csv(caminho_completo)

# 2. Arranjar as Datas
df["Data"] = pd.to_datetime(df["Data"])
df['Dia'] = df['Data'].dt.day
df['Mes'] = df['Data'].dt.month
df['Ano'] = df['Data'].dt.year
df['NumeroSemana'] = df['Data'].dt.isocalendar().week

# 2.3.1 Criação da variável dummy que nos dá a info se é, ou não, sexta
df['SerFriday'] = df['DiaSemana'].apply(lambda x: 1 if x == 'Friday' else 0)

# 3 Criação da variável que possui a soma dos números
df["SomaNumeros"] = df['Numero1'] + df['Numero2'] + df['Numero3'] + df['Numero4'] + df['Numero5']

# 4 Criação da variável que possui a soma dos números
df["SomaEstrelas"] = df['Estrela1'] + df['Estrela2']

# 5. - Ordenar o DataFrame corretamente com o tempo
df.sort_values(by=["Ano", "Mes", "Dia"], ascending=True, inplace=True)  # Colocar por ordem Crescente a data
df.reset_index(drop=True, inplace=True)  # Resetar o Índice do DataFrame

display(df)

data_folder = p / "data" / "processed"
nome_ficheiro = "euromilhoesTesteCorrigido.csv"

caminho_completo = os.path.join(data_folder, nome_ficheiro)
df.to_csv(caminho_completo, index=False)

# # Extras para o futuro
# ## 1. Ajustar os prémios
# def converterpremio(valor):
#     padrao = r'\b(\d+)\b'
#     numero = re.findall(padrao, valor)
#     numero = int(numero[0])
#     numero *= 1000000
#     return numero
#
#
# if "PremioFeio" in df.columns:
#     df["Premio"] = df["PremioFeio"].apply(converterpremio)
#     df.drop("PremioFeio", axis=1, inplace=True)
