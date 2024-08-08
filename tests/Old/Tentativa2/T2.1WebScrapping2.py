import datetime

import numpy as np
import pandas as pd
import urllib3
from bs4 import BeautifulSoup

# 0. Pré-requisitos do WebScrapping
# 0.1. Leitura do csv
df = pd.read_csv("../euromilhoes.csv")

# 0.2. Colocar os diferentes anos numa única lista
DiferentesAnos = df["Ano"].unique().tolist()

# 0.3. Criar um dicionário com o respetivo número de chaves de cada um dos anos
AllKeys = {}
for i in DiferentesAnos:
    AllKeys[i] = len(df[df["Ano"] == i])

# 0.4. Limpeza das coisas desnecessárias
del df
del DiferentesAnos

# 0.5. Nome das Respetivas Colunas

colunasTabela = [
    'Vencedores5+2',
    'Premio5+2',
    'Vencedores5+1',
    'Premio5+1',
    'Vencedores5+0',
    'Premio5+0',
    'Vencedores4+2',
    'Premio4+2',
    'Vencedores4+1',
    'Premio4+1',
    'Vencedores3+2',
    'Premio3+2',
    'Vencedores4+0',
    'Premio4+0',
    'Vencedores2+2',
    'Premio2+2',
    'Vencedores3+1',
    'Premio3+1',
    'Vencedores3+0',
    'Premio3+0',
    'Vencedores1+2',
    'Premio1+2',
    'Vencedores2+1',
    'Premio2+1',
    'Vencedores2+0',
    'Premio2+0',
    'DataBruta'
]


# 1. Aplicação do WebScrapping
# 1.1. Acessar a WebPage

lista_dados_estados = []
mudanca = False  # terça-feira, 10 maio 2011

for ano in range(2004, datetime.datetime.now().year + 1):
    for chaves in range(1, AllKeys[ano] + 1):
        numeroLink = str(int(str(ano)[2:]) * 1000 + chaves)
        if len(numeroLink) < 5:
            numeroLink = str(0) + str(numeroLink)
        url = f"https://www.euromillones.com/resultado-euromilhoes.asp?s={numeroLink}"  # Só muda os anos de cada um dos links
        conexao = urllib3.PoolManager()  # Realizar a conexão
        retorno = conexao.request("GET", url)  # Acessar o respetivo link
        pagina = BeautifulSoup(
            retorno.data, "html.parser"
        )  # Obter um HTML da respetiva “WebPage”
        tabela = pagina.find_all(
            "table", class_="tbl no-responsive ee size90 tbl-result no-back"
        )  # Acessar a tabela que possui os dados

        # Extra - Extrair a data
        data = pagina.find("span", class_="hide-responsive")
        texto_do_link = data.get_text()
        data = texto_do_link.split(">")[-1].strip()
        data = data.replace("\xa0", " ")
        if str(data) == 'terça-feira, 10 maio 2011':
            mudanca = True
        if str(data) == 'Último resultado':
            data = pagina.find("span", class_="hide-maxphablet")
            texto_do_link = data.get_text()
            data = texto_do_link.split(">")[-1].strip()
            data = data.replace("\xa0", " ")
        # print(str(data))
        # print(mudanca)

        # print(data) ---

        print(tabela)

        # 1.2. Recolher os dados de cada uma das células
        dado = []
        for celulas in tabela:
            celula = celulas.find_all("td")  # Encontrar as respetivas células da tabela
            for dados in celula:
                dado.append(dados.find(text=True))
            if ano <= 2011 and not mudanca:
                dado.append(None)
                dado.append(None)
            dado.append(data)

        # 1.2.1 Arranjar os dados
        for i in range(len(dado)):
            item = str(dado[i])
            if "+" not in item:
                lista_dados_estados.append(dado[i])

# 1.3. Colocar os dados numa matriz
colunas = 27  # Existem 14 colunas associadas a um jogo
linhas = len(lista_dados_estados) / colunas  # Cálculo das linhas
matriz_np = np.array(lista_dados_estados)  # Transformar em matriz
matriz_ajustada = np.reshape(
    matriz_np, (int(linhas), colunas)
)  # Reajustar a dimensão da matriz
print(matriz_ajustada)

# 1.4. Acrescentar os dados a um DataFrame Final
df = pd.DataFrame(matriz_ajustada, columns=colunasTabela)
df.reset_index(drop=True, inplace=True)  # Resetar o Índice do DataFrame
print(df)  # Breve Visualização dos dados

# 2. Guardar os dados num ficheiro csv
df.to_csv("PremioEuro.csv", index=False)
