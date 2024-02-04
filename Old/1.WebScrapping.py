import datetime

import numpy as np
import pandas as pd
import urllib3
from bs4 import BeautifulSoup

# 0. Criação do suposto DataFrame Final
# Colunas do DataFrame que ainda necessita de ser tratado
colunasTabela = [
    'Pensar',
    'DiaSemana',
    'PremioFeio',
    'Numero1',
    'Numero2',
    'Numero3',
    'Numero4',
    'Numero5',
    'Estrela1',
    'Estrela2',
    'Lixo1',
    'Lixo2',
    'Lixo3',
    'Ano'
]
euromilhoes = pd.DataFrame(columns=colunasTabela)  # Criação do DataFrame Final

# 1. Aplicação do WebScrapping
# 1.1. Acessar a WebPage
for ano in range(2004, datetime.datetime.now().year + 1):
    url = f"https://www.euromillones.com/resultados-euromilhoes.asp?y={ano}"  # Só muda os anos de cada um dos links
    conexao = urllib3.PoolManager()  # Realizar a conexão
    retorno = conexao.request('GET', url)  # Acessar o respetivo link
    pagina = BeautifulSoup(retorno.data, 'html.parser')  # Obter um HTML da respetiva “WebPage”
    print(pagina)
    tabela = pagina.find_all('table',
                             class_="tbl no-responsive ee hover no-back")  # Acessar a tabvela que possui os dados

    # 1.2. Recolher os dados de cada uma das células
    dado = []
    for celulas in tabela:
        celula = celulas.find_all('td')  # Encontrar as respetivas células da tabela
        for dados in celula:
            dado.append(dados.find(text=True))

    # 1.2.1 Arranjar os dados
    lista_dados_estados = []
    LastColumn = 0
    for i in range(len(dado)):
        item = str(dado[i])
        if item[0] != "[prêmios]":
            lista_dados_estados.append(dado[i])
            LastColumn += 1
        if LastColumn == 13:  # Colocar o ano como um dado que, eventualmente, será útil no futuro
            lista_dados_estados.append(str(ano))
            LastColumn = 0

    # 1.3. Colocar os dados numa matriz
    colunas = 14  # Existem 14 colunas associadas a um jogo
    linhas = len(lista_dados_estados) / colunas  # Cálculo das linhas
    matriz_np = np.array(lista_dados_estados)  # Transformar em matriz
    matriz_ajustada = np.reshape(matriz_np, (int(linhas), colunas))  # Reajustar a dimensão da matriz

    # 1.4. Acrescentar os dados a um DataFrame Final
    df = pd.DataFrame(matriz_ajustada,
                      columns=colunasTabela)  # Criação do DataFrame com as colunas criadas anteriormente
    euromilhoes = pd.concat([euromilhoes, df])

euromilhoes.drop(["Pensar", "Lixo1", "Lixo2", "Lixo3"],
                 axis=1, inplace=True)  # Eliminar as colunas desnecessárias
euromilhoes.reset_index(drop=True, inplace=True)  # Resetar o Índice do DataFrame
print(euromilhoes)  # Breve Visualização dos dados

# 2. Guardar os dados num ficheiro csv
euromilhoes.to_csv("euromilhoes.csv", index=False)
