import datetime
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import urllib3
from bs4 import BeautifulSoup


def todas_tercas_e_sextas(desde_ano, ate_ano):
    if desde_ano > datetime.now().year or desde_ano < 2004:
        raise ValueError
    if desde_ano == 2004:
        data = datetime(desde_ano, 2, 13)
    else:
        data = datetime(desde_ano, 1, 1)

    # Cria uma lista para armazenar as datas
    datas_tercas_e_sextas = []

    # Itera sobre todos os dias desde desde_ano até a data final
    while data.year <= ate_ano and data <= datetime.now():
        if data.weekday() == 4 and data <= datetime(2011, 5, 6):
            datas_tercas_e_sextas.append(data.strftime("%d-%m-%Y"))
        # Adiciona a data à lista se for terça ou sexta-feira
        elif data > datetime(2011, 5, 6) and (
                data.weekday() == 1 or data.weekday() == 4):  # 0 = segunda, 1 = terça, ..., 6 = domingo
            datas_tercas_e_sextas.append(data.strftime("%d-%m-%Y"))

        # Avança para o próximo dia
        data += timedelta(days=1)

    return datas_tercas_e_sextas


# Define os anos de início e fim
desde_ano = 2004
ate_ano = datetime.now().year
# ate_ano = 2024

# Obtém a lista de datas
datas = todas_tercas_e_sextas(desde_ano, ate_ano)

colunasTabela = [
    'Data',
    'DiaSemana',
    'Numero1',
    'Numero2',
    'Numero3',
    'Numero4',
    'Numero5',
    'Estrela1',
    'Estrela2',
]

euromilhoes = pd.DataFrame(columns=colunasTabela)  # Criação do DataFrame Final
passos = 0
print(f'-- {passos / len(datas) * 100}% --')

for data in datas:
    url = f"https://www.euro-millions.com/results/{data}"  # Só muda os anos de cada um dos links
    print(url)
    conexao = urllib3.PoolManager()  # Realizar a conexão
    retorno = conexao.request('GET', url)  # Acessar o respetivo link
    pagina = BeautifulSoup(retorno.data, 'html.parser')  # Obter um HTML da respetiva “WebPage”
    tabelaNumero = pagina.find_all('li',
                                   class_="resultBall ball")  # Numeros do euromilhões
    tabelaEstrela = pagina.find_all('li',
                                    class_="resultBall lucky-star")

    # 1.2. Recolher os dados de cada uma das células

    dados = list()
    dados.append(data)
    dados.append(datetime.strptime(data, "%d-%m-%Y").strftime("%A"))

    # Adicionar os números
    for i, celula in enumerate(tabelaNumero):
        if i < 5:
            numero = celula.text
            dados.append(int(numero))

    # Adicionar as estrelas
    for i, celula in enumerate(tabelaEstrela):
        if i < 2:
            numero = celula.text
            dados.append(int(numero))

    print(dados)
    # 1.3. Colocar os dados numa matriz
    colunas = 9  # Existem 9 colunas associadas a um jogo
    linhas = len(dados) / colunas  # Cálculo das linhas
    matriz_np = np.array(dados)  # Transformar em matriz
    matriz_ajustada = np.reshape(matriz_np, (int(linhas), colunas))  # Reajustar a dimensão da matriz

    # 1.4. Acrescentar os dados a um DataFrame Final
    df = pd.DataFrame(matriz_ajustada,
                      columns=colunasTabela)  # Criação do DataFrame com as colunas criadas anteriormente
    euromilhoes = pd.concat([euromilhoes, df])
    euromilhoes.reset_index(drop=True, inplace=True)  # Resetar o Índice do DataFrame

    passos += 1
    print(f'{passos / len(datas) * 100}%')

# 2. Guardar os dados num ficheiro csv
euromilhoes.to_csv("euromilhoesTeste.csv", index=False)
