import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from scipy.stats import chi2_contingency

# Configuração global do estilo para uniformizar todos os gráficos
def configurar_estilo():
    """Configura o estilo uniforme para todos os gráficos"""
    # Usar um estilo mais moderno
    plt.style.use('default')
    
    # Configurações globais para um visual profissional
    plt.rcParams['figure.figsize'] = (14, 9)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.dpi'] = 150
    
    # Cores modernas e profissionais
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
        '#7209B7', '#4361EE', '#3A0CA3', '#560BAD', '#F72585'
    ])
    
    # Configurações de grid e eixos
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    plt.rcParams['figure.facecolor'] = 'white'

# =============================================================================
# GRÁFICOS DE DISTRIBUIÇÃO (BOXPLOT E VIOLINPLOT)
# =============================================================================

def plot_boxplot(df, titulo='Distribuição dos Valores - Boxplot'):
    """Boxplot para distribuição dos valores"""
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(16, 10))
     
    dados_numeros = df[df.columns]
     
    # Criar boxplot mais elegante
    box_plot = ax.boxplot([dados_numeros[col].dropna() for col in dados_numeros.columns], 
                          patch_artist=True, 
                          medianprops=dict(color='#2c3e50', linewidth=2),
                          boxprops=dict(facecolor='#3498db', alpha=0.7, edgecolor='#2980b9'),
                          whiskerprops=dict(color='#34495e', linewidth=1.5),
                          capprops=dict(color='#34495e', linewidth=1.5),
                          flierprops=dict(marker='o', markerfacecolor='#e74c3c', 
                                        markeredgecolor='#c0392b', markersize=4))
     
    # Personalizar o gráfico
    ax.set_xlabel('Números', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_ylabel('Valores', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    ax.set_xticklabels(dados_numeros.columns, rotation=45, ha='right')
    
    # Grid mais elegante
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    plt.tight_layout()
    

def plot_violinplot(df, titulo='Distribuição dos Números Principais - Violinplot'):
    """Violinplot para distribuição dos valores"""
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(16, 10))
     
    dados_numeros = df[df.columns]
     
    # Criar violinplot mais elegante
    violin_parts = ax.violinplot([dados_numeros[col].dropna() for col in dados_numeros.columns], 
                                 showmeans=True, showmedians=True)
    
    # Personalizar cores do violinplot
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(plt.cm.viridis(i/len(dados_numeros.columns)))
        pc.set_alpha(0.7)
        pc.set_edgecolor('#2c3e50')
    
    # Personalizar medianas e médias
    violin_parts['cmeans'].set_color('#e74c3c')
    violin_parts['cmeans'].set_linewidth(2)
    violin_parts['cmedians'].set_color('#f39c12')
    violin_parts['cmedians'].set_linewidth=2
     
    # Personalizar o gráfico
    ax.set_xlabel('Números', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_ylabel('Valores', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    ax.set_xticks(range(1, len(dados_numeros.columns) + 1))
    ax.set_xticklabels(dados_numeros.columns, rotation=45, ha='right')
    
    # Grid mais elegante
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    plt.tight_layout()
        

# =============================================================================
# GRÁFICOS DE EVOLUÇÃO TEMPORAL (LINE PLOT)
# =============================================================================

def plot_lineplot(df, colunas, coluna_data='date', titulo='Evolução ao Longo do Tempo', n_recentes=None):
    """Gera um line plot genérico.

    Parâmetros:
    - df: DataFrame com os dados
    - colunas: nome da coluna (str) ou lista de colunas (list[str]) a representar no eixo Y
    - coluna_data: nome da coluna de data/tempo
    - titulo: título do gráfico
    - n_recentes: opcional, usa apenas os n registos mais recentes
    """
    # Configurar estilo
    configurar_estilo()

    # Normalizar parâmetro de colunas
    if isinstance(colunas, str):
        colunas = [colunas]

    # Validar colunas existentes
    colunas_existentes = [c for c in colunas if c in df.columns]
    if not colunas_existentes:
        print("Nenhuma das colunas fornecidas existe no dataset.")
        return None

    fig, ax = plt.subplots(figsize=(16, 10))

    # Preparar data/ordenação
    if coluna_data in df.columns:
        df_temp = df.copy()
        df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data], errors='coerce')
        df_temp = df_temp.sort_values(coluna_data)
    else:
        df_temp = df.copy()
        df_temp['indice'] = range(len(df_temp))
        coluna_data = 'indice'

    # Aplicar filtro de mais recentes
    if n_recentes is not None and n_recentes < len(df_temp):
        df_temp = df_temp.tail(n_recentes)

    # Garantir colunas numéricas quando possível
    for c in colunas_existentes:
        df_temp[c] = pd.to_numeric(df_temp[c], errors='coerce')

    # Cores
    cores = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#7209B7', '#4361EE']

    # Plot por coluna
    for i, col in enumerate(colunas_existentes):
        cor = cores[i % len(cores)]
        ax.plot(
            df_temp[coluna_data],
            df_temp[col],
            linewidth=3,
            marker='o',
            markersize=6,
            label=col,
            color=cor,
            alpha=0.9,
        )

        # Anotações nos pontos
        for x, y in zip(df_temp[coluna_data], df_temp[col]):
            if pd.notna(y):
                ax.annotate(
                    f"{int(y)}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color=cor,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor='white',
                        edgecolor=cor,
                        alpha=0.85,
                    ),
                )

    # Personalização
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel('Tempo', fontweight='bold', color='#34495e')
    ax.set_ylabel('Valor', fontweight='bold', color='#34495e')

    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    legend = ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        title="Séries",
        title_fontsize=13,
        fontsize=11,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')

    if coluna_data != 'indice':
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()


# =============================================================================
# GRÁFICOS DE DISTRIBUIÇÃO (HISTOGRAMA)
# =============================================================================

def plot_histogram(df, colunas, titulo='Histograma', bins=30, densidade=False, acumulado=False, mostrar_valores=True):
    """Desenha histogramas para uma ou várias colunas numéricas.

    Parâmetros:
    - df: DataFrame com os dados
    - colunas: nome da coluna (str) ou lista de colunas (list[str])
    - titulo: título do gráfico
    - bins: número de bins (int) ou sequência de arestas
    - densidade: se True, normaliza para densidade (área=1)
    - acumulado: se True, usa histograma acumulado
    """
    # Configurar estilo
    configurar_estilo()

    if isinstance(colunas, str):
        colunas = [colunas]

    colunas_existentes = [c for c in colunas if c in df.columns]
    if not colunas_existentes:
        print("Nenhuma das colunas fornecidas existe no dataset.")
        return None

    fig, ax = plt.subplots(figsize=(16, 10))

    cores = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#7209B7', '#4361EE']

    for i, col in enumerate(colunas_existentes):
        cor = cores[i % len(cores)]
        serie = pd.to_numeric(df[col], errors='coerce').dropna()
        n, b, patches = ax.hist(
            serie,
            bins=bins,
            alpha=0.7,
            color=cor,
            edgecolor='white',
            linewidth=1.2,
            density=densidade,
            cumulative=acumulado,
            label=col,
        )

        # Adicionar valores no topo de cada barra
        if mostrar_valores:
            for rect, v in zip(patches, n):
                if not np.isfinite(v) or v == 0:
                    continue
                altura = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width()/2.0,
                    altura,
                    f"{v:.2f}" if densidade else f"{int(v)}",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold',
                    color='#2c3e50',
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor='white',
                        edgecolor=cor,
                        alpha=0.85,
                    ),
                )

    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel('Valor', fontweight='bold', color='#34495e')
    ax.set_ylabel('Densidade' if densidade else 'Frequência', fontweight='bold', color='#34495e')

    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    legend = ax.legend(frameon=True, fancybox=True, shadow=True, title="Variáveis", title_fontsize=13, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')

    plt.tight_layout()

# =============================================================================
# GRÁFICOS DE FREQUÊNCIA (BARPLOT)
# =============================================================================

def plot_barplot(df, colunas, top_n=20, orientacao='vertical', titulo=None):
    """Barplot empilhado das frequências dos valores nas colunas fornecidas.

    Parâmetros:
    - df: DataFrame com os dados
    - colunas: nome da coluna (str) ou lista de colunas (list[str]) a considerar
    - top_n: quantos valores mais frequentes mostrar (com base no total agregado)
    - orientacao: 'vertical' ou 'horizontal'
    - titulo: título do gráfico (opcional)
    """
    # Configurar estilo
    configurar_estilo()

    # Normalizar colunas
    if isinstance(colunas, str):
        colunas = [colunas]
    colunas_existentes = [c for c in colunas if c in df.columns]
    if not colunas_existentes:
        print("Nenhuma das colunas fornecidas existe no dataset.")
        return None

    # Frequências por coluna
    contagens_por_coluna = {}
    indice_total = set()
    for c in colunas_existentes:
        serie = pd.to_numeric(df[c], errors='ignore')
        contagens = serie.dropna().value_counts()
        contagens_por_coluna[c] = contagens
        indice_total.update(contagens.index.tolist())

    # Total agregado por categoria
    indice_total = list(indice_total)
    total_por_categoria = pd.Series(0, index=indice_total, dtype=float)
    for c in colunas_existentes:
        total_por_categoria = total_por_categoria.add(contagens_por_coluna[c].reindex(indice_total, fill_value=0), fill_value=0)

    # Top N categorias
    top_categorias = total_por_categoria.sort_values(ascending=False).head(top_n)
    categorias_ordenadas = top_categorias.index.tolist()

    # Matriz empilhada (linhas=categorias, colunas=colunas)
    dados_empilhados = pd.DataFrame(0, index=categorias_ordenadas, columns=colunas_existentes, dtype=float)
    for c in colunas_existentes:
        dados_empilhados[c] = contagens_por_coluna[c].reindex(categorias_ordenadas, fill_value=0)

    # Preparar figura
    fig, ax = plt.subplots(figsize=(16, 10))

    cores = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#7209B7', '#4361EE', '#3A0CA3']

    # Plot empilhado
    if str(orientacao).lower().startswith('h'):
        acumulado = np.zeros(len(categorias_ordenadas))
        for i, c in enumerate(colunas_existentes):
            valores = dados_empilhados[c].values
            ax.barh(
                y=np.arange(len(categorias_ordenadas)),
                width=valores,
                left=acumulado,
                color=cores[i % len(cores)],
                edgecolor='white',
                linewidth=1.2,
                alpha=0.9,
                label=c,
            )
            acumulado += valores

        ax.set_yticks(np.arange(len(categorias_ordenadas)))
        ax.set_yticklabels(categorias_ordenadas)
        ax.set_xlabel('Frequência', fontweight='bold', color='#34495e')
        ax.set_ylabel('Valores', fontweight='bold', color='#34495e')

        # Ordenação visual decrescente (maior no topo)
        ax.invert_yaxis()

        # Etiquetas com totais no fim de cada barra
        desvio = 0.01 * (acumulado.max() if acumulado.max() > 0 else 1)
        for yi, total in enumerate(acumulado):
            if total > 0:
                ax.text(
                    total + desvio,
                    yi,
                    f"{int(total)}",
                    va='center',
                    ha='left',
                    fontsize=10,
                    fontweight='bold',
                    color='#2c3e50',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor='white', edgecolor='#bdc3c7', alpha=0.85),
                )
    else:
        acumulado = np.zeros(len(categorias_ordenadas))
        x = np.arange(len(categorias_ordenadas))
        for i, c in enumerate(colunas_existentes):
            valores = dados_empilhados[c].values
            ax.bar(
                x=x,
                height=valores,
                bottom=acumulado,
                color=cores[i % len(cores)],
                edgecolor='white',
                linewidth=1.2,
                alpha=0.9,
                label=c,
            )
            acumulado += valores

        ax.set_xticks(x)
        ax.set_xticklabels(categorias_ordenadas, rotation=45, ha='right')
        ax.set_xlabel('Valores', fontweight='bold', color='#34495e')
        ax.set_ylabel('Frequência', fontweight='bold', color='#34495e')

        # Etiquetas com totais no topo de cada barra
        desvio = 0.01 * (acumulado.max() if acumulado.max() > 0 else 1)
        for xi, total in enumerate(acumulado):
            if total > 0:
                ax.text(
                    xi,
                    total + desvio,
                    f"{int(total)}",
                    va='bottom',
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    color='#2c3e50',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor='white', edgecolor='#bdc3c7', alpha=0.85),
                )

    # Título e estilo
    if titulo is None:
        titulo = f"Top {top_n} Valores Mais Frequentes (Empilhado)"
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)

    grelha_eixo = 'x' if str(orientacao).lower().startswith('h') else 'y'
    ax.grid(True, alpha=0.2, axis=grelha_eixo, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')

    legend = ax.legend(frameon=True, fancybox=True, shadow=True, title="Colunas", title_fontsize=13, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')

    plt.tight_layout()

# =============================================================================
# GRÁFICOS DE DISTRIBUIÇÃO (PIEPLOT)
# =============================================================================

def plot_pieplot(df, colunas, top_n=5, titulo='Distribuição', tipo='contagem', colormap='viridis'):
    """Pieplot genérico com opções de contagem, soma ou faixas.
    
    Parâmetros:
    - df: DataFrame com os dados
    - colunas: nome da coluna (str) ou lista de colunas (list[str])
    - top_n: máximo de categorias a mostrar (resto vai para "Outros")
    - titulo: título do gráfico
    - tipo: 'contagem' (value_counts), 'soma' (soma das colunas), 'faixas' (faixas da soma)
    - colormap: nome do colormap ('viridis', 'plasma', 'Set3', 'tab10', 'hsv', etc.)
    """
    # Configurar estilo
    configurar_estilo()
    
    # Normalizar colunas
    if isinstance(colunas, str):
        colunas = [colunas]
    colunas_existentes = [c for c in colunas if c in df.columns]
    if not colunas_existentes:
        print("Nenhuma das colunas fornecidas existe no dataset.")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Preparar dados conforme o tipo
    if tipo == 'contagem':
        # Contagem simples de valores únicos
        todos_valores = []
        for col in colunas_existentes:
            todos_valores.extend(df[col].dropna().tolist())
        contagem = pd.Series(todos_valores).value_counts()
        
    elif tipo == 'soma':
        # Soma das colunas
        df_temp = df.copy()
        df_temp['Soma'] = df_temp[colunas_existentes].sum(axis=1)
        contagem = df_temp['Soma'].value_counts().sort_index()
        
    elif tipo == 'faixas':
        # Faixas da soma das colunas
        df_temp = df.copy()
        df_temp['Soma'] = df_temp[colunas_existentes].sum(axis=1)
        
        # Criar faixas automáticas baseadas nos dados
        min_val, max_val = df_temp['Soma'].min(), df_temp['Soma'].max()
        if pd.isna(min_val) or pd.isna(max_val):
            print("Não foi possível calcular faixas com os dados fornecidos.")
            return None
            
        # Criar 5 faixas aproximadamente iguais
        faixas = pd.cut(df_temp['Soma'], bins=5, include_lowest=True)
        contagem = faixas.value_counts().sort_index()
        
    else:
        print(f"Tipo '{tipo}' não reconhecido. Use 'contagem', 'soma' ou 'faixas'.")
        return None
    
    # Limitar a top_n categorias e agrupar o resto em "Outros"
    if len(contagem) > top_n:
        top_categorias = contagem.head(top_n)
        outros_valor = contagem.iloc[top_n:].sum()
        
        # Criar nova série com "Outros"
        contagem_final = pd.concat([
            top_categorias,
            pd.Series([outros_valor], index=['Outros'])
        ])
    else:
        contagem_final = contagem
    
    # Usar colormap em vez de cores fixas
    try:
        cmap = plt.cm.get_cmap(colormap)
        cores = [cmap(i) for i in np.linspace(0, 1, len(contagem_final))]
    except:
        # Fallback para cores padrão se o colormap não existir
        print(f"Colormap '{colormap}' não encontrado. Usando cores padrão.")
        cores = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e']
    
    # Criar o pieplot com design melhorado
    wedges, texts, autotexts = ax.pie(
        contagem_final.values, 
        labels=contagem_final.index, 
        autopct='%1.1f%%', 
        colors=cores,
        startangle=90, 
        shadow=True, 
        explode=[0.05]*len(contagem_final),
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    
    # Personalizar o título
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    
    # Personalizar as percentagens
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Adicionar legenda com design melhorado
    legend = ax.legend(
        wedges, 
        [f"{idx} ({val})" for idx, val in contagem_final.items()], 
        title="Categorias", 
        loc="center left", 
        bbox_to_anchor=(1, 0, 0.5, 1),
        title_fontsize=13, 
        fontsize=11
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')
    
    plt.tight_layout()

# =============================================================================
# GRÁFICOS ADICIONAIS INTERESSANTES
# =============================================================================

def plot_pearson_correlation(df, titulo='Correlação de Pearson'):
    """Heatmap de correlação de Pearson para variáveis numéricas
    
    Parâmetros:
    - df: DataFrame com os dados
    - titulo: título do gráfico
    """
    # Configurar estilo
    configurar_estilo()
    
    # Selecionar colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    
    if len(colunas_numericas) < 2:
        print("Não há colunas numéricas suficientes para criar o heatmap de Pearson")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Preparar dados para Pearson
    df_pearson = df[colunas_numericas].copy()
    # Converter colunas object para numérico se possível
    for col in df_pearson.columns:
        if df_pearson[col].dtype == "O":
            df_pearson[col] = df_pearson[col].factorize(sort=True)[0]
    
    correlacao_pearson = df_pearson.corr(method="pearson")
    
    # Criar heatmap
    im = ax.imshow(correlacao_pearson, cmap='RdBu_r', aspect='auto', 
                   vmin=-1, vmax=1, alpha=0.8)
    
    # Adicionar valores de correlação
    for i in range(len(correlacao_pearson.columns)):
        for j in range(len(correlacao_pearson.columns)):
            valor = correlacao_pearson.iloc[i, j]
            if pd.notna(valor):
                cor_texto = 'white' if abs(valor) > 0.5 else 'black'
                ax.text(j, i, f'{valor:.2f}', ha='center', va='center', 
                       fontsize=9, fontweight='bold', color=cor_texto)
    
    # Personalizar eixos
    ax.set_xticks(range(len(correlacao_pearson.columns)))
    ax.set_yticks(range(len(correlacao_pearson.columns)))
    ax.set_xticklabels(correlacao_pearson.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlacao_pearson.columns)
    
    # Título
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    
    # Barra de cores
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlação de Pearson', fontweight='bold', color='#34495e', fontsize=12)
    
    # Estilo
    ax.grid(False)
    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()


def plot_cramer_correlation(df, titulo='V de Cramér'):
    """Heatmap de V de Cramér para variáveis categóricas
    
    Parâmetros:
    - df: DataFrame com os dados
    - titulo: título do gráfico
    """
    # Configurar estilo
    configurar_estilo()
    
    # Selecionar colunas categóricas
    colunas_categoricas = df.select_dtypes(include=['object', 'category', 'int64']).columns
    
    if len(colunas_categoricas) < 2:
        print("Não há colunas categóricas suficientes para criar o heatmap de V de Cramér")
        return None
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Função para calcular V de Cramér (implementação completa)
    def cramers_v(x, y):
        """Calcula o V de Cramér entre duas variáveis categóricas"""
        try:
            # Criar tabela de contingência
            confusion_matrix = pd.crosstab(x, y)
            
            # Verificar se a tabela tem dados válidos
            if confusion_matrix.empty or confusion_matrix.sum().sum() == 0:
                return 0.0
            
            # Teste chi-quadrado
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            
            # Calcular phi²
            phi2 = chi2 / n
            
            # Obter dimensões da tabela
            r, k = confusion_matrix.shape
            
            # Correção de bias para phi²
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            
            # Correção de bias para as dimensões
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            
            # Calcular V de Cramér corrigido
            if min((kcorr - 1), (rcorr - 1)) <= 0:
                return 0.0
            cramer_v = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
            
            # Garantir que o resultado seja um número válido
            if np.isnan(cramer_v) or np.isinf(cramer_v):
                return 0.0
                
            return float(cramer_v)
        except Exception as e:
            # Em caso de erro, retornar 0.0
            return 0.0
    
    # Preparar dados para V de Cramér
    df_cramer = df[colunas_categoricas].copy()
    
    # Converter colunas categóricas para o tipo 'category'
    for col in df_cramer.columns:
        df_cramer[col] = df_cramer[col].astype('category')
    
    # Calcular matriz de V de Cramér
    correlacao_cramer = pd.DataFrame(index=colunas_categoricas, columns=colunas_categoricas)
    
    for col1 in colunas_categoricas:
        for col2 in colunas_categoricas:
            if col1 == col2:
                correlacao_cramer.loc[col1, col2] = 1.0
            else:
                correlacao_cramer.loc[col1, col2] = cramers_v(df_cramer[col1], df_cramer[col2])
    
    # Remover colunas com valores ausentes
    correlacao_cramer = correlacao_cramer.dropna(axis=1, how='any')
    correlacao_cramer = correlacao_cramer.dropna(axis=0, how='any')
    
    if correlacao_cramer.empty:
        print("Não foi possível calcular V de Cramér com os dados fornecidos")
        return None
    
    # Garantir que todos os valores sejam numéricos
    correlacao_cramer = correlacao_cramer.astype(float)
    
    # Verificar se ainda há valores não numéricos
    if not correlacao_cramer.select_dtypes(include=[np.number]).columns.equals(correlacao_cramer.columns):
        print("Erro: Ainda há valores não numéricos na matriz de correlação")
        return None
    
    # Criar heatmap
    im = ax.imshow(correlacao_cramer, cmap='viridis', aspect='auto', 
                   vmin=0, vmax=1, alpha=0.8)
    
    # Adicionar valores de V de Cramér
    for i in range(len(correlacao_cramer.columns)):
        for j in range(len(correlacao_cramer.columns)):
            valor = correlacao_cramer.iloc[i, j]
            if pd.notna(valor):
                cor_texto = 'white' if valor > 0.5 else 'black'
                ax.text(j, i, f'{valor:.2f}', ha='center', va='center', 
                       fontsize=9, fontweight='bold', color=cor_texto)
    
    # Personalizar eixos
    ax.set_xticks(range(len(correlacao_cramer.columns)))
    ax.set_yticks(range(len(correlacao_cramer.columns)))
    ax.set_xticklabels(correlacao_cramer.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlacao_cramer.columns)
    
    # Título
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    
    # Barra de cores
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('V de Cramér', fontweight='bold', color='#34495e', fontsize=12)
    
    # Estilo
    ax.grid(False)
    ax.set_facecolor('#f8f9fa')
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()

def plot_estatisticas_resumo(df, colunas, titulo='Estatísticas Resumo das Colunas'):
    """Gráfico de barras com estatísticas resumo das colunas especificadas
    
    Parâmetros:
    - df: DataFrame com os dados
    - colunas: nome da coluna (str) ou lista de colunas (list[str]) a analisar
    - titulo: título do gráfico
    """
    # Configurar estilo
    configurar_estilo()
    
    # Normalizar colunas
    if isinstance(colunas, str):
        colunas = [colunas]
    
    colunas_existentes = [c for c in colunas if c in df.columns]
    if not colunas_existentes:
        print("Nenhuma das colunas fornecidas existe no dataset.")
        return None
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Calcular estatísticas básicas para cada coluna
    stats_por_coluna = {}
    for col in colunas_existentes:
        # Converter para numérico se possível
        serie = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(serie) > 0:
            stats_por_coluna[col] = serie.describe()
    
    if not stats_por_coluna:
        print("Não foi possível calcular estatísticas para nenhuma das colunas fornecidas.")
        return None
    
    # Preparar dados para o gráfico
    categorias = ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
    
    # Calcular estatísticas para cada coluna
    dados_grafico = {}
    for col, stats in stats_por_coluna.items():
        dados_grafico[col] = [
            stats.loc['mean'],      # Média
            stats.loc['50%'],       # Mediana
            stats.loc['std'],       # Desvio padrão
            stats.loc['min'],       # Mínimo
            stats.loc['max']        # Máximo
        ]
    
    # Preparar posições das barras
    x = np.arange(len(categorias))
    width = 0.8 / len(colunas_existentes)  # Largura das barras ajustada ao número de colunas
    
    # Cores modernas
    cores = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
    
    # Plot das barras para cada coluna
    barras = []
    for i, (col, valores) in enumerate(dados_grafico.items()):
        cor = cores[i % len(cores)]
        posicao = x - (len(colunas_existentes) - 1) * width / 2 + i * width
        
        bar = ax.bar(posicao, valores, width, label=col, 
                    color=cor, alpha=0.8, edgecolor=cor, linewidth=1.5)
        barras.append(bar)
        
        # Adicionar valores nas barras
        for j, (barra, valor) in enumerate(zip(bar, valores)):
            if pd.notna(valor):
                ax.text(barra.get_x() + barra.get_width()/2., barra.get_height() + 0.01,
                       f'{valor:.1f}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold', color='#2c3e50',
                       bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor='white', 
                                edgecolor=cor, 
                                alpha=0.8))
    
    # Personalizar o gráfico
    ax.set_xlabel('Estatísticas', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_ylabel('Valores', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontweight='bold')
    
    # Legend com design melhorado
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                      fontsize=12, title_fontsize=13, title="Colunas")
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')
    
    # Grid mais elegante
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    plt.tight_layout()