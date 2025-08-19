import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os

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

def plot_evolucao_tempo(df, coluna_data='Data', titulo='Evolução dos Valores ao Longo do Tempo', n_recentes=None):
    """Line plot para evolução dos valores ao longo do tempo em um único gráfico"""
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colunas_numeros = df.filter(like="N").columns

    # Converter coluna de data se necessário
    if coluna_data in df.columns:
        df_temp = df.copy()
        df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data])
        df_temp = df_temp.sort_values(coluna_data)
    else:
        # Se não houver coluna de data, usar o índice
        df_temp = df.copy()
        df_temp['indice'] = range(len(df_temp))
        coluna_data = 'indice'
    
    # Aplicar filtro para mostrar apenas os n valores mais recentes
    if n_recentes is not None and n_recentes < len(df_temp):
        df_temp = df_temp.tail(n_recentes)
    
    # Cores modernas e atrativas
    cores = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    # Plotar cada número como uma linha separada no mesmo gráfico
    for i, col in enumerate(colunas_numeros):
        cor = cores[i % len(cores)]
        linha = ax.plot(df_temp[coluna_data], df_temp[col], 
                       linewidth=3, marker='o', markersize=6, 
                       label=col, color=cor, alpha=0.8)
        
        # Adicionar valores nos pontos
        for x, y in zip(df_temp[coluna_data], df_temp[col]):
            ax.annotate(f'{int(y)}', (x, y), 
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center', 
                       fontsize=9, 
                       fontweight='bold',
                       color=cor,
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor='white', 
                                edgecolor=cor, 
                                alpha=0.8))
    
    # Personalizar o gráfico
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50')
    ax.set_xlabel('Tempo', fontweight='bold', color='#34495e')
    ax.set_ylabel('Valor', fontweight='bold', color='#34495e')
    
    # Grid mais elegante
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    # Legend com design melhorado
    legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                      frameon=True, fancybox=True, shadow=True,
                      title="Números", title_fontsize=13, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')
    
    if coluna_data != 'indice':
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    

def plot_evolucao_premio_tempo(df, coluna_premio='Premio', coluna_data='Data', titulo='Evolução do Prémio ao Longo do Tempo'):
    """Line plot para evolução do prémio ao longo do tempo"""
    if coluna_premio not in df.columns:
        print(f"Coluna '{coluna_premio}' não encontrada no dataset")
        return None
    
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Converter coluna de data se necessário
    if coluna_data in df.columns:
        df_temp = df.copy()
        df_temp[coluna_data] = pd.to_datetime(df_temp[coluna_data])
        df_temp = df_temp.sort_values(coluna_data)
    else:
        df_temp = df.copy()
        df_temp['indice'] = range(len(df_temp))
        coluna_data = 'indice'
    
    # Converter prémio para numérico se necessário
    df_temp[coluna_premio] = pd.to_numeric(df_temp[coluna_premio], errors='coerce')
    
    # Cor gradiente para o prémio
    cor_principal = '#E74C3C'
    cor_secundaria = '#C0392B'
    
    # Plotar linha principal
    ax.plot(df_temp[coluna_data], df_temp[coluna_premio], 
            linewidth=4, marker='o', markersize=8, 
            color=cor_principal, alpha=0.9, 
            markerfacecolor='white', markeredgecolor=cor_principal, markeredgewidth=2)
    
    # Adicionar valores nos pontos
    for x, y in zip(df_temp[coluna_data], df_temp[coluna_premio]):
        if pd.notna(y):
            ax.annotate(f'€{int(y):,}', (x, y), 
                       textcoords="offset points", 
                       xytext=(0,15), 
                       ha='center', 
                       fontsize=10, 
                       fontweight='bold',
                       color=cor_principal,
                       bbox=dict(boxstyle="round,pad=0.4", 
                                facecolor='white', 
                                edgecolor=cor_principal, 
                                alpha=0.9))
    
    # Personalizar o gráfico
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    ax.set_xlabel('Tempo', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_ylabel('Prémio (€)', fontweight='bold', color='#34495e', fontsize=14)
    
    # Grid mais elegante
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    # Formatar eixo Y para valores monetários
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{int(x):,}'))
    
    if coluna_data != 'indice':
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    

# =============================================================================
# GRÁFICOS DE FREQUÊNCIA (BARPLOT)
# =============================================================================

def plot_frequencia_numeros(df, top_n=20, titulo=None):
    """Barplot com os números que mais saíram"""
    if titulo is None:
        titulo = f'Top {top_n} Números Mais Frequentes'
    
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colunas_numeros = [col for col in df.columns if 'Numero' in col or col.isdigit()][:5]
    if not colunas_numeros:
        colunas_numeros = df.columns[:5]
    
    # Contar frequência de cada número
    todos_numeros = []
    for col in colunas_numeros:
        todos_numeros.extend(df[col].dropna().tolist())
    
    # Contar frequências
    from collections import Counter
    freq_numeros = Counter(todos_numeros)
    
    # Pegar os top N mais frequentes
    top_freq = dict(sorted(freq_numeros.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    # Cores modernas e gradientes
    cores = plt.cm.viridis(np.linspace(0, 1, len(top_freq)))
    
    # Criar o gráfico com barras mais elegantes
    bars = ax.bar(range(len(top_freq)), list(top_freq.values()), 
                   color=cores, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Personalizar o gráfico
    ax.set_xlabel('Números', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_ylabel('Frequência', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    ax.set_xticks(range(len(top_freq)))
    ax.set_xticklabels(list(top_freq.keys()))
    
    # Grid mais elegante
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    # Adicionar valores nas barras com design melhorado
    for i, (bar, v) in enumerate(zip(bars, top_freq.values())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, 
               str(v), ha='center', va='bottom', 
               fontsize=11, fontweight='bold', color='#2c3e50',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='white', 
                        edgecolor=cores[i], 
                        alpha=0.8))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    

def plot_frequencia_estrelas(df, top_n=12, titulo=None):
    """Barplot com as estrelas que mais saíram"""
    if titulo is None:
        titulo = f'Top {top_n} Estrelas Mais Frequentes'
    
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    colunas_estrelas = [col for col in df.columns if 'Estrela' in col or 'Star' in col][:2]
    if not colunas_estrelas:
        colunas_estrelas = df.columns[5:7] if len(df.columns) >= 7 else df.columns[-2:]
    
    # Contar frequência de cada estrela
    todas_estrelas = []
    for col in colunas_estrelas:
        todas_estrelas.extend(df[col].dropna().tolist())
    
    # Contar frequências
    from collections import Counter
    freq_estrelas = Counter(todas_estrelas)
    
    # Pegar as top N mais frequentes
    top_freq = dict(sorted(freq_estrelas.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    # Cores modernas e gradientes para estrelas
    cores = plt.cm.plasma(np.linspace(0, 1, len(top_freq)))
    
    # Criar o gráfico com barras mais elegantes
    bars = ax.bar(range(len(top_freq)), list(top_freq.values()), 
                   color=cores, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Personalizar o gráfico
    ax.set_xlabel('Estrelas', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_ylabel('Frequência', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    ax.set_xticks(range(len(top_freq)))
    ax.set_xticklabels(list(top_freq.keys()))
    
    # Grid mais elegante
    ax.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.8)
    ax.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    
    # Adicionar valores nas barras com design melhorado
    for i, (bar, v) in enumerate(zip(bars, top_freq.values())):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, 
               str(v), ha='center', va='bottom', 
               fontsize=11, fontweight='bold', color='#2c3e50',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='white', 
                        edgecolor=cores[i], 
                        alpha=0.8))
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    

# =============================================================================
# GRÁFICOS DE DISTRIBUIÇÃO (PIEPLOT)
# =============================================================================

def plot_distribuicao_vencedores(df, coluna_vencedor='Vencedor', titulo='Distribuição de Vencedores do Primeiro Prémio'):
    """Pieplot com a distribuição se houve vencedor do primeiro prémio"""
    if coluna_vencedor not in df.columns:
        print(f"Coluna '{coluna_vencedor}' não encontrada no dataset")
        return None
    
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Contar valores únicos
    contagem = df[coluna_vencedor].value_counts()
    
    # Cores modernas e atrativas
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Criar o pieplot com design melhorado
    wedges, texts, autotexts = ax.pie(contagem.values, labels=contagem.index, 
                                       autopct='%1.1f%%', colors=colors[:len(contagem)],
                                       startangle=90, shadow=True, explode=[0.05]*len(contagem),
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Personalizar o título
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    
    # Personalizar as percentagens
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # Adicionar legenda com design melhorado
    legend = ax.legend(wedges, contagem.index, title="Status", 
                      loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                      title_fontsize=13, fontsize=11)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')
    
    plt.tight_layout()
    

def plot_distribuicao_soma_numeros(df, titulo='Distribuição da Soma dos Números por Faixas'):
    """Pieplot com a distribuição da soma dos números por faixas"""
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colunas_numeros = [col for col in df.columns if 'Numero' in col or col.isdigit()][:5]
    if not colunas_numeros:
        colunas_numeros = df.columns[:5]
    
    # Calcular soma dos números
    df_temp = df.copy()
    df_temp['SomaNumeros'] = df_temp[colunas_numeros].sum(axis=1)
    
    # Criar faixas
    faixas = ['0-50', '51-100', '101-150', '151-200', '201+']
    limites = [0, 50, 100, 150, 200, float('inf')]
    
    df_temp['Faixa'] = pd.cut(df_temp['SomaNumeros'], bins=limites, labels=faixas, right=False)
    contagem_faixas = df_temp['Faixa'].value_counts()
    
    # Cores modernas e atrativas
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Criar o pieplot com design melhorado
    wedges, texts, autotexts = ax.pie(contagem_faixas.values, labels=contagem_faixas.index, 
                                       autopct='%1.1f%%', colors=colors[:len(contagem_faixas)],
                                       startangle=90, shadow=True, explode=[0.05]*len(contagem_faixas),
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Personalizar o título
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    
    # Personalizar as percentagens
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    

def plot_distribuicao_soma_estrelas(df, titulo='Distribuição da Soma das Estrelas'):
    """Pieplot com a distribuição da soma das estrelas"""
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colunas_estrelas = [col for col in df.columns if 'Estrela' in col or 'Star' in col][:2]
    if not colunas_estrelas:
        colunas_estrelas = df.columns[5:7] if len(df.columns) >= 7 else df.columns[-2:]
    
    # Calcular soma das estrelas
    df_temp = df.copy()
    df_temp['SomaEstrelas'] = df_temp[colunas_estrelas].sum(axis=1)
    
    # Contar valores únicos
    contagem = df_temp['SomaEstrelas'].value_counts().sort_index()
    
    # Cores modernas e atrativas para estrelas
    colors = ['#f39c12', '#e67e22', '#d35400', '#e74c3c', '#c0392b', '#a93226']
    
    # Criar o pieplot com design melhorado
    wedges, texts, autotexts = ax.pie(contagem.values, labels=contagem.index, 
                                       autopct='%1.1f%%', colors=colors[:len(contagem)],
                                       startangle=90, shadow=True, explode=[0.05]*len(contagem),
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    # Personalizar o título
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    
    # Personalizar as percentagens
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    

# =============================================================================
# GRÁFICOS ADICIONAIS INTERESSANTES
# =============================================================================

def plot_heatmap_correlacao(df, titulo='Matriz de Correlação entre Números e Estrelas'):
    """Heatmap de correlação entre números e estrelas"""
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(14, 12))
     
    # Selecionar colunas numéricas
    colunas_numericas = df.select_dtypes(include=[np.number]).columns
     
    if len(colunas_numericas) < 2:
        print("Não há colunas numéricas suficientes para criar o heatmap")
        return None
     
    # Calcular correlação
    correlacao = df[colunas_numericas].corr()
     
    # Criar heatmap mais elegante
    im = ax.imshow(correlacao, cmap='RdBu_r', center=0, aspect='auto', 
                   vmin=-1, vmax=1, alpha=0.8)
    
    # Adicionar valores de correlação
    for i in range(len(correlacao.columns)):
        for j in range(len(correlacao.columns)):
            valor = correlacao.iloc[i, j]
            cor_texto = 'white' if abs(valor) > 0.5 else 'black'
            ax.text(j, i, f'{valor:.2f}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=cor_texto)
    
    # Personalizar eixos
    ax.set_xticks(range(len(correlacao.columns)))
    ax.set_yticks(range(len(correlacao.columns)))
    ax.set_xticklabels(correlacao.columns, rotation=45, ha='right')
    ax.set_yticklabels(correlacao.columns)
    
    # Personalizar o gráfico
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    
    # Adicionar barra de cores
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlação', fontweight='bold', color='#34495e', fontsize=12)
    
    # Grid mais elegante
    ax.grid(False)
    ax.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    

def plot_distribuicao_combinada(df, titulo='Distribuição Combinada de Números e Estrelas'):
    """Histograma combinado da distribuição de números e estrelas"""
    # Configurar estilo
    configurar_estilo()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
     
    colunas_numeros = [col for col in df.columns if 'Numero' in col or col.isdigit()][:5]
    colunas_estrelas = [col for col in df.columns if 'Estrela' in col or 'Star' in col][:2]
     
    if not colunas_numeros:
        colunas_numeros = df.columns[:5]
    if not colunas_estrelas:
        colunas_estrelas = df.columns[5:7] if len(df.columns) >= 7 else df.columns[-2:]
     
    # Distribuição dos números
    todos_numeros = []
    for col in colunas_numeros:
        todos_numeros.extend(df[col].dropna().tolist())
     
    # Histograma dos números com design melhorado
    n1, bins1, patches1 = ax1.hist(todos_numeros, bins=50, alpha=0.8, 
                                   color='#3498db', edgecolor='#2980b9', linewidth=1.5)
    ax1.set_title('Distribuição dos Números', fontweight='bold', color='#2c3e50', fontsize=16)
    ax1.set_xlabel('Valores', fontweight='bold', color='#34495e', fontsize=12)
    ax1.set_ylabel('Frequência', fontweight='bold', color='#34495e', fontsize=12)
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos para ax1
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#bdc3c7')
    ax1.spines['bottom'].set_color('#bdc3c7')
     
    # Distribuição das estrelas
    todas_estrelas = []
    for col in colunas_estrelas:
        todas_estrelas.extend(df[col].dropna().tolist())
     
    # Histograma das estrelas com design melhorado
    n2, bins2, patches2 = ax2.hist(todas_estrelas, bins=20, alpha=0.8, 
                                   color='#e74c3c', edgecolor='#c0392b', linewidth=1.5)
    ax2.set_title('Distribuição das Estrelas', fontweight='bold', color='#2c3e50', fontsize=16)
    ax2.set_xlabel('Valores', fontweight='bold', color='#34495e', fontsize=12)
    ax2.set_ylabel('Frequência', fontweight='bold', color='#34495e', fontsize=12)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)
    ax2.set_facecolor('#f8f9fa')
    
    # Eixos mais limpos para ax2
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#bdc3c7')
    ax2.spines['bottom'].set_color('#bdc3c7')
     
    plt.suptitle(titulo, fontsize=20, fontweight='bold', color='#2c3e50', y=0.98)
    plt.tight_layout()
    

def plot_estatisticas_resumo(df, titulo='Estatísticas Resumo: Números vs Estrelas'):
    """Gráfico de barras com estatísticas resumo dos dados"""
    # Configurar estilo
    configurar_estilo()
    
    fig, ax = plt.subplots(figsize=(16, 10))
     
    # Calcular estatísticas básicas
    colunas_numeros = [col for col in df.columns if 'Numero' in col or col.isdigit()][:5]
    colunas_estrelas = [col for col in df.columns if 'Estrela' in col or 'Star' in col][:2]
     
    if not colunas_numeros:
        colunas_numeros = df.columns[:5]
    if not colunas_estrelas:
        colunas_estrelas = df.columns[5:7] if len(df.columns) >= 7 else df.columns[-2:]
     
    # Estatísticas dos números
    numeros_stats = df[colunas_numeros].describe()
    estrelas_stats = df[colunas_estrelas].describe()
     
    # Preparar dados para o gráfico
    categorias = ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo']
     
    # Média dos números
    media_numeros = numeros_stats.loc['mean'].mean()
    # Mediana dos números
    mediana_numeros = numeros_stats.loc['50%'].mean()
    # Desvio padrão dos números
    std_numeros = numeros_stats.loc['std'].mean()
    # Mínimo dos números
    min_numeros = numeros_stats.loc['min'].min()
    # Máximo dos números
    max_numeros = numeros_stats.loc['max'].max()
     
    # Média das estrelas
    media_estrelas = estrelas_stats.loc['mean'].mean()
    # Mediana das estrelas
    mediana_estrelas = estrelas_stats.loc['50%'].mean()
    # Desvio padrão das estrelas
    std_estrelas = estrelas_stats.loc['std'].mean()
    # Mínimo das estrelas
    min_estrelas = estrelas_stats.loc['min'].min()
    # Máximo das estrelas
    max_estrelas = estrelas_stats.loc['max'].max()
     
    valores_numeros = [media_numeros, mediana_numeros, std_numeros, min_numeros, max_numeros]
    valores_estrelas = [media_estrelas, mediana_estrelas, std_estrelas, min_estrelas, max_estrelas]
     
    x = np.arange(len(categorias))
    width = 0.35
     
    # Cores modernas
    cor_numeros = '#3498db'
    cor_estrelas = '#e74c3c'
     
    bars1 = ax.bar(x - width/2, valores_numeros, width, label='Números', 
                   color=cor_numeros, alpha=0.8, edgecolor='#2980b9', linewidth=1.5)
    bars2 = ax.bar(x + width/2, valores_estrelas, width, label='Estrelas', 
                   color=cor_estrelas, alpha=0.8, edgecolor='#c0392b', linewidth=1.5)
     
    # Personalizar o gráfico
    ax.set_xlabel('Estatísticas', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_ylabel('Valores', fontweight='bold', color='#34495e', fontsize=14)
    ax.set_title(titulo, fontweight='bold', pad=20, color='#2c3e50', fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontweight='bold')
    
    # Legend com design melhorado
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                      fontsize=12, title_fontsize=13)
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
     
    # Adicionar valores nas barras com design melhorado
    for bars, cor in [(bars1, cor_numeros), (bars2, cor_estrelas)]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold', color='#2c3e50',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor='white', 
                            edgecolor=cor, 
                            alpha=0.8))
     
    plt.tight_layout()
    

# =============================================================================
# FUNÇÃO PRINCIPAL PARA EXECUTAR TODOS OS GRÁFICOS
# =============================================================================

def executar_todos_graficos(df, salvar_graficos=False, pasta_destino='graficos'):
    """Executa todos os gráficos disponíveis no dataset"""
    # Configurar estilo
    configurar_estilo()
    
    # Lista de todas as funções de gráfico
    funcoes_graficos = [
        plot_boxplot_numeros,
        plot_violinplot_numeros,
        plot_boxplot_estrelas,
        plot_violinplot_estrelas,
        plot_evolucao_numeros_tempo,
        plot_evolucao_estrelas_tempo,
        plot_evolucao_premio_tempo,
        plot_frequencia_numeros,
        plot_frequencia_estrelas,
        plot_distribuicao_vencedores,
        plot_distribuicao_soma_numeros,
        plot_distribuicao_soma_estrelas,
        plot_heatmap_correlacao,
        plot_distribuicao_combinada,
        plot_estatisticas_resumo
    ]
    
    graficos_gerados = []
    
    for i, funcao in enumerate(funcoes_graficos):
        try:
            print(f"Gerando gráfico {i+1}/{len(funcoes_graficos)}: {funcao.__name__}")
            fig = funcao(df)
            
            if fig is not None:
                graficos_gerados.append((funcao.__name__, fig))
                
                if salvar_graficos:
                    # Criar pasta se não existir
                    os.makedirs(pasta_destino, exist_ok=True)
                    nome_arquivo = f"{funcao.__name__}.png"
                    caminho_completo = os.path.join(pasta_destino, nome_arquivo)
                    fig.savefig(caminho_completo, dpi=300, bbox_inches='tight')
                    print(f"  Gráfico salvo: {caminho_completo}")
                
                plt.show()
                plt.close(fig)
            else:
                print(f"  Gráfico não pôde ser gerado: {funcao.__name__}")
                
        except Exception as e:
            print(f"  Erro ao gerar gráfico {funcao.__name__}: {str(e)}")
    
    print(f"\nTotal de gráficos gerados com sucesso: {len(graficos_gerados)}")
    return graficos_gerados

# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    # Exemplo de como usar as funções
    print("Módulo de visualização do Euromilhões carregado!")
    print("Use a função 'executar_todos_graficos(df)' para gerar todos os gráficos")
    print("Ou use funções individuais como 'plot_boxplot_numeros(df)'")
