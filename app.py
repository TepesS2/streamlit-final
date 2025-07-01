import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Dashboard de Tabagismo e Fatores de Risco",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega dados sintéticos de tabagismo e fatores de risco"""
    try:
        # Verifica se os dados já existem localmente
        data_path = Path("data/smoking_data.csv")
        
        if data_path.exists():
            return pd.read_csv(data_path)
        
        # Gera dados sintéticos
        np.random.seed(42)
        n_samples = 2500
        
        # Gera dados demográficos
        idades = np.random.normal(45, 15, n_samples).astype(int)
        idades = np.clip(idades, 18, 80)
        
        sexos = np.random.choice(['Masculino', 'Feminino'], n_samples, p=[0.52, 0.48])
        
        # Estados com probabilidades realistas
        estados = np.random.choice(
            ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'ES', 'MT', 'DF', 'PE'], 
            n_samples, 
            p=[0.25, 0.15, 0.12, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.04, 0.03, 0.07]
        )
        
        # Escolaridade baseada na idade
        escolaridade_probs = {
            'Fundamental': 0.3, 'Médio': 0.4, 'Superior': 0.25, 'Pós-graduação': 0.05
        }
        escolaridades = np.random.choice(
            list(escolaridade_probs.keys()), 
            n_samples, 
            p=list(escolaridade_probs.values())
        )
        
        # Renda correlacionada com escolaridade
        renda_base = {'Fundamental': 2000, 'Médio': 3500, 'Superior': 6000, 'Pós-graduação': 10000}
        rendas = []
        for esc in escolaridades:
            base = renda_base[esc]
            renda = max(1000, np.random.normal(base, base*0.3))
            rendas.append(round(renda, 2))
        
        # Status de tabagismo
        fumantes = np.random.choice(['Sim', 'Não'], n_samples, p=[0.15, 0.85])
        
        # Ex-fumantes (só entre não fumantes atuais)
        ex_fumantes = []
        for i, fumante in enumerate(fumantes):
            if fumante == 'Não':
                ex_fumantes.append(np.random.choice(['Sim', 'Não'], p=[0.25, 0.75]))
            else:
                ex_fumantes.append('Não')
        
        # Cigarros por dia (só para fumantes)
        cigarros_dia = []
        for fumante in fumantes:
            if fumante == 'Sim':
                cigarros_dia.append(max(1, int(np.random.exponential(8))))
            else:
                cigarros_dia.append(0)
        
        # Álcool
        consome_alcool = np.random.choice(['Sim', 'Não'], n_samples, p=[0.6, 0.4])
        
        # Exercício
        exercita = np.random.choice(['Sim', 'Não'], n_samples, p=[0.35, 0.65])
        
        # IMC
        imcs = np.random.normal(26, 4, n_samples)
        imcs = np.clip(imcs, 16, 45)
        
        # Categorias de IMC
        categorias_imc = []
        for imc in imcs:
            if imc < 18.5:
                categorias_imc.append('Abaixo do peso')
            elif imc < 25:
                categorias_imc.append('Peso normal')
            elif imc < 30:
                categorias_imc.append('Sobrepeso')
            else:
                categorias_imc.append('Obesidade')
        
        # Pressão arterial correlacionada com idade e IMC
        pressao_sistolica = 110 + (idades - 30) * 0.5 + (imcs - 25) * 2 + np.random.normal(0, 10, n_samples)
        pressao_sistolica = np.clip(pressao_sistolica, 90, 200)
        
        pressao_diastolica = 70 + (idades - 30) * 0.3 + (imcs - 25) * 1 + np.random.normal(0, 8, n_samples)
        pressao_diastolica = np.clip(pressao_diastolica, 60, 120)
        
        # Diabetes correlacionado com idade e IMC
        prob_diabetes = 0.02 + (idades - 30) * 0.002 + np.maximum(0, imcs - 25) * 0.01
        diabetes = np.random.binomial(1, prob_diabetes, n_samples)
        diabetes = ['Sim' if d else 'Não' for d in diabetes]
        
        # Doenças cardíacas correlacionadas com idade, tabagismo e pressão
        prob_cardiaca = 0.01 + (idades - 30) * 0.003
        prob_cardiaca += np.where(np.array(fumantes) == 'Sim', 0.05, 0)
        prob_cardiaca += (pressao_sistolica - 120) * 0.001
        prob_cardiaca = np.clip(prob_cardiaca, 0, 0.5)
        
        doenca_cardiaca = np.random.binomial(1, prob_cardiaca, n_samples)
        doenca_cardiaca = ['Sim' if d else 'Não' for d in doenca_cardiaca]
        
        # Cria o DataFrame
        df = pd.DataFrame({
            'idade': idades,
            'sexo': sexos,
            'estado': estados,
            'escolaridade': escolaridades,
            'renda_mensal': rendas,
            'fumante': fumantes,
            'ex_fumante': ex_fumantes,
            'cigarros_por_dia': cigarros_dia,
            'consome_alcool': consome_alcool,
            'pratica_exercicios': exercita,
            'imc': imcs,
            'categoria_imc': categorias_imc,
            'pressao_sistolica': pressao_sistolica,
            'pressao_diastolica': pressao_diastolica,
            'diabetes': diabetes,
            'doenca_cardiaca': doenca_cardiaca
        })
        
        # Salva os dados localmente
        data_path.parent.mkdir(exist_ok=True)
        df.to_csv(data_path, index=False)
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao gerar dados sintéticos: {str(e)}")
        return None

def add_sidebar_filters(df):
    """Adiciona filtros na barra lateral e retorna o dataframe filtrado"""
    st.sidebar.markdown("## 🔍 Filtros")
    
    # Filtro de idade
    min_age, max_age = st.sidebar.slider(
        "Faixa Etária",
        min_value=int(df['idade'].min()),
        max_value=int(df['idade'].max()),
        value=(int(df['idade'].min()), int(df['idade'].max()))
    )
    
    # Filtro de sexo
    sexos = st.sidebar.multiselect(
        "Sexo",
        options=df['sexo'].unique(),
        default=df['sexo'].unique()
    )
    
    # Filtro de escolaridade
    escolaridades = st.sidebar.multiselect(
        "Escolaridade",
        options=df['escolaridade'].unique(),
        default=df['escolaridade'].unique()
    )
    
    # Filtro de status de fumante
    status_fumante = st.sidebar.multiselect(
        "Status de Fumante",
        options=df['fumante'].unique(),
        default=df['fumante'].unique()
    )
    
    # Aplica filtros
    filtered_df = df[
        (df['idade'] >= min_age) & 
        (df['idade'] <= max_age) &
        (df['sexo'].isin(sexos)) &
        (df['escolaridade'].isin(escolaridades)) &
        (df['fumante'].isin(status_fumante))
    ]
    
    st.sidebar.markdown(f"**Registros filtrados: {len(filtered_df):,}**")
    
    return filtered_df

def show_home_page(df):
    """Página inicial com documentação e visão geral"""
    
    st.markdown("## 🎯 Objetivo do Dashboard")
    st.markdown("""
    Este dashboard interativo explora padrões de tabagismo e fatores de risco associados através de visualização 
    abrangente de dados. A análise ajuda a identificar relações entre hábitos de fumar, demografia, métricas de 
    saúde e fatores de estilo de vida.
    """)
    
    st.markdown("## 🧭 Como Navegar")
    st.markdown("""
    - **📈 Análise Geral**: Estatísticas gerais e prevalência do tabagismo
    - **🔍 Análise Demográfica**: Padrões de idade, gênero e educação
    - **🏥 Métricas de Saúde**: IMC, condições de saúde e correlações médicas
    - **🎯 Fatores de Risco**: Fatores de estilo de vida e avaliação de risco
    - **📊 Explorador Interativo**: Análise customizável com filtros e interações
    """)
    
    st.markdown("## 🔧 Funcionalidade dos Filtros")
    st.markdown("""
    Use os filtros da barra lateral para:
    - Filtrar por faixas etárias, gênero e nível educacional
    - Selecionar condições de saúde específicas ou faixas de IMC
    - Focar em status particulares de tabagismo
    - Personalizar visualizações em tempo real
    """)
    
    # Visão geral do dataset
    st.markdown("## 📋 Visão Geral do Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Registros", len(df))
    
    with col2:
        st.metric("Colunas", len(df.columns))
    
    with col3:
        smoking_rate = (df['fumante'].value_counts().get('Sim', 0) / len(df) * 100)
        st.metric("Taxa de Tabagismo", f"{smoking_rate:.1f}%")
    
    with col4:
        st.metric("Faixa Etária", f"{df['idade'].min()}-{df['idade'].max()}")
    
    # Dados de amostra
    st.markdown("### 🔍 Dados de Amostra")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Informações dos dados
    st.markdown("### 📊 Informações das Colunas")
    col_info = pd.DataFrame({
        'Coluna': df.columns,
        'Tipo de Dado': df.dtypes,
        'Contagem Não-Nula': df.count(),
        'Contagem Nula': df.isnull().sum()
    })
    st.dataframe(col_info, use_container_width=True)

def show_overview_page(df):
    """Página de análise geral"""
    st.markdown("# 📈 Análise Geral")
    
    # Adiciona filtros na barra lateral
    filtered_df = add_sidebar_filters(df)
    
    # Métricas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_people = len(filtered_df)
        st.metric("Total de Pessoas", f"{total_people:,}")
    
    with col2:
        smokers = len(filtered_df[filtered_df['fumante'] == 'Sim'])
        st.metric("Fumantes Atuais", f"{smokers:,}")
    
    with col3:
        smoking_rate = (smokers / total_people * 100) if total_people > 0 else 0
        st.metric("Taxa de Tabagismo", f"{smoking_rate:.1f}%")
    
    with col4:
        avg_age = filtered_df['idade'].mean()
        st.metric("Idade Média", f"{avg_age:.1f} anos")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚬 Distribuição do Status de Fumante")
        smoking_counts = filtered_df['fumante'].value_counts()
        
        fig = px.pie(
            values=smoking_counts.values,
            names=smoking_counts.index,
            title="Status de Fumante",
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 👥 Distribuição por Sexo")
        gender_counts = filtered_df['sexo'].value_counts()
        
        fig = px.bar(
            x=gender_counts.index,
            y=gender_counts.values,
            title="Distribuição por Sexo",
            color=gender_counts.index,
            color_discrete_map={'Masculino': '#4dabf7', 'Feminino': '#ff8cc8'}
        )
        fig.update_layout(showlegend=False, xaxis_title="Sexo", yaxis_title="Contagem")
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise de correlação entre tabagismo e outras variáveis
    st.markdown("### 🔗 Tabagismo por Categoria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Por Sexo")
        gender_smoking = pd.crosstab(filtered_df['sexo'], filtered_df['fumante'])
        
        fig = px.bar(
            gender_smoking,
            title="Taxa de Tabagismo por Sexo",
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Por Faixa Etária")
        # Cria grupos de idade
        filtered_df['faixa_etaria'] = pd.cut(
            filtered_df['idade'], 
            bins=[18, 30, 40, 50, 60, 80], 
            labels=['18-30', '31-40', '41-50', '51-60', '60+']
        )
        
        age_smoking = pd.crosstab(filtered_df['faixa_etaria'], filtered_df['fumante'])
        
        fig = px.bar(
            age_smoking,
            title="Taxa de Tabagismo por Faixa Etária",
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

def show_demographic_page(df):
    """Página de análise demográfica"""
    st.markdown("# 🔍 Análise Demográfica")
    
    filtered_df = add_sidebar_filters(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Escolaridade vs Tabagismo")
        education_smoking = pd.crosstab(filtered_df['escolaridade'], filtered_df['fumante'])
        
        fig = px.bar(
            education_smoking,
            title="Taxa de Tabagismo por Escolaridade",
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🗺️ Distribuição por Estado")
        state_counts = filtered_df['estado'].value_counts().head(10)
        
        fig = px.bar(
            x=state_counts.values,
            y=state_counts.index,
            orientation='h',
            title="Top 10 Estados por Amostra",
            color=state_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise de renda
    st.markdown("### 💰 Análise de Renda")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            filtered_df,
            x='renda_mensal',
            color='fumante',
            title="Distribuição de Renda por Status de Fumante",
            nbins=30,
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            filtered_df,
            x='fumante',
            y='renda_mensal',
            title="Renda por Status de Fumante",
            color='fumante',
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_health_page(df):
    """Página de métricas de saúde"""
    st.markdown("# 🏥 Métricas de Saúde")
    
    filtered_df = add_sidebar_filters(df)
    
    # Métricas de saúde
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_imc = filtered_df['imc'].mean()
        st.metric("IMC Médio", f"{avg_imc:.1f}")
    
    with col2:
        diabetes_rate = (filtered_df['diabetes'] == 'Sim').mean() * 100
        st.metric("Taxa de Diabetes", f"{diabetes_rate:.1f}%")
    
    with col3:
        heart_disease_rate = (filtered_df['doenca_cardiaca'] == 'Sim').mean() * 100
        st.metric("Taxa de Doença Cardíaca", f"{heart_disease_rate:.1f}%")
    
    with col4:
        avg_systolic = filtered_df['pressao_sistolica'].mean()
        st.metric("Pressão Sistólica Média", f"{avg_systolic:.0f} mmHg")
    
    # Gráficos de saúde
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Distribuição de IMC")
        fig = px.histogram(
            filtered_df,
            x='imc',
            color='fumante',
            title="Distribuição de IMC por Status de Fumante",
            nbins=30,
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        fig.update_layout(barmode='overlay')
        fig.update_traces(opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏷️ Categorias de IMC")
        imc_counts = filtered_df['categoria_imc'].value_counts()
        
        fig = px.pie(
            values=imc_counts.values,
            names=imc_counts.index,
            title="Distribuição das Categorias de IMC"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Pressão arterial
    st.markdown("### 🩺 Análise da Pressão Arterial")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_df,
            x='pressao_sistolica',
            y='pressao_diastolica',
            color='fumante',
            title="Pressão Sistólica vs Diastólica",
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            filtered_df,
            x='fumante',
            y='pressao_sistolica',
            title="Pressão Sistólica por Status de Fumante",
            color='fumante',
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_risk_factors_page(df):
    """Página de fatores de risco"""
    st.markdown("# 🎯 Fatores de Risco")
    
    filtered_df = add_sidebar_filters(df)
    
    # Fatores de estilo de vida
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🍺 Consumo de Álcool vs Tabagismo")
        alcohol_smoking = pd.crosstab(filtered_df['consome_alcool'], filtered_df['fumante'])
        
        fig = px.bar(
            alcohol_smoking,
            title="Tabagismo por Consumo de Álcool",
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏃‍♂️ Exercício vs Tabagismo")
        exercise_smoking = pd.crosstab(filtered_df['pratica_exercicios'], filtered_df['fumante'])
        
        fig = px.bar(
            exercise_smoking,
            title="Tabagismo por Prática de Exercícios",
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        fig.update_layout(barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cigarros por dia para fumantes
    smokers_df = filtered_df[filtered_df['fumante'] == 'Sim']
    if len(smokers_df) > 0:
        st.markdown("### 🚬 Análise de Cigarros por Dia (Fumantes)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                smokers_df,
                x='cigarros_por_dia',
                title="Distribuição de Cigarros por Dia",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Categoriza consumo
            smokers_df['categoria_consumo'] = pd.cut(
                smokers_df['cigarros_por_dia'],
                bins=[0, 5, 10, 20, float('inf')],
                labels=['Leve (1-5)', 'Moderado (6-10)', 'Pesado (11-20)', 'Muito Pesado (20+)']
            )
            
            consumption_counts = smokers_df['categoria_consumo'].value_counts()
            
            fig = px.pie(
                values=consumption_counts.values,
                names=consumption_counts.index,
                title="Categorias de Consumo de Cigarros"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_interactive_page(df):
    """Página de exploração interativa"""
    st.markdown("# 📊 Explorador Interativo")
    
    filtered_df = add_sidebar_filters(df)
    
    st.markdown("### 🔍 Análise Personalizada")
    
    # Seleções para gráficos personalizados
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox(
            "Selecione a variável do eixo X:",
            options=['idade', 'imc', 'renda_mensal', 'pressao_sistolica', 'pressao_diastolica', 'cigarros_por_dia']
        )
    
    with col2:
        y_axis = st.selectbox(
            "Selecione a variável do eixo Y:",
            options=['imc', 'renda_mensal', 'pressao_sistolica', 'pressao_diastolica', 'cigarros_por_dia', 'idade']
        )
    
    # Gráfico scatter personalizado
    if x_axis != y_axis:
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color='fumante',
            size='idade',
            hover_data=['sexo', 'escolaridade'],
            title=f"{y_axis.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}",
            color_discrete_map={'Sim': '#ff6b6b', 'Não': '#51cf66'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de correlação
    st.markdown("### 🔗 Matriz de Correlação")
    
    numeric_cols = ['idade', 'renda_mensal', 'imc', 'pressao_sistolica', 'pressao_diastolica', 'cigarros_por_dia']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Matriz de Correlação - Variáveis Numéricas",
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas descritivas
    st.markdown("### 📈 Estatísticas Descritivas")
    st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)

def main():
    """Função principal da aplicação"""
    
    # Cabeçalho
    st.markdown('<h1 class="main-header">🚭 Dashboard de Tabagismo e Fatores de Risco</h1>', unsafe_allow_html=True)
    
    # Carrega dados
    df = load_data()
    
    if df is None:
        st.error("Não foi possível carregar os dados. Verifique a configuração.")
        st.stop()
    
    # Barra lateral
    st.sidebar.title("📊 Navegação e Filtros")
    st.sidebar.markdown("---")
    
    # Navegação
    page = st.sidebar.selectbox(
        "Escolha uma página:",
        ["🏠 Início", "📈 Análise Geral", "🔍 Análise Demográfica", 
         "🏥 Métricas de Saúde", "🎯 Fatores de Risco", "📊 Explorador Interativo"]
    )
    
    # Exibe a página selecionada
    if page == "🏠 Início":
        show_home_page(df)
    elif page == "📈 Análise Geral":
        show_overview_page(df)
    elif page == "🔍 Análise Demográfica":
        show_demographic_page(df)
    elif page == "🏥 Métricas de Saúde":
        show_health_page(df)
    elif page == "🎯 Fatores de Risco":
        show_risk_factors_page(df)
    elif page == "📊 Explorador Interativo":
        show_interactive_page(df)

if __name__ == "__main__":
    main()
