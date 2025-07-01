import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import kagglehub
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
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega e armazena em cache o dataset de tabagismo"""
    try:
        # Verifica se os dados já existem localmente
        data_path = Path("data/smoking_data.csv")
        
        if not data_path.exists():
            # Baixa o dataset usando kagglehub
            st.info("Baixando dataset... Isso pode levar alguns momentos.")
            path = kagglehub.dataset_download("khushikyad001/smoking-and-other-risk-factors-dataset")
            
            # Encontra o arquivo CSV no caminho baixado
            csv_files = list(Path(path).glob("*.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                # Salva localmente para uso futuro
                data_path.parent.mkdir(exist_ok=True)
                df.to_csv(data_path, index=False)
            else:
                st.error("Nenhum arquivo CSV encontrado no dataset baixado.")
                return None
        else:
            df = pd.read_csv(data_path)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

def main():
    """Função principal da aplicação"""
    
    # Cabeçalho
    st.markdown('<h1 class="main-header">🚭 Dashboard de Tabagismo e Fatores de Risco</h1>', unsafe_allow_html=True)
    
    # Carrega dados
    df = load_data()
    
    if df is None:
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
        smoking_rate = (df['Smoking_Status'].value_counts().get('Current', 0) / len(df) * 100)
        st.metric("Taxa de Tabagismo", f"{smoking_rate:.1f}%")
    
    with col4:
        st.metric("Faixa Etária", f"{df['Age'].min()}-{df['Age'].max()}")
    
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
    add_sidebar_filters(df)
    
    # Aplica filtros
    filtered_df = apply_filters(df)
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Participantes", len(filtered_df))
    
    with col2:
        smokers = len(filtered_df[filtered_df['Smoking_Status'] == 'Current'])
        st.metric("Fumantes Atuais", smokers)
    
    with col3:
        avg_age = filtered_df['Age'].mean()
        st.metric("Idade Média", f"{avg_age:.1f}")
    
    with col4:
        avg_bmi = filtered_df['BMI'].mean() if 'BMI' in filtered_df.columns else 0
        st.metric("IMC Médio", f"{avg_bmi:.1f}")
    
    # Visualizações
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição do status de tabagismo
        fig1 = px.pie(
            filtered_df, 
            names='Smoking_Status',
            title="Distribuição do Status de Tabagismo",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Distribuição de idade
        fig2 = px.histogram(
            filtered_df,
            x='Age',
            nbins=20,
            title="Distribuição de Idade",
            color_discrete_sequence=['#1f77b4']
        )
        fig2.update_layout(xaxis_title="Idade", yaxis_title="Contagem")
        st.plotly_chart(fig2, use_container_width=True)

def show_demographic_page(df):
    """Página de análise demográfica"""
    st.markdown("# 🔍 Análise Demográfica")
    
    # Adiciona filtros
    add_sidebar_filters(df)
    filtered_df = apply_filters(df)
    
    # Análise por gênero
    col1, col2 = st.columns(2)
    
    with col1:
        # Gênero vs Tabagismo
        gender_smoking = pd.crosstab(filtered_df['Gender'], filtered_df['Smoking_Status'])
        fig1 = px.bar(
            gender_smoking.reset_index(),
            x='Gender',
            y=gender_smoking.columns.tolist(),
            title="Status de Tabagismo por Gênero",
            barmode='group'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Faixas etárias por status de tabagismo
        filtered_df['age_group'] = pd.cut(filtered_df['Age'], 
                                        bins=[0, 30, 45, 60, 100], 
                                        labels=['18-30', '31-45', '46-60', '60+'])
        age_smoking = pd.crosstab(filtered_df['age_group'], filtered_df['Smoking_Status'])
        fig2 = px.bar(
            age_smoking.reset_index(),
            x='age_group',
            y=age_smoking.columns.tolist(),
            title="Status de Tabagismo por Faixa Etária",
            barmode='stack'
        )
        st.plotly_chart(fig2, use_container_width=True)

def show_health_page(df):
    """Página de análise de métricas de saúde"""
    st.markdown("# 🏥 Análise de Métricas de Saúde")
    
    add_sidebar_filters(df)
    filtered_df = apply_filters(df)
    
    # Análises de saúde mais específicas baseadas no dataset real
    col1, col2 = st.columns(2)
    
    with col1:
        if 'BMI' in filtered_df.columns:
            fig1 = px.box(
                filtered_df,
                x='Smoking_Status',
                y='BMI',
                title="Distribuição de IMC por Status de Tabagismo"
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Condições de saúde (se disponível)
        if 'Chronic_Lung_Disease' in filtered_df.columns:
            health_smoking = pd.crosstab(filtered_df['Chronic_Lung_Disease'], filtered_df['Smoking_Status'])
            fig2 = px.bar(
                health_smoking.reset_index(),
                x='Chronic_Lung_Disease',
                y=health_smoking.columns.tolist(),
                title="Doença Pulmonar Crônica por Status de Tabagismo"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Análise adicional - Cigarros por dia vs Condições de saúde
    col3, col4 = st.columns(2)
    
    with col3:
        if 'Cigarettes_Per_Day' in filtered_df.columns and 'Smoking_Status' in filtered_df.columns:
            # Filtra apenas fumantes atuais e ex-fumantes para esta análise
            smokers_df = filtered_df[filtered_df['Smoking_Status'].isin(['Current', 'Former'])]
            if not smokers_df.empty:
                fig3 = px.histogram(
                    smokers_df,
                    x='Cigarettes_Per_Day',
                    color='Smoking_Status',
                    title="Distribuição de Cigarros por Dia",
                    nbins=20
                )
                st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        if 'Years_Smoking' in filtered_df.columns:
            smokers_df = filtered_df[filtered_df['Smoking_Status'].isin(['Current', 'Former'])]
            if not smokers_df.empty:
                fig4 = px.box(
                    smokers_df,
                    x='Smoking_Status',
                    y='Years_Smoking',
                    title="Anos de Tabagismo por Status"
                )
                st.plotly_chart(fig4, use_container_width=True)
    
    # Matriz de correlação para métricas de saúde
    st.markdown("### 📊 Correlações entre Fatores de Saúde")
    health_cols = ['Age', 'BMI', 'Years_Smoking', 'Cigarettes_Per_Day', 'Physical_Activity_Level']
    available_health_cols = [col for col in health_cols if col in filtered_df.columns]
    
    if len(available_health_cols) > 1:
        health_data = filtered_df[available_health_cols].select_dtypes(include=[np.number])
        if not health_data.empty:
            correlation_matrix = health_data.corr()
            fig5 = px.imshow(
                correlation_matrix,
                title="Matriz de Correlação - Fatores de Saúde",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig5, use_container_width=True)

def show_risk_factors_page(df):
    """Página de análise de fatores de risco"""
    st.markdown("# 🎯 Análise de Fatores de Risco")
    
    add_sidebar_filters(df)
    filtered_df = apply_filters(df)
    
    st.info("Esta página analisa vários fatores de risco associados aos padrões de tabagismo.")
    
    # Análise de fatores ambientais
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Air_Pollution_Level' in filtered_df.columns:
            pollution_smoking = pd.crosstab(filtered_df['Air_Pollution_Level'], filtered_df['Smoking_Status'])
            fig1 = px.bar(
                pollution_smoking.reset_index(),
                x='Air_Pollution_Level',
                y=pollution_smoking.columns.tolist(),
                title="Nível de Poluição do Ar vs Status de Tabagismo",
                barmode='group'
            )
            st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if 'Secondhand_Smoke_Exposure' in filtered_df.columns:
            secondhand_smoking = pd.crosstab(filtered_df['Secondhand_Smoke_Exposure'], filtered_df['Smoking_Status'])
            fig2 = px.bar(
                secondhand_smoking.reset_index(),
                x='Secondhand_Smoke_Exposure',
                y=secondhand_smoking.columns.tolist(),
                title="Exposição ao Fumo Passivo vs Status de Tabagismo",
                barmode='group'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    # Análise socioeconômica
    col3, col4 = st.columns(2)
    
    with col3:
        if 'Education_Level' in filtered_df.columns:
            education_smoking = pd.crosstab(filtered_df['Education_Level'], filtered_df['Smoking_Status'])
            fig3 = px.bar(
                education_smoking.reset_index(),
                x='Education_Level',
                y=education_smoking.columns.tolist(),
                title="Nível Educacional vs Status de Tabagismo",
                barmode='stack'
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        if 'Income_Level' in filtered_df.columns:
            income_smoking = pd.crosstab(filtered_df['Income_Level'], filtered_df['Smoking_Status'])
            fig4 = px.bar(
                income_smoking.reset_index(),
                x='Income_Level',
                y=income_smoking.columns.tolist(),
                title="Nível de Renda vs Status de Tabagismo",
                barmode='stack'
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    # Análise de correlação
    st.markdown("### 📊 Matriz de Correlação de Fatores de Risco")
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        correlation_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Matriz de Correlação de Variáveis Numéricas",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise de histórico familiar
    if 'Family_History' in filtered_df.columns:
        st.markdown("### 👨‍👩‍👧‍👦 Análise de Histórico Familiar")
        family_smoking = pd.crosstab(filtered_df['Family_History'], filtered_df['Smoking_Status'])
        fig_family = px.bar(
            family_smoking.reset_index(),
            x='Family_History',
            y=family_smoking.columns.tolist(),
            title="Histórico Familiar vs Status de Tabagismo",
            barmode='group'
        )
        st.plotly_chart(fig_family, use_container_width=True)

def show_interactive_page(df):
    """Página do explorador interativo"""
    st.markdown("# 📊 Explorador Interativo de Dados")
    
    add_sidebar_filters(df)
    filtered_df = apply_filters(df)
    
    # Gráfico de dispersão interativo
    st.markdown("## 🔍 Gráfico de Dispersão Personalizado")
    
    col1, col2, col3 = st.columns(3)
    
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
    
    with col1:
        x_axis = st.selectbox("Eixo X", numeric_cols, index=0 if 'Age' in numeric_cols else 0)
    
    with col2:
        y_axis = st.selectbox("Eixo Y", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
    
    with col3:
        color_by = st.selectbox("Colorir por", categorical_cols, 
                               index=categorical_cols.index('Smoking_Status') if 'Smoking_Status' in categorical_cols else 0)
    
    if x_axis and y_axis:
        fig = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            title=f"{y_axis} vs {x_axis} (colorido por {color_by})",
            hover_data=['Age', 'Gender'] if 'Age' in filtered_df.columns and 'Gender' in filtered_df.columns else None
        )
        st.plotly_chart(fig, use_container_width=True)

def add_sidebar_filters(df):
    """Adiciona filtros à barra lateral"""
    st.sidebar.markdown("### 🔧 Filtros")
    
    # Filtro de idade
    if 'Age' in df.columns:
        age_range = st.sidebar.slider(
            "Faixa Etária",
            min_value=int(df['Age'].min()),
            max_value=int(df['Age'].max()),
            value=(int(df['Age'].min()), int(df['Age'].max()))
        )
        st.session_state['age_filter'] = age_range
    
    # Filtro de gênero
    if 'Gender' in df.columns:
        genders = ['Todos'] + df['Gender'].unique().tolist()
        selected_gender = st.sidebar.selectbox("Gênero", genders)
        st.session_state['gender_filter'] = selected_gender
    
    # Filtro de status de tabagismo
    if 'Smoking_Status' in df.columns:
        smoking_statuses = ['Todos'] + df['Smoking_Status'].unique().tolist()
        selected_smoking = st.sidebar.multiselect("Status de Tabagismo", smoking_statuses, default=['Todos'])
        st.session_state['smoking_filter'] = selected_smoking
    
    # Filtros adicionais baseados no dataset real
    if 'Education_Level' in df.columns:
        education_levels = ['Todos'] + df['Education_Level'].unique().tolist()
        selected_education = st.sidebar.selectbox("Nível Educacional", education_levels)
        st.session_state['education_filter'] = selected_education
    
    if 'Region' in df.columns:
        regions = ['Todos'] + df['Region'].unique().tolist()
        selected_region = st.sidebar.selectbox("Região", regions)
        st.session_state['region_filter'] = selected_region

def apply_filters(df):
    """Aplica os filtros selecionados ao dataframe"""
    filtered_df = df.copy()
    
    # Aplica filtro de idade
    if 'age_filter' in st.session_state and 'Age' in df.columns:
        age_min, age_max = st.session_state['age_filter']
        filtered_df = filtered_df[(filtered_df['Age'] >= age_min) & (filtered_df['Age'] <= age_max)]
    
    # Aplica filtro de gênero
    if 'gender_filter' in st.session_state and 'Gender' in df.columns:
        if st.session_state['gender_filter'] != 'Todos':
            filtered_df = filtered_df[filtered_df['Gender'] == st.session_state['gender_filter']]
    
    # Aplica filtro de status de tabagismo
    if 'smoking_filter' in st.session_state and 'Smoking_Status' in df.columns:
        if 'Todos' not in st.session_state['smoking_filter'] and st.session_state['smoking_filter']:
            filtered_df = filtered_df[filtered_df['Smoking_Status'].isin(st.session_state['smoking_filter'])]
    
    # Aplica filtros adicionais
    if 'education_filter' in st.session_state and 'Education_Level' in df.columns:
        if st.session_state['education_filter'] != 'Todos':
            filtered_df = filtered_df[filtered_df['Education_Level'] == st.session_state['education_filter']]
    
    if 'region_filter' in st.session_state and 'Region' in df.columns:
        if st.session_state['region_filter'] != 'Todos':
            filtered_df = filtered_df[filtered_df['Region'] == st.session_state['region_filter']]
    
    return filtered_df

if __name__ == "__main__":
    main()
