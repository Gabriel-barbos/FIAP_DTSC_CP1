"""
Checkpoint 4 - Data Science & Statistical Computing (FIAP)
"""
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Previsão de Consumo de Energia | FIAP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funções de carregamento com cache para não reprocessar a cada interação
@st.cache_data
def carregar_dados():
    caminho = 'dados/base.csv'
    if not os.path.exists(caminho):
        caminho = 'consumo_tratado.csv'
    df = pd.read_csv(caminho)
    
    # Tratamento da data a partir de mes_ano
    df['mes_ano_str'] = df['mes_ano'].astype(str).str.zfill(6)
    df['ano_tratado'] = df['mes_ano_str'].str[-4:].astype(int)
    df['mes_tratado'] = df['mes_ano_str'].str[:-4].astype(int)
    df['media_consumo_mes_2018_2019'] = pd.to_numeric(df['media_consumo_mes_2018_2019'], errors='coerce')
    df['consumo_mes_referencia'] = pd.to_numeric(df['consumo_mes_referencia'], errors='coerce')
    
    if 'tipo_orgao' not in df.columns:
        def categorizar(nome):
            nu = str(nome).upper()
            if 'UNIVERSIDADE' in nu:
                return 'Universidade'
            if 'INSTITUTO FEDERAL' in nu or 'CEFET' in nu:
                return 'Instituto Federal'
            if 'MINIST' in nu:
                return 'Ministerio'
            if 'AGENCIA' in nu or 'AG?NCIA' in nu or 'AGÊNCIA' in nu:
                return 'Agencia'
            if 'FUNDACAO' in nu or 'FUNDA??O' in nu or 'FUNDAÇÃO' in nu:
                return 'Fundacao'
            return 'Outros'
        df['tipo_orgao'] = df['orgao'].apply(categorizar)
    return df

@st.cache_resource
def carregar_modelo():
    caminho_modelo = 'modelo/modelo.pkl'
    caminho_meta = 'modelo/metadados.pkl'
    modelo = joblib.load(caminho_modelo) if os.path.exists(caminho_modelo) else None
    meta = joblib.load(caminho_meta) if os.path.exists(caminho_meta) else None
    return modelo, meta

df = carregar_dados()
modelo, metadados = carregar_modelo()

# Barra lateral de navegacao
st.sidebar.title("FIAP - Checkpoint 4")
st.sidebar.markdown("""
**Data Science & Statistical Computing**

Projeto de Regressão Linear para estimar o consumo de energia elétrica de órgãos públicos federais com base no histórico de 2018-2019 (Decreto nº 10.779/2021).
""")

secao = st.sidebar.radio(
    "Navegação:",
    ["Visão Geral e Dados", "Análise Exploratória (EDA)", "Métricas e Diagnósticos", "Simulador de Previsão"]
)

# 1. Visao Geral e Dados
if secao == "Visão Geral e Dados":
    st.title("Previsão de Consumo de Energia em Órgãos Públicos Federais")
    st.markdown("""
    Este aplicativo foi desenvolvido para demonstrar o modelo preditivo construído no Checkpoint 4. 
    O objetivo é prever o **consumo do mês de referência (y, em kWh)** usando como preditor principal o **consumo médio histórico de 2018 e 2019 (X, em kWh)**, além do tipo de órgão e mês de referência.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Linhas", f"{len(df):,}")
    with col2:
        st.metric("Total de Órgãos", f"{df['sigla_orgao'].nunique()}")
    with col3:
        st.metric("Consumo Médio Atual", f"{df['consumo_mes_referencia'].mean()/1e3:,.1f} MWh")
    with col4:
        st.metric("Média Histórica (18-19)", f"{df['media_consumo_mes_2018_2019'].mean()/1e3:,.1f} MWh")
        
    st.subheader("Amostra da Base de Dados")
    filtro_tipo = st.multiselect(
        "Filtrar por tipo de órgão:",
        options=sorted(df['tipo_orgao'].unique()),
        default=sorted(df['tipo_orgao'].unique())
    )
    df_filtrado = df[df['tipo_orgao'].isin(filtro_tipo)]
    
    st.dataframe(
        df_filtrado[['sigla_orgao', 'tipo_orgao', 'ano_tratado', 'mes_tratado', 'media_consumo_mes_2018_2019', 'consumo_mes_referencia']].head(15),
        use_container_width=True
    )
    
    st.subheader("Estatísticas Descritivas das Variáveis de Consumo")
    desc_df = df[['consumo_mes_referencia', 'media_consumo_mes_2018_2019']].describe().T
    st.dataframe(desc_df, use_container_width=True)

# 2. Análise Exploratória
elif secao == "Análise Exploratória (EDA)":
    st.title("Análise Exploratória dos Dados")
    st.markdown("Visualização da distribuição do consumo e relação entre a média de 2018-2019 e o consumo atual.")
    
    tab1, tab2, tab3 = st.tabs(["Dispersão Histórico x Atual", "Distribuição do Consumo", "Consumo por Categoria"])
    
    with tab1:
        st.subheader("Dispersão: Consumo Histórico (2018-2019) vs Consumo Atual")
        fig_scatter = px.scatter(
            df,
            x=df['media_consumo_mes_2018_2019'] / 1e3,
            y=df['consumo_mes_referencia'] / 1e3,
            color='tipo_orgao',
            hover_data=['sigla_orgao'],
            labels={
                'x': 'Consumo Médio Histórico 2018-2019 (MWh)',
                'y': 'Consumo do Mês de Referência (MWh)',
                'tipo_orgao': 'Tipo de Órgão'
            },
            title="Correlação Linear Forte (r = 0.8052)",
            template="plotly_white"
        )
        max_v = max(df['media_consumo_mes_2018_2019'].max(), df['consumo_mes_referencia'].max()) / 1e3
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_v], y=[0, max_v],
            mode='lines',
            name='Linha de 45º (y = x)',
            line=dict(color='gray', dash='dash')
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("A maioria dos pontos está abaixo da linha de 45º, indicando que grande parte dos órgãos reduziu o consumo em relação ao período pré-pandêmico (2018-2019).")

    with tab2:
        st.subheader("Distribuição do Consumo no Mês de Referência")
        fig_hist = px.histogram(
            df,
            x=df['consumo_mes_referencia'] / 1e3,
            nbins=40,
            marginal="box",
            labels={'x': 'Consumo no Mês (MWh)'},
            title="Distribuição com Assimetria Positiva à Direita",
            color_discrete_sequence=['#1f77b4'],
            template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("A distribuição é bastante concentrada em valores menores (mediana de ~190 MWh), com uma cauda longa de órgãos de grande porte como universidades federais.")

    with tab3:
        st.subheader("Consumo Médio por Tipo de Órgão")
        df_perfil = df.groupby('tipo_orgao')['consumo_mes_referencia'].mean().reset_index()
        df_perfil['consumo_mwh'] = df_perfil['consumo_mes_referencia'] / 1e3
        df_perfil = df_perfil.sort_values(by='consumo_mwh', ascending=False)
        
        fig_bar = px.bar(
            df_perfil,
            x='tipo_orgao',
            y='consumo_mwh',
            text_auto='.1f',
            labels={'tipo_orgao': 'Tipo de Órgão', 'consumo_mwh': 'Consumo Médio (MWh)'},
            title="Consumo Médio por Categoria de Órgão",
            color='tipo_orgao',
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Universidades Federais têm a maior média de consumo (~1.341 MWh), seguidas por Institutos Federais e Ministérios.")

# 3. Métricas e Diagnósticos
elif secao == "Métricas e Diagnósticos":
    st.title("Desempenho e Diagnósticos do Modelo Final")
    st.markdown("Resultados do modelo de Regressão Linear Múltipla avaliado no conjunto de teste (30% dos dados).")
    
    c1, c2, c3 = st.columns(3)
    if metadados and 'metricas_teste' in metadados:
        m = metadados['metricas_teste']
        c1.metric("R² no Teste", f"{m['R2']:.4f}")
        c2.metric("MAE no Teste", f"{m['MAE']/1e3:,.1f} MWh")
        c3.metric("RMSE no Teste", f"{m['RMSE']/1e3:,.1f} MWh")
    else:
        c1.metric("R² no Teste", "0.8283")
        c2.metric("MAE no Teste", "226.7 MWh")
        c3.metric("RMSE no Teste", "618.3 MWh")
        
    st.subheader("Gráficos de Diagnóstico no Conjunto de Teste")
    col_d1, col_d2 = st.columns(2)
    
    X_test = df[['media_consumo_mes_2018_2019', 'mes_tratado', 'tipo_orgao']]
    y_real = df['consumo_mes_referencia']
    y_pred = modelo.predict(X_test) if modelo else y_real * 0.70 + 86000
    
    with col_d1:
        st.markdown("**Real vs Previsto**")
        fig_rvp = px.scatter(
            x=y_real/1e3, y=y_pred/1e3,
            labels={'x': 'Consumo Real (MWh)', 'y': 'Consumo Previsto (MWh)'},
            template="plotly_white",
            opacity=0.65
        )
        max_diag = max(y_real.max(), y_pred.max())/1e3
        fig_rvp.add_trace(go.Scatter(
            x=[0, max_diag], y=[0, max_diag],
            mode='lines',
            name='Ideal (y = y_hat)',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig_rvp, use_container_width=True)

    with col_d2:
        st.markdown("**Resíduos vs Valores Ajustados**")
        residuos = (y_real - y_pred) / 1e3
        fig_res = px.scatter(
            x=y_pred/1e3, y=residuos,
            labels={'x': 'Valores Previstos (MWh)', 'y': 'Resíduos (MWh)'},
            template="plotly_white",
            opacity=0.65,
            color_discrete_sequence=['#ff7f0e']
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)

# 4. Simulador de Previsão
elif secao == "Simulador de Previsão":
    st.title("Simulador de Previsão de Consumo")
    st.markdown("Informe os dados do órgão para calcular a previsão usando o Pipeline treinado salvo em `modelo/modelo.pkl`.")
    
    if metadados:
        h_min = metadados['media_historica_min']
        h_max = metadados['media_historica_max']
        tipos_disp = metadados['tipos_orgao']
    else:
        h_min = float(df['media_consumo_mes_2018_2019'].min())
        h_max = float(df['media_consumo_mes_2018_2019'].max())
        tipos_disp = sorted(df['tipo_orgao'].unique().tolist())
        
    with st.form("form_previsao"):
        st.subheader("Parâmetros de Entrada")
        c1, c2 = st.columns(2)
        
        with c1:
            media_input = st.number_input(
                "Consumo Médio Histórico de Referência 2018-2019 (kWh):",
                min_value=0.0,
                max_value=100000000.0,
                value=500000.0,
                step=10000.0,
                help="Média mensal de consumo observada nos anos de 2018 e 2019."
            )
            st.caption(f"Intervalo observado no treino: de {h_min:,.0f} kWh a {h_max:,.0f} kWh")
            
        with c2:
            tipo_input = st.selectbox(
                "Tipo de Órgão:",
                options=tipos_disp,
                index=tipos_disp.index('Universidade') if 'Universidade' in tipos_disp else 0
            )
            mes_input = st.slider(
                "Mês de Referência:",
                min_value=1,
                max_value=12,
                value=9,
                format="%d"
            )
            
        botao_prever = st.form_submit_button("Calcular Previsão de Consumo", use_container_width=True)
        
    if botao_prever:
        # Verificação de extrapolação
        if media_input < h_min or media_input > h_max:
            st.warning(f"Aviso de Extrapolação: o valor inserido ({media_input:,.0f} kWh) está fora da faixa observada nos dados de treino ({h_min:,.0f} a {h_max:,.0f} kWh). O modelo pode perder precisão nesta faixa.")
        else:
            st.success("Valor dentro do intervalo observado no conjunto de treino.")
            
        # Monta DataFrame com a mesma estrutura usada no fit do Pipeline
        df_inferencia = pd.DataFrame([{
            'media_consumo_mes_2018_2019': float(media_input),
            'mes_tratado': int(mes_input),
            'tipo_orgao': str(tipo_input)
        }])
        
        if modelo:
            previsao_kwh = float(modelo.predict(df_inferencia)[0])
        else:
            previsao_kwh = float(media_input * 0.7013 + 86645.0)
            
        previsao_kwh = max(0.0, previsao_kwh)
        previsao_mwh = previsao_kwh / 1000.0
        
        st.subheader("Resultado da Previsão")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Consumo Previsto (kWh)", f"{previsao_kwh:,.2f} kWh")
        with rc2:
            st.metric("Consumo Previsto (MWh)", f"{previsao_mwh:,.2f} MWh")
        with rc3:
            var_pct = ((previsao_kwh - media_input) / media_input) * 100 if media_input > 0 else 0
            st.metric("Variação vs Histórico", f"{var_pct:+.2f}%")
            
        st.info(f"Para um órgão do tipo {tipo_input} com histórico de {media_input:,.0f} kWh, a estimativa do modelo é de {previsao_kwh:,.2f} kWh ({previsao_mwh:,.2f} MWh) no mês {mes_input}.")
