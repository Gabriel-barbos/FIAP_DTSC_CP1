# Checkpoint 4 - Data Science & Statistical Computing
## FIAP - Regressão Linear e App Streamlit

Repositório com o trabalho do Checkpoint 4 da disciplina Data Science & Statistical Computing na FIAP.
O projeto analisa e modela o consumo de energia elétrica de órgãos públicos federais com base no histórico de 2018-2019 e nas diretrizes do Decreto nº 10.779/2021.

## Estrutura do Projeto

projeto/
|-- app.py                          # Aplicação interativa em Streamlit
|-- notebook.ipynb                  # Jupyter Notebook completo e executado
|-- requirements.txt                # Bibliotecas necessárias
|-- README.md                       # Documentação do projeto
|-- dados/
|   └── base.csv                    # Base de dados de consumo
└── modelo/
    |-- modelo.pkl                  # Pipeline treinado salvo com joblib
    └── metadados.pkl               # Estatísticas de treino para apoio ao app

## Como Configurar o Ambiente

Recomendamos utilizar o Python 3.10 ou superior.

1. Criar e ativar o ambiente virtual:

# No Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1

# No Linux/macOS:
python3 -m venv .venv
source .venv/bin/activate

2. Instalar as dependências:
pip install -r requirements.txt

## Como Executar

### 1. Notebook Jupyter
Para visualizar o passo a passo da limpeza, análises exploratórias, treino dos 4 modelos e diagnósticos de resíduos:

jupyter notebook notebook.ipynb

### 2. App Streamlit
Para rodar a interface interativa com o simulador de previsão:

streamlit run app.py

O app abrirá no navegador no endereço http://localhost:8501.

## Resumo dos Resultados

1. Limpeza dos Dados:
- A coluna `mes_ano` foi tratada para gerar a data completa e separar mês e ano.
- Os outliers de consumo muito alto (grandes universidades como UFMG, UFRJ) foram mantidos por representarem estruturas reais de grande porte que o modelo precisa saber estimar.

2. Modelagem e Comparação (Conjunto de Teste - 30%):
- Baseline (Média): R² = -0.0183 | MAE = 791.861 kWh | RMSE = 1.505.844 kWh
- Regressão Linear Simples: R² = 0.8288 | MAE = 226.163 kWh | RMSE = 617.362 kWh
- Regressão Linear Múltipla: R² = 0.8283 | MAE = 226.730 kWh | RMSE = 618.294 kWh
- Regressão Polinomial (Grau 2): R² = 0.8764 | MAE = 292.147 kWh | RMSE = 524.550 kWh

O modelo escolhido para produção foi a Regressão Linear Múltipla, que equilibra baixo erro absoluto (MAE), estabilidade nas previsões e facilidade de interpretação dos coeficientes.

3. Diagnósticos:
- Resíduos centrados em zero e com média praticamente nula.
- Q-Q Plot com boa aderência à normalidade no miolo da distribuição.
- Fator de Inflação da Variância (VIF) de 1,0007, comprovando ausência de multicolinearidade.

4. Interpretação dos Coeficientes:
- O coeficiente do consumo histórico foi de aproximadamente 0,7013, indicando que cada 1 kWh a mais no histórico de 2018-2019 reflete em média 0,7013 kWh no consumo atual (redução geral de ~30% em relação ao período pré-pandemia).
- Lembramos que correlação e associação estatística não significam causalidade física direta.

## Integrantes do Grupo
Trabalho desenvolvido para o Checkpoint 4 da FIAP.
