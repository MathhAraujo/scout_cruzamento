# ⚽ Scout — Hub de Inteligência e Cruzamento de Dados Esportivos

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.9.1-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-v0.52.0-red.svg)](https://shap.readthedocs.io/)
[![Arquitetura](https://img.shields.io/badge/Arquitetura-Multi--Dataset%20Modular-brightgreen.svg)](#-estrutura-do-repositório)
[![Status](https://img.shields.io/badge/Status-Em%20Expansão-yellow.svg)](#-roadmap-de-novos-cruzamentos)

O **Scout** é um repositório centralizado e modular para ciência de dados esportivos (*Sports Analytics & Machine Learning*). O projeto reúne investigações empíricas, cruzamentos estatísticos multivariados e modelagem preditiva e probabilística aplicada a fatores de rendimento, controle de carga de treinamento, assimetrias biomecânicas e risco de lesões em atletas.

Para manter a escalabilidade e clareza científica, **cada conjunto de dados analisado reside em seu próprio diretório dedicado**, contendo seus dados, visualizações, notebooks interativos e documentação técnica aprofundada.

---

## 📋 Sumário Geral

1. [Visão Geral e Arquitetura](#-visão-geral-e-arquitetura)
2. [Catálogo de Cruzamentos e Datasets](#-catálogo-de-cruzamentos-e-datasets)
   - [SIRP-600 (Sports Injury Risk Prediction)](#1-sirp-600--sports-injury-risk-prediction)
3. [Roadmap de Novos Cruzamentos](#-roadmap-de-novos-cruzamentos)
4. [Estrutura do Repositório](#-estrutura-do-repositório)
5. [Como Adicionar um Novo Cruzamento](#-como-adicionar-um-novo-cruzamento)
6. [Instalação e Execução Local](#-instalação-e-execução-local)
7. [Ressalvas Científicas e Metodológicas](#-ressalvas-científicas-e-metodológicas)

---

## 🏛️ Visão Geral e Arquitetura

O repositório é concebido sob o princípio de **módulos analíticos autocontidos**:

* **Independência Operacional:** Cada pasta correspondente a um dataset possui seu próprio pipeline de análise, dados, notebooks e figuras, permitindo executar e evoluir um estudo sem interferir nos demais.
* **Documentação Dedicada:** Cada cruzamento conta com um `README.md` individual detalhando objetivos, análise exploratória (EDA), formulações matemáticas, interpretação visual das figuras e tabelas de métricas.
* **Padronização Visual e Técnica:** Todas as análises seguem padrões consolidados de boas práticas: reprodutibilidade com sementes fixas, calibração de probabilidades contínuas, explicabilidade com valores SHAP e exportação de gráficos em alta resolução (300 DPI).

---

## 🗂️ Catálogo de Cruzamentos e Datasets

Abaixo estão listados os estudos analíticos e cruzamentos de dados atualmente disponíveis no repositório:

| Diretório / Dataset | Tema Central | Amostra | Algoritmos & Técnicas | Status | Link para Estudo |
| :--- | :--- | :---: | :--- | :---: | :---: |
| [`sirp-600/`](sirp-600/) | **Predição de Risco de Lesão Esportiva** | 600 atletas | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP | Concluído | [Acessar Estudo](sirp-600/README.md) |
| [`soccermon/`](soccermon/) | **Carga Externa (GPS), Sono e Risco de Lesão (7d)** | 50 atletas (36.550 atleta-dias) | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP, ACWR | Concluído (Diagnóstico Crítico) | [Acessar Estudo](soccermon/README.md) |
| [`multimodal_injury/`](multimodal_injury/) | **Carga (ACWR, RPE), Sono e Recuperação Autonômica (HRV)** | 156 atletas (15.420 sessões) | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP, Gabbett ACWR | Concluído | [Acessar Estudo](multimodal_injury/README.md) |

---

### 1. `sirp-600` — Sports Injury Risk Prediction

* **Pasta do Estudo:** [`📁 sirp-600/`](sirp-600/)
* **Documentação Científica:** [`📄 sirp-600/README.md`](sirp-600/README.md)
* **Notebook Principal:** [`📓 sirp-600/notebooks/analise_risco_lesao_sirp600.ipynb`](sirp-600/notebooks/analise_risco_lesao_sirp600.ipynb)

#### 🎯 Resumo da Investigação
Investigação empírica das interações entre variáveis comportamentais de rotina (sono, aquecimento, intensidade de treino), dimensões antropométricas, fisiológicas (tempo de recuperação cardíaca) e mecânicas funcionais (assimetria muscular contralateral) na ocorrência de lesões musculares e articulares.

#### 📊 Principais Conclusões e Métricas
* **Fatores Críticos:** A **Assimetria Muscular Contralateral ($r = +0.374$, 17.9% Gini)** e o **Histórico de Lesões ($r = +0.307$)** revelaram-se os maiores impulsionadores de risco. Por outro lado, **Tempo de Aquecimento ($r = -0.273$)** e **Horas de Sono ($r = -0.270$)** atuam como fortes moduladores protetores.
* **Performance Probabilística:** O modelo calibrado atingiu **$\text{ROC-AUC} = 0.9021$**, **$\text{PR-AUC} = 0.8238$** e **$\text{Brier Score} = 0.1149$**, assegurando estimativas de risco contínuas (0% a 100%) aderentes à realidade empírica.
* **Interpretabilidade SHAP:** A decomposição por Shapley Additive exPlanations comprovou um efeito sinérgico agravante quando a privação de sono se associa a índices elevados de estresse.

> 📖 **Para conferir a análise detalhada, metodologia completa e todos os 6 painéis visuais comentados:**  
> 👉 [Consulte o README do SIRP-600](sirp-600/README.md)

---

### 2. `soccermon` — Soccer Athlete Health & Load Monitoring (Futebol Feminino de Elite)

* **Pasta do Estudo:** [`📁 soccermon/`](soccermon/)
* **Documentação Científica:** [`📄 soccermon/README.md`](soccermon/README.md)
* **Notebook Principal:** [`📓 soccermon/soccermon_pipeline.ipynb`](soccermon/soccermon_pipeline.ipynb)

#### 🎯 Resumo da Investigação
Investigação longitudinal do cruzamento multivariado entre variáveis cinemáticas de carga externa de movimento (GPS/IMU: desacelerações severas, sprints, ACWR) e parâmetros diários de rotina e bem-estar (duração e qualidade do sono, estresse, fadiga) na predição de risco de lesão de isquiotibiais em janela de 7 dias subsequentes ($[t+1, t+7]$).

#### 📊 Principais Conclusões e Diagnóstico Crítico
* **Diagnóstico de Viabilidade:** O estudo revelou que o SoccerMon **não é recomendado para predição direta de lesões** devido à prevalência irrisória (0,26% no teste, gerando precisão de ~3% no limiar padrão) e descontinuidade de prontuários médicos pelo clube em 2021.
* **Redirecionamento Recomendado (Pivot):** A base é excepcionalmente rica e recomendada para predição de **Prontidão Física (*Readiness*)**, **Fadiga Acumulada** e **Recuperação Subjetiva**, onde o feedback é contínuo e diário.
* **Performance Probabilística Atual:** O modelo calibrado atingiu **$\text{ROC-AUC} = 0.8411$**, **$\text{PR-AUC} = 0.0514$** (~20x superior à taxa base) e **$\text{Brier Score} = 0.0027$**.
* **Interpretabilidade SHAP & Interação 2D:** A decomposição por SHAP e a análise de interação 2D evidenciaram o efeito multiplicador da privação de sono ($< 6.0\text{h}$) associada a picos de desacelerações severas ($<-3\text{ m/s}^2$).

> 📖 **Para conferir a análise detalhada, comprovação matemática do desbalanceamento e os 7 painéis visuais comentados:**  
> 👉 [Consulte o README do SoccerMon](soccermon/README.md)

---

### 3. `multimodal_injury` — Multimodal Sports Injury Prediction

* **Pasta do Estudo:** [`📁 multimodal_injury/`](multimodal_injury/)
* **Documentação Científica:** [`📄 multimodal_injury/README.md`](multimodal_injury/README.md)
* **Notebook Principal:** [`📓 multimodal_injury/multimodal_sports_injury_pipeline.ipynb`](multimodal_injury/multimodal_sports_injury_pipeline.ipynb)

#### 🎯 Resumo da Investigação
Investigação longitudinal do cruzamento multivariado entre métricas de carga de treinamento (razão ACWR $ATL_7 / CTL_{28}$, intensidade subjetiva RPE, fadiga sistêmica) e parâmetros de recuperação e rotina via sensores vestíveis (qualidade do sono, escore de recuperação autonômica / HRV proxy) em 156 atletas monitorados ao longo de 6 meses (15.420 sessões).

#### 📊 Principais Conclusões e Métricas
* **Fatores Críticos:** O **Escore de Recuperação Autonômica / HRV Proxy ($\rho = -0.323$, >18% Gini)** e a **Qualidade do Sono ($\rho = -0.278$)** são os maiores fatores protetores. Por outro lado, o **Índice de Fadiga ($\rho = +0.334$)**, a **Carga Total ($\rho = +0.228$)** e picos de **ACWR ($\rho = +0.142$)** representam os principais impulsionadores de risco.
* **Performance Probabilística:** O modelo calibrado com Platt Scaling atingiu **$\text{ROC-AUC} = 0.8806$**, **$\text{PR-AUC} = 0.5494$** (3,66x a prevalência base de 15,01%), **$\text{Brier Score} = 0.0932$** e **$\text{Log Loss} = 0.3009$**.
* **Interpretabilidade SHAP & Interação 2D:** A análise de contorno bidimensional comprovou que a sobrecarga mecânica aguda (ACWR $> 1.4$) é multiplicada não linearmente quando imposta sobre um organismo em colapso autonômico (recuperação $< 40$), elevando a probabilidade empírica de risco para além de 80% a 90%.

> 📖 **Para conferir a análise detalhada, metodologia completa e todos os 7 painéis visuais comentados:**  
> 👉 [Consulte o README do Multimodal Injury](multimodal_injury/README.md)

---

## 🚀 Roadmap de Novos Cruzamentos

O repositório Scout está planejado para receber novos cruzamentos estatísticos e analíticos em esportes de alto rendimento. Dentre as bases mapeadas para integração:

* 🏃 **GPS & External Load Monitoring:** Cruzamento de métricas cinemáticas (distância total percorrida, volume em sprint $> 25\text{ km/h}$, desacelerações de alta intensidade) com marcadores bioquímicos de dano muscular (CK e PCR).
* ⚽ **Match Performance & Tactical Event Data:** Cruzamento de eventos técnicos (passes progressivos, xG/xA gerado, duelos defensivos ganhos) com minutagem e eficiência tática posicional.
* 🏀 **Load Management & Back-to-Back (NBA):** Avaliação de fadiga acumulada em viagens e jogos consecutivos sobre a eficiência de arremesso e lesões por sobrecarga.
* 🏋️ **Screening Biomecânico & Funcional:** Análise de testes de salto (Countermovement Jump - CMJ), dinamometria isocinética e desvios de valgo dinâmico de joelho em atletas profissionais.

---

## 📁 Estrutura do Repositório

```
📁 Scout/
│
├── 📄 README.md                            # Documentação principal e catálogo geral do repositório
├── 📄 .gitignore                           # Regras globais de exclusão do Git (datasets, venv, checkpoints)
│
├── 📁 sirp-600/                            # Módulo do Dataset SIRP-600
│   ├── 📄 README.md                        # Documentação científica dedicada do estudo SIRP-600
│   ├── 📁 figures/                         # Suíte de 6 figuras em alta resolução (300 DPI)
│   │   ├── 🖼️ metricas_avaliacao_calibracao.png
│   │   ├── 🖼️ matriz_correlacao_e_ranking.png
│   │   ├── 🖼️ distribuicoes_kde_fatores_risco.png
│   │   ├── 🖼️ feature_importance_rf.png
│   │   ├── 🖼️ shap_summary_beeswarm.png
│   │   └── 🖼️ shap_summary_bar.png
│   └── 📁 notebooks/                       # Notebooks interativos do estudo
│       └── 📓 analise_risco_lesao_sirp600.ipynb
│
├── 📁 soccermon/                           # Módulo do Dataset SoccerMon (Futebol Feminino de Elite)
│   ├── 📄 README.md                        # Documentação científica dedicada e diagnóstico de viabilidade
│   ├── 📓 soccermon_pipeline.ipynb         # Jupyter Notebook executado com todas as saídas e figuras
│   ├── 📁 figures/                         # Suíte de 7 figuras alinhadas em alta resolução (300 DPI)
│   │   ├── 🖼️ metricas_avaliacao_calibracao.png
│   │   ├── 🖼️ matriz_correlacao_e_ranking.png
│   │   ├── 🖼️ distribuicoes_kde_fatores_risco.png
│   │   ├── 🖼️ feature_importance_rf.png
│   │   ├── 🖼️ shap_summary_beeswarm.png
│   │   ├── 🖼️ shap_summary_bar.png
│   │   └── 🖼️ interacao_sono_desaceleracao.png
│   └── 📁 data/                            # Dados do SoccerMon (baixados programaticamente via Zenodo)
│       ├── 📁 subjective/                  # Bem-estar diário, cargas e prontuário de lesões
├── 📁 multimodal_injury/                   # Módulo Multimodal Sports Injury Dataset
│   ├── 📄 README.md                        # Documentação científica dedicada do estudo
│   ├── 📓 multimodal_sports_injury_pipeline.ipynb # Jupyter Notebook executado com saídas e figuras
│   ├── 📊 multimodal_sports_injury_dataset.csv # Base de dados (baixada via Kaggle API)
│   └── 📁 figures/                         # Suíte de 7 figuras em alta resolução (300 DPI)
│       ├── 🖼️ metricas_avaliacao_calibracao.png
│       ├── 🖼️ matriz_correlacao_e_ranking.png
│       ├── 🖼️ distribuicoes_kde_fatores_risco.png
│       ├── 🖼️ feature_importance_rf.png
│       ├── 🖼️ shap_summary_beeswarm.png
│       ├── 🖼️ shap_summary_bar.png
│       └── 🖼️ interacao_sono_acwr.png
│
└── 📁 [futuro-dataset]/                    # Próximos cruzamentos estruturados no mesmo padrão
    ├── 📄 README.md
    ├── 📊 dados.csv
    ├── 📁 figures/
    └── 📁 notebooks/
```

---

## 🛠️ Como Adicionar um Novo Cruzamento

Para integrar um novo conjunto de dados ao repositório mantendo o padrão arquitetural, siga os passos abaixo:

1. **Crie um diretório com o nome do dataset** (utilize letras minúsculas e hífen, e.g., `sirp-600`, `nba-tracking`, `soccer-events`):
   ```bash
   mkdir nome-do-dataset
   cd nome-do-dataset
   mkdir notebooks figures
   ```

2. **Adicione os dados e notebooks:**
   * Posicione o arquivo de dados na pasta do dataset (respeitando o `.gitignore` para dados confidenciais ou de grande porte).
   * Desenvolva o notebook de cruzamento dentro da subpasta `notebooks/`.
   * Salve os gráficos gerados com visual profissional na subpasta `figures/`.

3. **Crie o `README.md` dedicado da pasta:**
   * Contextualização do dataset e objetivos do cruzamento.
   * Análise Exploratória de Dados (EDA) com matriz de correlação.
   * Metodologia de modelagem ou inferência estatística.
   * Imagens geradas acompanhadas de explicação sobre o que cada gráfico demonstra.
   * Tabela de métricas e limitações metodológicas.
   * Instruções de execução local para o módulo.

4. **Atualize o Catálogo Principal:**
   * Adicione o novo dataset na tabela da seção [Catálogo de Cruzamentos e Datasets](#-catálogo-de-cruzamentos-e-datasets) deste README principal com links diretos.

---

## 💻 Instalação e Execução Local

### 1. Clonar ou Navegar até o Repositório
```bash
git clone https://github.com/MathhAraujo/scout_cruzamento.git
cd Scout
```

### 2. Configurar o Ambiente Virtual Python (Recomendado)
```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar no Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Ativar no Linux / macOS:
source .venv/bin/activate
```

### 3. Instalar as Dependências Globais
```bash
pip install numpy pandas scikit-learn matplotlib seaborn shap nbformat jupyter
```

### 4. Executar um Estudo Específico
Navegue até a pasta do dataset desejado e abra o Jupyter Notebook:
```bash
# Exemplo para o dataset SIRP-600:
cd sirp-600
jupyter notebook notebooks/analise_risco_lesao_sirp600.ipynb
```
*(Ou abra diretamente pelo VS Code / Cursor apontando para o interpretador do ambiente virtual criado).*

---

## ⚠️ Ressalvas Científicas e Metodológicas

> [!WARNING]
> ### 📌 Nota sobre Uso dos Modelos e Interpretação Estatística
> 
> * **Natureza Exploratória e Analítica:** Todos os estudos e cruzamentos presentes no repositório Scout têm como objetivo a investigação de padrões estatísticos, formulação de hipóteses e demonstração técnica de métodos modernos de Ciência de Dados e Machine Learning.
> * **Correlação $\neq$ Causalidade:** Associações estatísticas identificadas (como correlações lineares ou importância em árvores de decisão) não comprovam relação de causa e efeito direta. Fatores de confusão (*confounders*) e interações multivariadas devem ser considerados.
> * **Não Aplicabilidade Clínica Direta:** Os modelos e estimativas probabilísticas disponibilizados **não devem ser empregados como sistemas autônomos de diagnóstico ou prescrição médica/fisioterápica sem validação clínica prospectiva** conduzida por profissionais de saúde qualificados.

---

## 👨‍💻 Autor e Desenvolvimento

Desenvolvido no âmbito do projeto **Scout — Sports Analytics & Machine Learning Research**.  
Para dúvidas, discussões metodológicas ou sugestões de novos datasets, utilize as *Issues* do repositório.
