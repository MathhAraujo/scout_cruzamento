# ⚽ Scout — Hub de Inteligência e Cruzamento de Dados Esportivos

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.9.1-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-v0.52.0-red.svg)](https://shap.readthedocs.io/)
[![Arquitetura](https://img.shields.io/badge/Arquitetura-Multi--Dataset%20Modular-brightgreen.svg)](#-visão-geral-e-arquitetura)
[![Status](https://img.shields.io/badge/Status-Estudos%20Concluídos-brightgreen.svg)](#-catálogo-de-cruzamentos-e-datasets)

O **Scout** é um repositório centralizado e modular para ciência de dados esportivos (*Sports Analytics & Machine Learning*). O projeto reúne investigações empíricas, cruzamentos estatísticos multivariados e modelagem preditiva e probabilística aplicada a fatores de rendimento, controle de carga de treinamento, assimetrias biomecânicas e risco de lesões musculoesqueléticas em atletas.

Para manter a escalabilidade e o rigor metodológico, **cada conjunto de dados analisado reside em seu próprio módulo dedicado**, contendo seus dados brutos e processados, suítes de visualização analítica (300 DPI), notebooks interativos reproduzíveis e documentação técnica aprofundada.

---

## 📋 Sumário Geral

1. [Visão Geral e Arquitetura](#-visão-geral-e-arquitetura)
2. [Catálogo de Cruzamentos e Datasets](#-catálogo-de-cruzamentos-e-datasets)
   - [SIRP-600 (Sports Injury Risk Prediction)](#1-sirp-600--sports-injury-risk-prediction)
   - [SoccerMon (Futebol Feminino de Elite)](#2-soccermon--soccer-athlete-health--load-monitoring)
   - [Multimodal Sports Injury (Carga, Sono e Recuperação Autonômica)](#3-multimodal_injury--multimodal-sports-injury-prediction)
3. [Síntese Consolidada de Resultados: As Variáveis Mais Determinantes](#-síntese-consolidada-de-resultados-as-variáveis-mais-determinantes)
   - [Tabela Geral de Preditores de Maior Relevância](#-tabela-geral-de-preditores-de-maior-relevância)
   - [Detalhamento Conceitual, Fisiológico e Impacto Estatístico das Variáveis](#-detalhamento-conceitual-fisiológico-e-impacto-estatístico-das-variáveis)
   - [A Dinâmica Não Linear: O Desacoplamento entre Demanda e Recuperação](#-a-dinâmica-não-linear-o-desacoplamento-entre-demanda-e-recuperação)
4. [Comparativo Intermódulos e Lições sobre Viabilidade Amostral](#-comparativo-intermódulos-e-lições-sobre-viabilidade-amostral)
5. [Ressalvas Científicas e Metodológicas](#-ressalvas-científicas-e-metodológicas)

---

## 🏛️ Visão Geral e Arquitetura

O repositório é concebido sob o princípio de **módulos analíticos autocontidos**:

* **Independência Operacional:** Cada pasta correspondente a um dataset possui seu próprio pipeline de análise, dados, notebooks e figuras, permitindo executar e evoluir um estudo sem interferir nos demais.
* **Documentação Dedicada:** Cada cruzamento conta com um `README.md` individual detalhando objetivos, análise exploratória (EDA), formulações matemáticas, interpretação visual das figuras e tabelas de métricas.
* **Padronização Visual e Técnica:** Todas as análises seguem padrões consolidados de boas práticas: reprodutibilidade com sementes fixas, calibração de probabilidades contínuas (Platt Scaling), explicabilidade com valores SHAP e exportação de gráficos em alta resolução (300 DPI).

---

## 🗂️ Catálogo de Cruzamentos e Datasets

Abaixo estão listados os estudos analíticos e cruzamentos de dados desenvolvidos no repositório:

| Diretório / Dataset | Tema Central | Amostra | Algoritmos & Técnicas | Status | Link para Estudo |
| :--- | :--- | :---: | :--- | :---: | :---: |
| [`sirp-600/`](sirp-600/) | **Predição de Risco de Lesão Esportiva** | 600 atletas | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP | Concluído | [Acessar Estudo](sirp-600/README.md) |
| [`soccermon/`](soccermon/) | **Carga Externa (GPS), Sono e Risco de Lesão (7d)** | 50 atletas (36.550 atleta-dias) | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP, ACWR | Concluído (Diagnóstico Crítico) | [Acessar Estudo](soccermon/README.md) |
| [`multimodal_injury/`](multimodal_injury/) | **Carga (ACWR, RPE), Sono e Recuperação Autonômica (HRV)** | 156 atletas (15.420 sessões) | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP, Gabbett ACWR | Concluído | [Acessar Estudo](multimodal_injury/README.md) |

---

### 1. `sirp-600` — Sports Injury Risk Prediction

* **Pasta do Estudo:** [`📁 sirp-600/`](sirp-600/)
* **Documentação Científica Dedicada:** [`📄 sirp-600/README.md`](sirp-600/README.md)
* **Notebook Principal:** [`📓 sirp-600/notebooks/analise_risco_lesao_sirp600.ipynb`](sirp-600/notebooks/analise_risco_lesao_sirp600.ipynb)

#### 🎯 Resumo da Investigação
Investigação empírica das interações entre variáveis comportamentais de rotina (sono, aquecimento, intensidade de treino), dimensões antropométricas, fisiológicas (tempo de recuperação cardíaca) e mecânicas funcionais (assimetria muscular contralateral) na ocorrência de lesões musculares e articulares.

#### 📊 Principais Conclusões e Métricas
* **Fatores Críticos:** A **Assimetria Muscular Contralateral ($r = +0.374$, 17.9% Gini)** e o **Histórico de Lesões ($r = +0.307$)** revelaram-se os maiores impulsionadores de risco. Por outro lado, **Tempo de Aquecimento ($r = -0.273$)** e **Horas de Sono ($r = -0.270$)** atuam como fortes moduladores protetores.
* **Performance Probabilística:** O modelo calibrado atingiu **$\text{ROC-AUC} = 0.9021$**, **$\text{PR-AUC} = 0.8238$** e **$\text{Brier Score} = 0.1149$**, assegurando estimativas de risco contínuas (0% a 100%) aderentes à realidade empírica.
* **Interpretabilidade SHAP:** A decomposição por Shapley Additive exPlanations comprovou um efeito sinérgico agravante quando a privação de sono se associa a índices elevados de estresse e déficits de simetria bilateral.

---

### 2. `soccermon` — Soccer Athlete Health & Load Monitoring

* **Pasta do Estudo:** [`📁 soccermon/`](soccermon/)
* **Documentação Científica Dedicada:** [`📄 soccermon/README.md`](soccermon/README.md)
* **Notebook Principal:** [`📓 soccermon/soccermon_pipeline.ipynb`](soccermon/soccermon_pipeline.ipynb)

#### 🎯 Resumo da Investigação
Investigação longitudinal do cruzamento multivariado entre variáveis cinemáticas de carga externa de movimento (GPS/IMU: desacelerações severas, sprints, ACWR) e parâmetros diários de rotina e bem-estar (duração e qualidade do sono, estresse, fadiga) na predição de risco de lesão de isquiotibiais em janela de 7 dias subsequentes ($[t+1, t+7]$).

#### 📊 Diagnóstico Crítico de Viabilidade
* **Inviabilidade para Predição Direta de Lesão:** O estudo revelou que o SoccerMon **não é recomendado para predição direta de lesão em grade calendária**, devido à atrição na notificação médica no 2º ano (apenas 4 lesões no período de teste) e prevalência de apenas **0,26%**, gerando **91% a 97% de alarmes falsos** ($\text{PR-AUC} = 0.0514$).
* **Redirecionamento Recomendado (Pivot):** Por possuir mais de 17.700 registros densos ininterruptos de bem-estar, a base é recomendada para predição de **Prontidão Física Diária (*Athlete Readiness*)** e **Fadiga**, e não para eventos raros de lesão subnotificados.

---

### 3. `multimodal_injury` — Multimodal Sports Injury Prediction

* **Pasta do Estudo:** [`📁 multimodal_injury/`](multimodal_injury/)
* **Documentação Científica Dedicada:** [`📄 multimodal_injury/README.md`](multimodal_injury/README.md)
* **Notebook Principal:** [`📓 multimodal_injury/multimodal_sports_injury_pipeline.ipynb`](multimodal_injury/multimodal_sports_injury_pipeline.ipynb)

#### 🎯 Resumo da Investigação
Investigação longitudinal do cruzamento multivariado entre métricas de carga de treinamento (razão ACWR $ATL_7 / CTL_{28}$, intensidade subjetiva RPE, fadiga sistêmica) e parâmetros de recuperação e rotina via sensores vestíveis (qualidade do sono, escore de recuperação autonômica / HRV proxy) em 156 atletas monitorados ao longo de 6 meses (15.420 sessões).

#### 📊 Principais Conclusões e Métricas
* **Fatores Críticos:** O **Escore de Recuperação Autonômica / HRV Proxy ($\rho = -0.323$, >18% Gini)** e a **Qualidade do Sono ($\rho = -0.278$)** são os maiores fatores protetores. Por outro lado, o **Índice de Fadiga ($\rho = +0.334$)**, a **Carga Total ($\rho = +0.228$)** e picos de **ACWR ($\rho = +0.142$)** representam os principais impulsionadores de risco.
* **Performance Probabilística:** O modelo calibrado com Platt Scaling atingiu **$\text{ROC-AUC} = 0.8806$**, **$\text{PR-AUC} = 0.5494$** (3,66x a prevalência base de 15,01%), **$\text{Brier Score} = 0.0932$** e **$\text{Log Loss} = 0.3009$**.
* **Interpretabilidade SHAP & Interação 2D:** A análise de contorno bidimensional comprovou que a sobrecarga mecânica aguda (ACWR $> 1.4$) é multiplicada não linearmente quando imposta sobre um organismo em colapso autonômico (recuperação $< 40$), elevando a probabilidade empírica de risco para além de 80% a 90%.

---

## 🏆 Síntese Consolidada de Resultados: As Variáveis Mais Determinantes

Ao cruzar os achados obtidos nos estudos que apresentaram solidez amostral e alto valor preditivo ([`sirp-600`](sirp-600/) e [`multimodal_injury`](multimodal_injury/)), foi possível identificar **quais variáveis biológicas, comportamentais e de carga exercem maior impacto real sobre a predição de lesão**, superando amplamente características antropométricas estáticas (como idade, peso ou gênero isolados).

> [!NOTE]
> Conforme constatado no diagnóstico científico do repositório, o cruzamento do **SoccerMon** apresentou taxa extrema de falsos alarmes (91% a 97%) decorrente de subnotificação médica em séries temporais de dias vazios. Por essa razão, **os preditores abaixo refletem estritamente as bases com consistência estatística validada**, garantindo que as conclusões técnicas sejam fiéis aos dados reais.

---

### 📊 Tabela Geral de Preditores de Maior Relevância

A tabela a seguir consolida as variáveis mais influentes encontradas no repositório, ranqueadas por relevância estatística, capacidade de discriminação e magnitude de efeito nos modelos de Machine Learning (Random Forest + Platt Scaling + TreeSHAP):

| Variável / Fator de Monitoramento | Domínio Esportivo | Origem Primária | Associação Estatística (r ou ρ) | Importância nas Árvores (Gini) | Impacto Médio SHAP | Direção do Efeito no Risco |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Assimetria Muscular Contralateral** | Mecânica Funcional | `sirp-600` | r = +0.374 | **17.9%** (Top 1) | **> 0.12** (Top 1 absoluto) | 🔴 **Agravante Forte** (risco dispara com assimetria > 8%) |
| **2. Escore de Recuperação Autonômica (HRV Proxy)** | Fisiologia / Wearables | `multimodal_injury` | ρ = -0.323 | **> 18.0%** (com média móvel 7d) | **> 0.07** (Top 1) | 🟢 **Protetor Crítico** (queda < 40 antecede lesão) |
| **3. Índice de Fadiga Sistêmica Percebida** | Carga Interna / Bem-Estar | `multimodal_injury` | ρ = +0.334 | Top 2 do modelo | > 0.05 | 🔴 **Agravante Primário** (níveis > 70 disparam vulnerabilidade) |
| **4. Histórico de Lesões Anteriores** | Clínico / Recidiva | `sirp-600` | r = +0.307 | Elevado ganho de pureza | > 0.06 | 🔴 **Agravante Estrutural** (histórico prévio multiplica risco) |
| **5. Sono: Duração (Horas) e Qualidade Subjetiva** | Rotina / Regeneração | Consistente em ambos | r = -0.270 e ρ = -0.278 | **9.5%** em `sirp-600` | 0.06 a 0.07 | 🟢 **Protetor Fundamental** (sono < 6h eleva risco) |
| **6. Tempo de Aquecimento Pré-Treino** | Rotina / Treinamento | `sirp-600` | r = -0.273 | **9.1%** | > 0.05 | 🟢 **Protetor Ativo** (< 10 min concentra 80% das lesões) |
| **7. Razão de Carga Aguda:Crônica (ACWR)** | Carga Externa / Treino | `multimodal_injury` | ρ = +0.142 | Relevante em interação | Modulador sinérgico | ⚠️ **Agravante Não Linear** (perigoso quando ACWR > 1.4) |
| **8. Tempo de Recuperação Cardíaca Pós-Esforço** | Fisiologia Cardiovascular | `sirp-600` | r = +0.222 | **9.7%** | > 0.06 | 🔴 **Agravante de Desgaste** (recuperação lenta reflete estresse) |
| **9. Nível de Estresse Psicológico** | Psicofisiologia | `sirp-600` | r = +0.285 | **9.2%** | > 0.06 | 🔴 **Agravante Fisiológico** (elevação de tônus basal e cortisol) |

---

### 🔬 Detalhamento Conceitual, Fisiológico e Impacto Estatístico das Variáveis

#### 1. Assimetria Muscular Contralateral (`muscle_imbalance_%`)
* **O que é:** Percentual de discrepância mecânica de força isométrica/isocinética ou de secção muscular transversa entre o membro dominante e o contralateral (ex: quadríceps direito vs. esquerdo ou razão isquiotibiais/quadríceps).
* **O quanto influencia:**
  * Apresentou a maior correlação linear de todo o dataset SIRP-600 (**$r = +0.374$**).
  * Liderou a importância de divisão das árvores com **$17.9\%$ de redução de impureza de Gini** e o impacto marginal SHAP com magnitude **$\text{mean } |\text{SHAP}| > 0.12$**.
* **Significado Esportivo:** Diferenças bilaterais superiores a **$8\% - 10\%$** forçam transferências compensatórias de torque mecânico durante sprints, frenagens e saltos. O membro fraco atinge falha excêntrica prematura, enquanto o membro oposto sofre sobrecarga reativa nas estruturas tendíneas e articulares.

#### 2. Escore de Recuperação Autonômica / HRV Proxy (`recovery_score` & `recovery_score_rolling7`)
* **O que é:** Índice escalar contínuo (0 a 100) derivado de sensores vestíveis esportivos noturnos, que atua como proxy do tônus vagal e da modulação do Sistema Nervoso Autônomo (SNA), refletindo a prontidão tecidual para absorção de carga.
* **O quanto influencia:**
  * Maior correlação protetora isolada do estudo multimodal (**$\rho = -0.323$**).
  * Somado à sua média móvel de 7 dias, responde por **mais de $18\%$ da importância total de Gini** do ensemble e gera o maior deslocamento absoluto de probabilidade (**$\text{mean } |\text{SHAP}| > 0.07$**).
* **Significado Esportivo:** Valores colapsados ($< 40$) revelam que o organismo permanece sob predomínio simpático de estresse metabólico. Treinar em alta intensidade nesse estado esgota a capacidade de reparação de microdanos musculares, elevando drasticamente a probabilidade de falha tecidual.

#### 3. Sono: Duração Contínua e Qualidade Subjetiva (`sleep_hours` & `sleep_quality`)
* **O que é:** Medição quantitativa de horas totais de descanso por noite associada à qualidade percebida (arquitetura do sono profundo / sono de ondas lentas).
* **O quanto influencia:**
  * Correlação protetora consistente em ambos os módulos (**$r = -0.270$** no SIRP-600; **$\rho = -0.278$** no Multimodal Injury).
  * Distribuição KDE comprovou que atletas lesionados dormem significativamente menos (média de **$6.1\text{h}$ vs. $7.6\text{h}$** no SIRP-600; qualidade média de **$4.57$ vs. $6.03$** no Multimodal).
  * No SHAP Beeswarm, a privação de sono ($< 6.0\text{h}$) desloca os valores de risco invariavelmente para a margem positiva ($\text{SHAP} > +0.25$).
* **Significado Esportivo:** É durante o sono profundo que ocorrem os picos de secreção de hormônio do crescimento (GH), síntese proteica miofibrilar e depuração de metabólitos inflamatórios. A restrição crônica de sono reduz o tempo de reação neuromuscular e compromete a coordenação motora fina sob fadiga.

#### 4. Índice de Fadiga Sistêmica Percebida (`fatigue_index`)
* **O que é:** Grau de esgotamento psicofisiológico relatado pelo atleta em escala contínua (0 a 100), sensível ao estresse acumulado das sessões de treino e calendários de viagens.
* **O quanto influencia:**
  * Apresentou a maior correlação de risco positivo no estudo multimodal (**$\rho = +0.334$**).
  * Médias de densidade KDE revelaram contraste evidente: **$76.47$** nos atletas que sofreram lesão contra **$56.92$** nos atletas saudáveis.
* **Significado Esportivo:** A fadiga sistêmica atua como precursora da fadiga periférica local. Quando atinge níveis elevados ($> 70$), há inibição neural reflexa, alteração nos padrões de recrutamento motor e incapacidade de absorver cargas excêntricas de alta intensidade.

#### 5. Histórico de Lesões Anteriores (`previous_injuries`)
* **O que é:** Contagem cumulativa de eventos prévios de estiramentos, entorses ou rupturas musculares e ligamentares registradas na carreira do atleta.
* **O quanto influencia:**
  * Forte correlação linear com novas ocorrências (**$r = +0.307$**).
  * O gráfico SHAP revelou que ter $2$ ou mais lesões prévias adiciona uma penalidade de risco basal permanente, agindo como multiplicador quando o atleta negligencia o sono ou o aquecimento.
* **Significado Esportivo:** Tecidos biológicos cicatrizados substituem fibras contráteis organizadas por colágeno tipo III denso, caracterizado por menor elasticidade e menor capacidade de deformação sob tensão máxima. Cria-se um ponto de concentração de estresse mecânico na transição miotendínea, propenso à recidiva.

#### 6. Razão de Carga Aguda:Crônica (ACWR — *Acute:Chronic Workload Ratio*)
* **O que é:** Relação matemática entre a carga de treino dos últimos 7 dias ($ATL_7$ — fadiga recente) e a média móvel dos últimos 28 dias ($CTL_{28}$ — capacidade crônica/condicionamento), baseada na modelagem de Tim Gabbett (2016):
  $$\text{ACWR} = \frac{ATL_7}{CTL_{28} + 10^{-6}}$$
* **O quanto influencia:**
  * Correlação monotônica de **$\rho = +0.142$**.
  * Embora com coeficiente bivariado moderado, possui **o mais forte efeito multiplicador não linear** quando cruzada com o estado fisiológico do atleta.
* **Significado Esportivo:** Cargas no intervalo de $0.8$ a $1.2$ representam o *"sweet spot"* de adaptação atlética segura. Picos repentinos com $\text{ACWR} > 1.4$ dobram ou triplicam a probabilidade empírica de sobrecarga quando o atleta não possui lastro crônico suficiente para absorver a nova demanda.

#### 7. Tempo de Aquecimento Pré-Sessão (`warmup_time_min`)
* **O que é:** Minutos dedicados a protocolos de mobilidade articular, ativação neuromuscular, corrida progressiva e exercícios dinâmicos antes de esforços em intensidade máxima.
* **O quanto influencia:**
  * Correlação protetora expressiva de **$r = -0.273$** e **$9.1\%$ de importância de Gini**.
  * A distribuição KDE evidenciou que os lesionados apresentaram média de apenas **$9.4\text{ minutos}$**, enquanto atletas saudáveis dedicaram em média **$17.6\text{ minutos}$**.
* **Significado Esportivo:** O aquecimento estruturado eleva a temperatura intramuscular (reduzindo a viscosidade viscoelástica das miofibrilas), otimiza a condução de potenciais de ação neuromuscular e melhora a complacência de tendões e ligamentos. Sessões com menos de $10\text{ minutos}$ deixam as estruturas vulneráveis a tensões de cisalhamento.

---

### 🔄 A Dinâmica Não Linear: O Desacoplamento entre Demanda e Recuperação

Um dos achados analíticos mais expressivos consolidados no repositório é que **a lesão esportiva não é explicada por nenhuma variável de forma isolada**, mas sim pelo **desacoplamento entre a carga mecânica imposta e a capacidade biológica de regeneração**.

A figura abaixo, extraída da análise de contorno bidimensional do módulo [`multimodal_injury`](multimodal_injury/figures/interacao_sono_acwr.png), sintetiza matematicamente essa interação:

```
                            MATRIZ CONCEITUAL DE INTERAÇÃO 2D
               
   Escore de Recuperação / Sono
             ▲
        Alta │  🟢 RISCO BAIXO (< 15%)            🟡 RISCO MODERADO (25% - 40%)
       (>75) │  Organismo adaptado absorve picos  Estresse controlado pelo lastro fisiológico
             │
             │
       Média │  🟡 RISCO MODERADO                 🟠 RISCO ELEVADO (50% - 65%)
     (50-70) │  Vigilância recomendada            Necessidade de modulação de volume
             │
             │
       Baixa │  🟠 RISCO ELEVADO                  🔴 ZONA CRÍTICA DE RUPTURA (> 80% - 90%)
       (<40) │  Fadiga prévia sem sobrecarga      Desacoplamento: pico de carga aguda imposto
             │                                    a um tecido em colapso autonômico
             └────────────────────────────────────────────────────────────────────────►
                   Carga Estável (ACWR 0.8 - 1.2)           Pico Agudo (ACWR > 1.4 - 1.6)
```

* **Cenário 1 (Tolerância Fisiológica):** Quando o atleta apresenta recuperação autonômica robusta ($\text{Escore} > 75$) e sono consistente, mesmo uma elevação aguda de carga ($\text{ACWR} = 1.35$) mantém o risco de lesão contido na faixa segura ($< 15\%$).
* **Cenário 2 (Multiplicação de Risco):** Em contraste, quando o atleta apresenta déficit autonômico ou privação de sono ($\text{Escore} < 40$) e é submetido ao mesmo pico agudo ($\text{ACWR} > 1.40$), **a probabilidade de lesão salta de forma não linear para a faixa de $80\% \text{ a } 90\%$**, demonstrando que o colapso estrutural ocorre pela combinação de sobrecarga e esgotamento.

---

## 🔬 Comparativo Intermódulos e Lições sobre Viabilidade Amostral

A tabela abaixo sintetiza os resultados de machine learning alcançados nos três cruzamentos do repositório, evidenciando as diferenças metodológicas entre cenários estáticos, séries temporais longas com subnotificação clínica e séries multimodais contínuas:

| Dimensão Metodológica | Módulo [`sirp-600`](sirp-600/) | Módulo [`soccermon`](soccermon/) | Módulo [`multimodal_injury`](multimodal_injury/) |
| :--- | :---: | :---: | :---: |
| **Natureza dos Dados** | Tabular Estático Transversal | Série Temporal Diária de Futebol Feminino | Série Temporal Longitudinal Multimodal |
| **Volume Amostral** | 600 atletas individuais | 50 atletas (36.550 dias calendário) | 156 atletas (15.420 sessões de treino) |
| **Taxa de Prevalência Positiva** | 31,5% (Balanceado) | 0,26% no teste (Desbalanceamento extremo) | **15,01% (Equilibrado e representativo)** |
| **Subnotificação / Atrição Clínica** | Inexistente (Base estática benchmark) | Severa (Clube abandonou anotações em 2021) | Inexistente (6 meses ininterruptos) |
| **ROC-AUC Score** | **`0.9021`** | **`0.8410`** | **`0.8806`** |
| **PR-AUC (Average Precision)** | **`0.8238`** | **`0.0349`** (Diluído em dias vazios) | **`0.5494`** (3,66x a taxa aleatória) |
| **Brier Score (Calibrado)** | **`0.1149`** | **`0.0030`** | **`0.0932`** |
| **Log Loss** | **`0.3578`** | **`0.0191`** | **`0.3009`** |
| **Conclusão Metodológica** | **Excelente para fatores estruturais:** Assimetria muscular, sono e aquecimento explicam risco individual. | **Inadequado para lesão diária:** Recomendado pivotamento do corpus para Prontidão (*Readiness*) e Fadiga. | **Excelente para interação carga-recuperação:** Comprova a dinâmica entre ACWR e déficit fisiológico. |

---

## ⚠️ Ressalvas Científicas e Metodológicas

> [!WARNING]
> ### 📌 Nota sobre Uso dos Modelos e Interpretação Estatística
> 
> * **Natureza Exploratória e Analítica:** Todos os estudos e cruzamentos presentes no repositório Scout têm como objetivo a investigação de padrões estatísticos, formulação de hipóteses e demonstração técnica de métodos modernos de Ciência de Dados e Machine Learning aplicados ao esporte.
> * **Correlação $\neq$ Causalidade:** Associações estatísticas identificadas (como correlações monotônicas de Spearman, importâncias em árvores de decisão e valores SHAP) não comprovam relação de causa e efeito direta. Fatores de confusão (*confounders*) e interações multivariadas não mensuradas devem ser considerados.
> * **Não Aplicabilidade Clínica Direta:** Os modelos e estimativas probabilísticas disponibilizados **não devem ser empregados como sistemas autônomos de diagnóstico ou prescrição médica/fisioterápica sem validação clínica prospectiva** conduzida por profissionais de saúde e comissões multidisciplinares devidamente habilitadas.
