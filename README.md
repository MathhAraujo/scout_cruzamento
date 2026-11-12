# ⚽ Scout — Hub de Inteligência e Cruzamento de Dados Esportivos

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.9.1-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-v0.52.0-red.svg)](https://shap.readthedocs.io/)
[![Arquitetura](https://img.shields.io/badge/Arquitetura-Multi--Dataset%20Modular-brightgreen.svg)](#-visão-geral-e-arquitetura)
[![Status](https://img.shields.io/badge/Status-Estudos%20e%20Auditorias-blue.svg)](#-catálogo-de-cruzamentos-e-datasets)

O **Scout** é um repositório centralizado e modular para ciência de dados esportivos (*Sports Analytics & Machine Learning*). O projeto reúne investigações empíricas, cruzamentos estatísticos multivariados e modelagem preditiva e probabilística aplicada a fatores de rendimento, controle de carga de treinamento, assimetrias biomecânicas e risco de lesões musculoesqueléticas em atletas.

Para manter a escalabilidade e o rigor metodológico, **cada conjunto de dados analisado reside em seu próprio módulo dedicado**, contendo seus dados brutos e processados, visualizações analíticas, notebooks interativos reproduzíveis e documentação técnica aprofundada.

---

## 📋 Sumário Geral

1. [Visão Geral e Arquitetura](#-visão-geral-e-arquitetura)
2. [Catálogo de Cruzamentos e Datasets](#-catálogo-de-cruzamentos-e-datasets)
   - [SIRP-600 (Sports Injury Risk Prediction)](#1-sirp-600--sports-injury-risk-prediction)
   - [SoccerMon (Futebol Feminino de Elite)](#2-soccermon--soccer-athlete-health--load-monitoring)
   - [Multimodal Sports Injury (Carga, Sono e Recuperação Autonômica)](#3-multimodal_injury--multimodal-sports-injury-prediction)
   - [Shalaj (Lesões de Isquiotibiais)](#4-shalaj--lesões-de-isquiotibiais)
   - [University Football (Rotina e Condição Física)](#5-university_football--rotina-e-condição-física)
   - [SoccerNet-GSR (Movimento por Vídeo)](#6-soccernet_gsr--movimento-por-vídeo)
3. [Síntese Consolidada de Resultados: Classificação das Variáveis](#-síntese-consolidada-de-resultados-classificação-das-variáveis)
   - [Tabela Geral de Variáveis e Evidências](#-tabela-geral-de-variáveis-e-evidências)
   - [Detalhamento Conceitual e Impacto Estatístico das Variáveis](#-detalhamento-conceitual-e-impacto-estatístico-das-variáveis)
   - [A Dinâmica Não Linear: O Desacoplamento entre Demanda e Recuperação](#-a-dinâmica-não-linear-o-desacoplamento-entre-demanda-e-recuperação)
4. [Comparativo Intermódulos e Lições sobre Viabilidade Amostral](#-comparativo-intermódulos-e-lições-sobre-viabilidade-amostral)
5. [Ressalvas Científicas e Metodológicas](#-ressalvas-científicas-e-metodológicas)

---

## 🏛️ Visão Geral e Arquitetura

O repositório é concebido sob o princípio de **módulos analíticos autocontidos**:

* **Independência Operacional:** Cada pasta correspondente a um dataset possui seu próprio pipeline de análise, dados, notebooks e figuras, permitindo executar e evoluir um estudo sem interferir nos demais.
* **Documentação Dedicada:** Cada cruzamento conta com um `README.md` individual detalhando objetivos, análise exploratória (EDA), formulações matemáticas, interpretação visual das figuras e tabelas de métricas.
* **Padronização Visual e Técnica:** Os módulos documentam seus métodos, métricas e limitações. Os estudos tabulares incluem Random Forest e calibração de probabilidades; os novos cruzamentos também incluem regressão logística e avaliação fora do treino (OOF). A explicação por SHAP pertence aos estudos anteriores; o SoccerNet-GSR apresenta análise cinemática descritiva.

---

## 🗂️ Catálogo de Cruzamentos e Datasets

Abaixo estão listados os estudos analíticos e cruzamentos de dados desenvolvidos no repositório:

| Diretório / Dataset | Tema Central | Amostra | Algoritmos & Técnicas | Status | Link para Estudo |
| :--- | :--- | :---: | :--- | :---: | :---: |
| [`sirp-600/`](sirp-600/) | **Predição de Risco de Lesão Esportiva** | 600 atletas | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP | Concluído | [Acessar Estudo](sirp-600/README.md) |
| [`soccermon/`](soccermon/) | **Carga Derivada de Sessões, Sono e Registro de Lesão (7d)** | 50 atletas (36.550 atleta-dias) | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP, ACWR | Concluído (Diagnóstico Crítico) | [Acessar Estudo](soccermon/README.md) |
| [`multimodal_injury/`](multimodal_injury/) | **Carga (ACWR, RPE), Sono e Recuperação Autonômica (HRV)** | 156 atletas (15.420 sessões) | Random Forest, Platt Scaling (`CalibratedClassifierCV`), SHAP, Gabbett ACWR | Concluído | [Acessar Estudo](multimodal_injury/README.md) |
| [`shalaj/`](shalaj/) | **Testes de Pré-Temporada e Lesão de Isquiotibiais** | 143 atletas (40 com lesão) | Regressão logística, Random Forest calibrada, Spearman, Fisher | Cruzamento concluído; validação interna | [Acessar Estudo](shalaj/README.md) |
| [`university_football/`](university_football/) | **Rotina, Condição Física e Rótulo de Risco** | 800 registros | Regressão logística, Random Forest calibrada, Spearman | Cruzamento exploratório concluído; alvo incerto | [Acessar Estudo](university_football/README.md) |
| [`soccernet_gsr/`](soccernet_gsr/) | **Posições, Trajetórias e Cinemática no Campo** | 164 clips (3.481 trajetórias locais) | Junção de frames e anotações, derivadas suavizadas | Cruzamento cinemático concluído; sem alvo clínico | [Acessar Estudo](soccernet_gsr/README.md) |

---

### 1. `sirp-600` — Sports Injury Risk Prediction

* **Pasta do Estudo:** [`📁 sirp-600/`](sirp-600/)
* **Documentação Científica Dedicada:** [`📄 sirp-600/README.md`](sirp-600/README.md)
* **Notebook Principal:** [`📓 sirp-600/notebooks/analise_risco_lesao_sirp600.ipynb`](sirp-600/notebooks/analise_risco_lesao_sirp600.ipynb)

#### 🎯 Resumo da Investigação
Investigação empírica das interações entre variáveis comportamentais de rotina (sono, aquecimento, intensidade de treino), dimensões antropométricas, fisiológicas (tempo de recuperação cardíaca) e mecânicas funcionais (assimetria muscular contralateral) na ocorrência de lesões musculares e articulares.

#### 📊 Principais Conclusões e Métricas
* **Fatores Associados na Base:** A **Assimetria Muscular Contralateral ($r = +0.374$, 17.9% Gini)** e o **Histórico de Lesões ($r = +0.307$)** apresentaram associação positiva com o alvo. **Tempo de Aquecimento ($r = -0.273$)** e **Horas de Sono ($r = -0.270$)** apresentaram associação inversa. A classificação agregada abaixo atualiza a interpretação desses resultados com a coorte de Shalaj e as limitações do benchmark.
* **Performance Probabilística:** O modelo calibrado atingiu **$\text{ROC-AUC} = 0.9021$**, **$\text{PR-AUC} = 0.8238$** e **$\text{Brier Score} = 0.1149$**, no estudo de benchmark, sem validação de probabilidades clínicas fora dessa base.
* **Interpretabilidade SHAP:** A decomposição por Shapley Additive exPlanations descreveu padrões do modelo quando a privação de sono se associa a índices elevados de estresse e déficits de simetria bilateral.

---

### 2. `soccermon` — Soccer Athlete Health & Load Monitoring

* **Pasta do Estudo:** [`📁 soccermon/`](soccermon/)
* **Documentação Científica Dedicada:** [`📄 soccermon/README.md`](soccermon/README.md)
* **Notebook Principal:** [`📓 soccermon/soccermon_pipeline.ipynb`](soccermon/soccermon_pipeline.ipynb)

#### 🎯 Resumo da Investigação
Investigação longitudinal do cruzamento entre carga, rotina e bem-estar (sono, estresse e fadiga), com alvo de registro de lesão de **coxa** nos sete dias seguintes. As colunas cinemáticas do notebook são derivadas de duração e sRPE; os arquivos GPS locais não são lidos nesse pipeline. O rótulo anatômico não distingue especificamente os isquiotibiais.

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
* **Fatores Associados na Base:** O **Escore de Recuperação Autonômica / HRV Proxy ($\rho = -0.323$, >18% Gini)** e a **Qualidade do Sono ($\rho = -0.278$)** apresentam associação inversa com o alvo. Por outro lado, o **Índice de Fadiga ($\rho = +0.334$)**, a **Carga Total ($\rho = +0.228$)** e picos de **ACWR ($\rho = +0.142$)** apresentam associação positiva com o alvo.
* **Performance Probabilística:** O modelo calibrado com Platt Scaling atingiu **$\text{ROC-AUC} = 0.8806$**, **$\text{PR-AUC} = 0.5494$** (3,66x a prevalência base de 15,01%), **$\text{Brier Score} = 0.0932$** e **$\text{Log Loss} = 0.3009$**.
* **Interpretabilidade SHAP & Interação 2D:** Os cenários produzidos pelo modelo mostram interação entre carga e recuperação na base analisada. Não demonstram causalidade nem validam limiares individuais de lesão. As janelas móveis do código são de 7/28 **sessões**, não necessariamente dias.

---

### 4. `shalaj` — Lesões de Isquiotibiais

* **Pasta do Estudo:** [`📁 shalaj/`](shalaj/)
* **Documentação Científica Dedicada:** [`📄 shalaj/README.md`](shalaj/README.md)
* **Notebook Principal:** [`📓 shalaj/shalaj_pipeline.ipynb`](shalaj/shalaj_pipeline.ipynb)

#### 🎯 Resumo da Investigação
Cruzamento entre testes de pré-temporada, histórico de lesões e ocorrência de lesão de isquiotibiais ao longo da temporada 2013/2014. O suplemento público contém 143 atletas, com 40 positivos. Os 43 episódios relatados no artigo não correspondem a 43 atletas distintos.

#### 📊 Principais Conclusões e Métricas
* **Fatores Associados:** Idade (ρ = +0,474), falha no Nordic (ρ = +0,286), IMC (ρ = +0,250) e histórico anterior (ρ = +0,191) têm associação marginal positiva com a ocorrência de lesão.
* **Contraste com SIRP-600:** A assimetria de torque a 60°/s calculada neste módulo tem ρ = −0,065. O resultado não reproduz a associação positiva do SIRP-600; protocolos e populações diferem.
* **Desempenho Interno:** RF calibrada: **ROC-AUC = 0,8235**, **AP = 0,6266**, **Brier = 0,1512**, **Log loss = 0,4600**. Logística: ROC-AUC = 0,8053. Avaliação OOF em cinco folds, sem validação externa ou separação por clube.
* **Alcance:** A base sustenta análise de ocorrência na temporada. Não contém as datas necessárias para avaliar lesão nos próximos sete dias.

---

### 5. `university_football` — Rotina e Condição Física

* **Pasta do Estudo:** [`📁 university_football/`](university_football/)
* **Documentação Científica Dedicada:** [`📄 university_football/README.md`](university_football/README.md)
* **Notebook Principal:** [`📓 university_football/university_football_pipeline.ipynb`](university_football/university_football_pipeline.ipynb)

#### 🎯 Resumo da Investigação
Cruzamento de 18 atributos de rotina, treinamento, antropometria e condição física com o rótulo `Injury_Next_Season`. O CSV contém 800 registros, sem ausências e com 400 rótulos em cada classe.

#### 📊 Principais Conclusões e Métricas
* **Associações com o Rótulo:** Estresse (ρ = +0,540), sono (ρ = −0,516), equilíbrio (ρ = −0,499) e histórico (ρ = +0,381) apresentam os padrões de interesse para a comparação com os estudos anteriores.
* **Desempenho Interno:** Logística: **ROC-AUC = 0,9900**, **AP = 0,9901**, **Brier = 0,0389**, **Log loss = 0,1298**. RF calibrada: ROC-AUC = 0,9845. Métricas OOF por linha, sem identidade de atleta auditável no arquivo.
* **Limite de Interpretação:** As fontes descrevem o alvo de formas conflitantes, como lesão observada e como classificação de alto risco. Há 715 escores de estresse acima do máximo de 40 da PSS-10 descrita no artigo. A unidade da coluna de sprint também requer esclarecimento. A alta separabilidade do rótulo não resolve essas inconsistências.

---

### 6. `soccernet_gsr` — Movimento por Vídeo

* **Pasta do Estudo:** [`📁 soccernet_gsr/`](soccernet_gsr/)
* **Documentação Científica Dedicada:** [`📄 soccernet_gsr/README.md`](soccernet_gsr/README.md)
* **Notebook Principal:** [`📓 soccernet_gsr/soccernet_gsr_pipeline.ipynb`](soccernet_gsr/soccernet_gsr_pipeline.ipynb)

#### 🎯 Resumo da Investigação
Associação entre frames, posições anotadas no campo e trajetórias locais de jogadores/goleiros, com derivação de velocidade e aceleração em segmentos observados. Foram analisados todos os arquivos de anotações encontrados em treino, validação e teste, versão 1.3; frames de imagem e challenge não integram esta análise.

#### 📊 Principais Conclusões e Métricas
* **Volume:** 164 clips, 123.000 frames, 9 jogos e 3.481 trajetórias locais; 3.454 trajetórias têm movimento utilizável.
* **Cobertura:** Mediana de **70,3% dos frames por trajetória**. Há 1.736.361 anotações com coordenadas e 22.385 sem coordenadas no campo.
* **Controle Cinemático:** Dos 1.625.340 pontos derivados após segmentação e retirada de bordas, 1.610.694 passam pelos controles definidos no estudo.
* **Alcance:** Não há alvo de lesão ou fadiga. As posições são anotações de referência, não previsões de um rastreador executado nesta tarefa; não se reportam AUC clínica ou acurácia de rastreamento.

---

## 🏆 Síntese Consolidada de Resultados: Classificação das Variáveis

A síntese mantém a comparação por domínio esportivo e incorpora os novos cruzamentos. **Os coeficientes descrevem associação com ocorrência de lesão ou com o rótulo de cada base; não medem gravidade, grau ou nível clínico da lesão.** A documentação sintética do SIRP-600 e do Multimodal, a incerteza do alvo University e os diferentes horizontes impedem converter os resultados em um ranking clínico universal.

A classificação a seguir combina **origem da evidência, compatibilidade do desfecho e consistência do resultado**. Ela não estabelece pesos para o modelo teórico:

| Classe | Significado na análise do Scout |
| :--- | :--- |
| **A — Associação observada em coorte específica** | Resultado marginal observado em Shalaj, com lesão de isquiotibiais acompanhada na temporada. Ainda limitado à amostra e sem validação externa. |
| **B — Indício exploratório** | Associação em benchmark sintético ou com rótulo de construção incerta. Útil para formular hipóteses, sem confirmação clínica por este cruzamento. |
| **C — Não confirmada ou dependente da medida** | Associação pequena, divergência entre estudos ou unidade a esclarecer. Não implica que o fator seja irrelevante em toda população. |
| **D — Relação com lesão ainda não avaliada** | Há uma hipótese ou medidas de movimento, mas faltam desfechos correspondentes nos mesmos atletas. |

> [!NOTE]
> `r` indica Pearson; `ρ`, Spearman. Valores positivos representam associação com maior frequência do alvo 1; negativos, com menor frequência. Medidas diferentes e coeficientes de bases distintas são apresentados lado a lado, sem média ou soma. Gini e SHAP descrevem o modelo em que foram calculados e não equivalem a correlação ou peso clínico.

---

### 📊 Tabela Geral de Variáveis e Evidências

| Variável / Fator de Monitoramento | Domínio Esportivo | Associação Estatística e Origem | Direção Observada | Classe / Leitura Atual |
| :--- | :--- | :--- | :--- | :--- |
| **Idade** | Característica individual | Shalaj: ρ = **+0,474**; University: ρ = +0,021; SIRP-600: r = −0,032 | Positiva em Shalaj; próxima de zero nas demais medidas citadas | **A** — Ganha relevância na coorte de isquiotibiais; não deve ser descartada com base no benchmark anterior. |
| **Falha no Teste Nordic** | Função muscular | Shalaj: ρ = **+0,286** | Positiva com falha no teste | **A** — Frequência de lesão de 44,4% entre quem falhou e 18,0% entre quem passou; associação, sem efeito causal demonstrado. |
| **Índice de Massa Corporal (IMC)** | Antropometria | Shalaj: ρ = **+0,250**; University: ρ = +0,036 | Positiva em Shalaj | **A** — Resultado marginal dependente da população; não define uma faixa clínica de IMC. |
| **Histórico de Lesões Anteriores** | Clínico / Recidiva | Shalaj: ρ = **+0,191**; SIRP-600: r = +0,307; University: ρ = +0,381 | Positiva nas análises marginais | **A** — O coeficiente ajustado da logística de Shalaj é negativo; essa instabilidade não deve ser interpretada como proteção. |
| **Sono: Duração e Qualidade** | Rotina / Recuperação | SIRP-600: r = **−0,270**; Multimodal: ρ = **−0,278**; University: ρ = **−0,516** | Inversa nas medidas citadas | **B** — Padrão recorrente em fontes de alcance limitado; horas de sono e qualidade não são a mesma medida. |
| **Nível de Estresse** | Psicofisiologia | SIRP-600: r = **+0,285**; University: ρ = **+0,540** | Positiva | **B** — A inconsistência de escala na University impede interpretar seus valores como PSS-10 ou importar limiares. |
| **Índice de Fadiga** | Carga interna / Bem-estar | Multimodal: ρ = **+0,334** | Positiva | **B** — Associação no demonstrador multimodal; sem confirmação nova em coorte com fadiga e lesão datadas. |
| **Escore de Recuperação / HRV Proxy** | Recuperação | Multimodal: ρ = **−0,323** | Inversa | **B** — Escore proxy, não uma medida bruta de HRV validada nesta análise. |
| **Tempo de Aquecimento / Adesão à Rotina** | Rotina / Treinamento | SIRP-600: r = **−0,273**; University: ρ = **−0,265** | Inversa | **B** — Duração em minutos e adesão binária têm definições distintas. |
| **Carga Total de Treino** | Exposição à carga | Multimodal: ρ = **+0,228** | Positiva | **B** — Resultado na base sintética; não equivale a horas semanais de treino de outra coorte. |
| **Razão de Carga Aguda:Crônica (ACWR)** | Exposição à carga | Multimodal: ρ = **+0,142** | Positiva, de pequena magnitude | **B** — Não estabelece limiares seguros ou perigosos; falta novo teste direto de carga × recuperação com desfecho clínico. |
| **Tempo de Recuperação Cardíaca** | Fisiologia | SIRP-600: r = **+0,222** | Positiva | **B** — Medida distinta de escore de recuperação ou HRV; associação restrita ao benchmark. |
| **Equilíbrio e Testes Funcionais** | Condição física | University: equilíbrio ρ = **−0,499**; força do joelho −0,462; flexibilidade −0,457; agilidade −0,401 | Inversa com o rótulo fornecido | **B** — Não confirma capacidade de prever lesões clinicamente observadas. |
| **Tempo de Reação** | Condição física | University: ρ = **+0,482** | Positiva com o rótulo | **B** — Resultado depende da construção não esclarecida do alvo. |
| **Qualidade da Nutrição** | Rotina | University: ρ = **−0,422** | Inversa com o rótulo | **B** — Indício exploratório, sem confirmação clínica independente nesta expansão. |
| **Assimetria Muscular** | Mecânica funcional | SIRP-600: r = **+0,374**; Shalaj, torque a 60°/s: ρ = **−0,065** | Divergente entre as medidas | **C** — A liderança no benchmark não se transfere para a medida de Shalaj; não sustenta prioridade universal. |
| **Torque Relativo Médio a 240°/s** | Função muscular | Shalaj: ρ = **−0,069** | Associação marginal próxima de zero | **C** — Esta definição não teve associação relevante no recorte; não representa todos os testes de força possíveis. |
| **Horas de Treino por Semana** | Volume de treino | University: ρ = **+0,028** | Associação próxima de zero | **C** — Não confirma associação nesta tabela; não é intercambiável com carga mecânica ou ACWR. |
| **Sprint da University** | Condição física | University: ρ = **−0,476** | Inversa com o rótulo | **C** — Nome da coluna e unidade precisam ser esclarecidos antes de interpretar o sentido fisiológico. |
| **Velocidade, Aceleração e Desaceleração por Vídeo** | Movimento / Exposição mecânica | SoccerNet-GSR: sem alvo clínico | Não avaliada em relação à lesão | **D** — Há cinemática derivada de posições anotadas, sem associação demonstrada com lesão ou fadiga. |
| **Interação Carga × Recuperação** | Exposição e estado do atleta | Cenários do modelo multimodal; sem nova coorte que reúna as medidas e o evento | Hipótese em investigação | **D** — Os novos cruzamentos não fornecem um teste clínico direto dessa interação. |

---

### 🔬 Detalhamento Conceitual e Impacto Estatístico das Variáveis

#### 1. Assimetria Muscular Contralateral (`Muscle_Asymmetry` / `hamstring_asymmetry_60_pct`)
* **O que é:** Diferença entre medidas dos membros; o significado depende do teste, velocidade e fórmula utilizados. Não equivale automaticamente à razão isquiotibiais/quadríceps.
* **Resultado Anterior:** SIRP-600: r = +0,374, com 17,9% de importância de Gini no modelo documentado.
* **Atualização:** A medida de torque a 60°/s de Shalaj apresentou ρ = −0,065. Isso reduz a sustentação de um destaque universal da assimetria, sem provar ausência de associação em outros protocolos. **Classe C.**

#### 2. Escore de Recuperação Autonômica / HRV Proxy (`recovery_score` e média móvel)
* **O que é:** Escore de recuperação apresentado no módulo multimodal; sua interpretação como proxy não substitui uma medição direta de HRV.
* **Resultado:** ρ = −0,323 com o alvo, com destaque nas importâncias internas do modelo anterior.
* **Atualização:** Não houve uma nova coorte que reunisse recuperação e lesão datada para confirmar o achado. As médias móveis do código são por sessões. **Classe B.**

#### 3. Sono: Duração e Qualidade (`Sleep_Hours`, `sleep_quality`, `Sleep_Hours_Per_Night`)
* **O que é:** Horas de sono e qualidade subjetiva são dimensões relacionadas, com unidades distintas.
* **Resultado:** Associação inversa no SIRP-600 (r = −0,270), no Multimodal (ρ = −0,278) e no rótulo University (ρ = −0,516).
* **Atualização:** A repetição do padrão mantém sono como variável candidata, mas a incerteza das fontes não permite declarar proteção causal ou um limite universal de horas. **Classe B.**

#### 4. Índice de Fadiga Sistêmica (`fatigue_index`)
* **O que é:** Escore de fadiga apresentado na tabela multimodal.
* **Resultado:** ρ = +0,334 com o alvo dessa base.
* **Atualização:** O SoccerNet-GSR acrescenta movimento, sem rótulo de fadiga. Ele não valida o escore nem sua relação com lesão. **Classe B.**

#### 5. Histórico de Lesões (`Injury_History`, `previous_hamstring_injury`, `Previous_Injury_Count`)
* **O que é:** Contagem de lesões anteriores em algumas bases e presença/ausência de lesão anterior de isquiotibiais em Shalaj.
* **Resultado:** Associação marginal positiva em SIRP-600 (+0,307), Shalaj (+0,191) e University (+0,381), respeitando o tipo de coeficiente e definição de cada medida.
* **Atualização:** Em Shalaj, 17/41 atletas com histórico e 23/102 sem histórico tiveram lesão. O sinal ajustado negativo na logística não deve ser convertido em efeito protetor. **Classe A para a associação marginal da coorte.**

#### 6. Razão de Carga Aguda:Crônica (ACWR)
* **O que é:** Relação entre médias recentes e mais longas da mesma medida de carga. No Multimodal, janelas de 7/28 sessões; no SoccerMon, de 7/28 dias.
* **Resultado:** ρ = +0,142 no Multimodal. Os cenários de interação pertencem ao modelo dessa base.
* **Atualização:** Os novos cruzamentos não fornecem uma coorte com carga, recuperação e lesão datada para testar diretamente a interação. Não há novo fundamento para faixas clínicas de ACWR. **Classe B; interação com recuperação em classe D.**

#### 7. Aquecimento (`Warmup_Time` / `Warmup_Routine_Adherence`)
* **O que é:** Duração do aquecimento no SIRP-600 e adesão binária à rotina na University.
* **Resultado:** Associação inversa: r = −0,273 e ρ = −0,265, respectivamente.
* **Atualização:** O novo cruzamento repete o sentido com o rótulo, sem validar equivalência entre medidas ou um número mínimo de minutos. **Classe B.**

#### 8. Idade (`age` / `Age`)
* **O que é:** Idade em anos no momento da avaliação inicial.
* **Resultado:** Em Shalaj, ρ = +0,474; médias de 26,1 anos entre lesionados e 22,1 entre não lesionados.
* **Atualização:** A associação próxima de zero em benchmarks anteriores não justifica descartar idade em uma população específica. **Classe A na coorte de Shalaj.**

#### 9. Teste Nordic (`nordic_failed`)
* **O que é:** Indicador de falha no teste conforme os códigos e protocolo do suplemento de Shalaj.
* **Resultado:** ρ = +0,286; 24/54 atletas que falharam tiveram lesão (44,4%), contra 16/89 entre os que passaram (18,0%).
* **Atualização:** Acrescenta um indicador funcional associado a um evento específico de isquiotibiais. A diferença observada não é uma probabilidade individual validada. **Classe A.**

#### 10. Índice de Massa Corporal (`bmi` / `BMI`)
* **O que é:** Relação entre massa corporal e estatura ao quadrado.
* **Resultado:** ρ = +0,250 em Shalaj; associação próxima de zero na University (+0,036).
* **Atualização:** A relevância depende da população e do desfecho; a associação marginal não identifica mecanismo causal ou faixa de risco. **Classe A em Shalaj.**

---

### 🔄 A Dinâmica Não Linear: O Desacoplamento entre Demanda e Recuperação

O projeto mantém a hipótese de que **o significado da carga pode depender da recuperação e da vulnerabilidade do atleta**. A [superfície de interação do módulo Multimodal](multimodal_injury/figures/interacao_sono_acwr.png) mostra cenários calculados por aquele modelo, sem demonstrar que as probabilidades sejam transferíveis a atletas reais de outra coorte.

| Estado de Recuperação | Carga Estável | Aumento de Carga |
| :--- | :--- | :--- |
| **Recuperação adequada** | Cenário conceitual de equilíbrio entre demanda e recuperação. | Investigar a resposta individual à exposição adicional. |
| **Recuperação reduzida** | Investigar déficit de recuperação mesmo sem pico de carga. | Combinação de interesse para testar a hipótese de interação. |

Os novos cruzamentos acrescentam evidência de vulnerabilidade em Shalaj e descrição de movimento no SoccerNet-GSR, mas **não reúnem carga, recuperação e lesão datada nos mesmos atletas para testar diretamente essa interação**. Por isso, a matriz permanece qualitativa, sem percentuais ou condutas clínicas.

O [modelo teórico de risco de lesão](modelo-teorico-risco-lesao/README.md) preserva o recorte de nova lesão de isquiotibiais nos sete dias após uma avaliação anterior à sessão. Nenhuma nova base, no formato analisado, valida integralmente esse recorte.

---

## 🔬 Comparativo Intermódulos e Lições sobre Viabilidade Amostral

### Estudos Anteriores

Os valores a seguir são os resultados documentados nos módulos anteriores; esses modelos não foram retreinados na expansão. Para SoccerMon, foram usados os números da seção de resultados do próprio módulo, atualizando a tabela agregada anterior.

| Dimensão Metodológica | Módulo [`sirp-600`](sirp-600/) | Módulo [`soccermon`](soccermon/) | Módulo [`multimodal_injury`](multimodal_injury/) |
| :--- | :---: | :---: | :---: |
| **Natureza dos Dados** | Tabela estática; natureza sintética declarada | Série diária de futebol feminino | Tabela longitudinal; natureza sintética declarada |
| **Volume Amostral** | 600 atletas | 50 atletas; 36.550 atleta-dias | 156 atletas; 15.420 sessões |
| **Alvo** | Rótulo estático de risco | Registro de lesão de coxa nos próximos 7 dias | `injury_occurred == 2` na sessão |
| **Validação** | Divisão estratificada 80/20 | Corte temporal em 01/06/2021 | Divisão aleatória estratificada 80/20 de sessões |
| **Prevalência Positiva no Teste** | 31,7% | 0,26% | 15,01% |
| **ROC-AUC** | **0,9021** | **0,8411** | **0,8806** |
| **Average Precision (AP)** | **0,8238** | **0,0514** | **0,5494** |
| **Brier Score** | **0,1149** | **0,0027** | **0,0932** |
| **Log Loss** | **0,3578** | **0,0177** | **0,3009** |
| **Leitura Atual** | Exploração de fatores e modelo em benchmark | AP baixa e desfechos escassos; Brier baixo deve ser lido junto à prevalência | Exploração de carga e recuperação; sem validação prospectiva por essa divisão |

### Novos Cruzamentos Tabulares

A avaliação dos novos módulos usa cinco folds externos. Cada atleta de Shalaj ou linha da University recebe uma previsão OOF, de modelo sem treino naquela observação. A RF é calibrada em três folds internos. A referência de prevalência usa somente o treino de cada fold.

| Dataset | Modelo | ROC-AUC OOF | AP | Brier | Log Loss |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Shalaj** | Referência de prevalência | 0,4942 | 0,2774 | 0,2015 | 0,5927 |
| **Shalaj** | Regressão logística | 0,8053 | 0,5442 | 0,1617 | 0,4855 |
| **Shalaj** | RF calibrada | **0,8235** | **0,6266** | **0,1512** | **0,4600** |
| **University** | Referência de prevalência | 0,5000 | 0,5000 | 0,2500 | 0,6931 |
| **University** | Regressão logística | **0,9900** | **0,9901** | **0,0389** | **0,1298** |
| **University** | RF calibrada | 0,9845 | 0,9843 | 0,0473 | 0,1634 |

* **Shalaj:** 143 atletas, 40 positivos (27,97%). O intervalo bootstrap de 95% da AUC da RF é **0,754–0,890**, condicional às previsões OOF; não inclui toda a incerteza do treino nem corrige a dependência por clube. A amostra não oferece identificadores de clube para esse teste.
* **University:** 800 linhas e 400 positivos (50%). O intervalo condicional da AUC logística é **0,985–0,994**. A semântica clínica do alvo permanece incerta; a classificação da tabela não confirma previsão de lesões reais.
* **Comparabilidade dos Modelos:** Os estudos anteriores usam RF de 500 árvores; os novos usam RF de 200 árvores, com outras configurações de profundidade/folhas e avaliação. A logística foi uma comparação adicional. Não se trata de um experimento que varia apenas o dataset.
* **Leitura das Métricas:** ROC-AUC e AP aumentam com melhor discriminação; Brier e Log loss diminuem com melhor qualidade probabilística no alvo. AP é Average Precision, chamada de PR-AUC nos estudos anteriores, e não precisão de um limiar. Um Brier baixo não comprova calibração ou utilidade clínica isoladamente.

### SoccerNet-GSR: Cinemática sem Avaliação Preditiva de Lesão

| Dimensão | SoccerNet-GSR |
| :--- | :--- |
| **Material Inspecionado** | Anotações v1.3 de 164 clips |
| **Unidade** | Trajetória local por clip |
| **Volume Processado** | 123.000 frames; 3.481 trajetórias, 3.454 com movimento utilizável |
| **Cobertura / Qualidade** | Cobertura mediana de 70,3%; 1.610.694 pontos cinemáticos aceitos de 1.625.340 derivados |
| **Resultado Disponível** | Cinemática descritiva de posições anotadas |
| **ROC-AUC / AP / Brier / Log Loss** | Não se aplicam à análise executada, sem alvo de lesão/fadiga |
| **Limite** | Sem rastreador automático executado ou referência GPS de validação |

---

## ⚠️ Ressalvas Científicas e Metodológicas

> [!WARNING]
> ### 📌 Nota sobre Uso dos Modelos e Interpretação Estatística
>
> * **Natureza Exploratória e Analítica:** Os estudos investigam padrões estatísticos e hipóteses. O SIRP-600 e o Multimodal têm natureza sintética declarada; a origem e construção do rótulo University permanecem conflitantes nas fontes.
> * **Correlação ≠ Causalidade:** Correlações, importâncias de árvores e SHAP não demonstram causa, não são pesos universais e não permitem comparar diretamente protocolos distintos. A classificação das variáveis se refere às evidências deste projeto, sem medir gravidade de lesão.
> * **Limites Temporais:** Dias, sessões, avaliações de pré-temporada e clips de 30 segundos não são unidades intercambiáveis. Coxa, isquiotibiais, lesão genérica e alto risco não são o mesmo evento.
> * **Qualidade dos Estudos Anteriores:** No SoccerMon, o preenchimento com observações posteriores pode trazer informação futura e as métricas cinemáticas são derivadas de duração/sRPE. No Multimodal, o pré-processamento anterior à divisão e o teste aleatório por sessões limitam a interpretação prospectiva. Os gráficos de interação são cenários dos modelos; o SHAP usa árvores separadas da composição calibrada final.
> * **Ausência de Integração de Participantes entre Bases:** Os cruzamentos acontecem dentro de cada fonte. A síntese compara achados; não junta sono de uma coorte com lesões ou trajetórias de outra como se fossem do mesmo atleta.
> * **Não Aplicabilidade Clínica Direta:** Nenhum dos resultados desta expansão valida uma ferramenta individual para nova lesão de isquiotibiais em sete dias. Uso clínico requer avaliação prospectiva e externa compatível com o evento e o instante de avaliação.
