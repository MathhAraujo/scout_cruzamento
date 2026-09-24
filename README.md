# 🏃‍♂️ Estudo Exploratório de Correlações e Indicadores de Risco de Lesão Esportiva (SIRP-600)

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.9.1-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-v0.52.0-red.svg)](https://shap.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-SIRP--600-brightgreen.svg)](#sobre-o-dataset-sirp-600)
[![Status](https://img.shields.io/badge/Status-Estudo%20Analítico-lightgrey.svg)](#ressalvas-e-limitações-metodológicas)

Este repositório contém uma investigação analítica detalhada sobre as relações estatísticas entre **variáveis comportamentais de rotina, biométricas e mecânicas funcionais** e a ocorrência de lesões musculoesqueléticas, utilizando a base de dados **SIRP-600** (*Sports Injury Risk Prediction*).

> [!WARNING]
> ### ⚠️ Nota Fundamental de Escopo e Limitação Científica
> **Este estudo possui finalidade estritamente exploratória e analítica.**
> 
> As análises, métricas probabilísticas e pesos obtidos refletem **correlações estatísticas e associações de padrões no conjunto de dados analisado**. 
> **ESTE MODELO NÃO DEVE SER UTILIZADO COMO SISTEMA PREDITIVO EM PRODUÇÃO OU PARA TOMADA DE DECISÃO CLÍNICA/MÉDICA.** 
> Correlação estatística **não implica relação de causalidade direta**. A transição de um estudo observacional para uma ferramenta de predição diagnóstica exigiria ensaios clínicos prospectivos, validação em coortes longitudinais reais e acompanhamento médico individualizado.

---

## 📋 Sumário
1. [Sobre o Dataset SIRP-600](#sobre-o-dataset-sirp-600)
2. [Cruzamento de Dados e Análise de Correlações (EDA)](#cruzamento-de-dados-e-análise-de-correlações-eda)
3. [Distribuição de Densidade por Histórico de Lesão](#distribuição-de-densidade-por-histórico-de-lesão)
4. [Metodologia de Modelagem e Avaliação Probabilística](#metodologia-de-modelagem-e-avaliação-probabilística)
5. [O que Cada Métrica Significa e Resultados Obtidos](#o-que-cada-métrica-significa-e-resultados-obtidos)
6. [Pesos das Variáveis e Interpretabilidade com SHAP](#pesos-das-variáveis-e-interpretabilidade-com-shap)
7. [Estratificação Teórica de Risco](#estratificação-teórica-de-risco)
8. [Ressalvas e Limitações Metodológicas](#ressalvas-e-limitações-metodológicas)
9. [Estrutura do Repositório e Como Executar](#estrutura-do-repositório-e-como-executar)

---

## 🗃️ Sobre o Dataset SIRP-600

O **SIRP-600** é composto por **600 amostras individuais de atletas** (335 mulheres e 265 homens), com faixa etária entre 18 e 40 anos, sem valores ausentes (*ML-Ready*), registrando 15 variáveis independentes e uma variável alvo de desfecho:

* **Variável Alvo (`Injury_Risk`):** 
  * `0`: Não lesionado / Controle saudável (68.5% — 411 atletas).
  * `1`: Lesão ocorrida (31.5% — 189 atletas).
* **Dimensões Físicas e Demográficas:** Idade, Sexo, Estatura (cm), Peso Corporal (kg) e Índice de Massa Corporal (IMC).
* **Rotina e Treinamento:** Frequência Semanal (dias/sem), Duração da Sessão (min), Tempo de Aquecimento (min) e Intensidade Subjetiva de Esforço (1-10).
* **Fisiologia e Recuperação:** Horas Médias de Sono (h/dia), Tempo de Recuperação da Frequência Cardíaca (s), Escore de Flexibilidade (0-100) e Assimetria Muscular Contralateral (%).
* **Fatores Clínicos e Emocionais:** Histórico de Lesões Anteriores (qtd) e Nível de Estresse Psicológico (1-10).

---

## 📈 Cruzamento de Dados e Análise de Correlações (EDA)

Para avaliar como cada atributo se relaciona com o desfecho de lesão, foram calculados os coeficientes de **correlação linear de Pearson ($r$)** e **monotônica de Spearman ($\rho$)**.

![Matriz de Correlação e Ranking Individual](figures/matriz_correlacao_e_ranking.png)

### Principais Constatações da Análise de Correlação:

1. **Fatores de Maior Risco Associado (Correlação Positiva):**
   * **Assimetria Muscular ($r = +0.374$ / $\rho = +0.342$):** É a variável isolada com maior grau de correlação linear com a lesão. Desequilíbrios superiores a 8% criam sobrecargas mecânicas compensatórias.
   * **Histórico de Lesões Anteriores ($r = +0.307$ / $\rho = +0.294$):** Forte indicador de fragilidade residual do tecido ou instabilidade articular pregressa.
   * **Nível de Estresse ($r = +0.285$ / $\rho = +0.286$):** Fator sistêmico expressivo, associado a aumento no tônus muscular basal e perda de foco atencional.
   * **Recuperação Cardíaca ($r = +0.222$ / $\rho = +0.230$):** Recuperações mais lentas (tempo elevado em segundos) indicam menor eficiência parassimpática e estresse autonômico acumulado.

2. **Fatores Protetores Chave (Correlação Negativa):**
   * **Tempo de Aquecimento ($r = -0.273$ / $\rho = -0.273$):** Correlação inversa sólida. Sessões com aquecimento adequado ($> 15\text{ min}$) estão associadas a menor prevalência de lesões agudas.
   * **Horas de Sono ($r = -0.270$ / $\rho = -0.264$):** Principal fator biológico protetor. Níveis médios superiores a 7.5 horas diárias correlacionam-se com recuperação tecidual eficiente.

3. **Fatores com Fraca Correlação Linear:**
   * **Idade ($r = -0.032$):** A faixa etária limitada da base (18-40 anos) não evidenciou linearidade direta com o risco.
   * **Frequência de Treino ($r = +0.040$):** O número bruto de dias de treino isolado não explica lesões; é a interação com descanso e intensidade que dita o risco.

---

## 🔬 Distribuição de Densidade por Histórico de Lesão

Ao inspecionar as distribuições de probabilidade contínua (KDE) entre o grupo controle (saudável) e o grupo lesionado, o contraste é visível:

![Distribuições de Densidade por Risco de Lesão](figures/distribuicoes_kde_fatores_risco.png)

* **Horas de Sono:** A curva do grupo lesionado possui pico concentrado abaixo de 6.0 horas, enquanto a média do grupo saudável situa-se em 7.5 horas.
* **Assimetria Muscular:** Atletas lesionados apresentam cauda longa estendendo-se até 18-20% de desequilíbrio contralateral, contra menos de 5% no grupo saudável.
* **Tempo de Aquecimento:** A densidade de atletas com lesão concentra-se massivamente na faixa de 0 a 10 minutos de aquecimento.

---

## ⚙️ Metodologia de Modelagem e Avaliação Probabilística

Para capturar as interações não lineares entre as variáveis, foi implementado um modelo baseado em ensemble:

* **Algoritmo Base:** `RandomForestClassifier` com **500 árvores de decisão**, controle de profundidade via folhas (`min_samples_split=4`, `min_samples_leaf=2`) e paralelização total (`n_jobs=-1`, 16 threads lógicas).
* **Calibração de Probabilidade:** Envolvimento com **`CalibratedClassifierCV(estimator=rf, method='sigmoid', cv=5, n_jobs=-1)`** (Platt Scaling em 5 folds).
  * *Por que calibrar?* Árvores de decisão puras tendem a empurrar probabilidades para perto de 0 ou 1 devido à pureza das folhas. O método Sigmoid ajusta uma curva logística sobre as saídas out-of-fold para garantir que uma probabilidade estimada de 40% signifique que, historicamente, 40 em cada 100 atletas sob aquelas condições sofreram lesão.
* **Validação:** Divisão estratificada (80% treino / 20% teste), mantendo a proporção de 31.5% de positivos em ambos os subconjuntos.

---

## 📊 O que Cada Métrica Significa e Resultados Obtidos

Em problemas de saúde com classes desbalanceadas, a **Acurácia simples é uma métrica enganosa** (um classificador que dissesse "sempre saudável" teria 68.5% de acurácia, mas seria perigoso). Por isso, foram avaliadas métricas contínuas, de ranking e de calibração:

![Painel de Métricas e Curva de Calibração](figures/metricas_avaliacao_calibracao.png)

### Tabela Consolidada de Resultados ($N = 120$ atletas no conjunto de teste):

| Métrica | Valor Obtido | Significado Teórico e Prático |
| :--- | :---: | :--- |
| **ROC-AUC Score** | **`0.9021`** | **Capacidade de Discriminação Global:** Mede a probabilidade de o modelo ranquear um atleta aleatório de risco acima de um atleta saudável. Valores $> 0.90$ indicam capacidade discriminativa excepcional. |
| **Average Precision (PR-AUC)** | **`0.8238`** | **Precisão Média Ponderada pelo Recall:** Área sob a curva Precision-Recall. Foca estritamente na classe minoritária de risco (prevalência base: 31.7%). O valor de 0.8238 demonstra alta precisão ao detectar atletas suscetíveis. |
| **Brier Score Loss** | **`0.1149`** | **Calibração Quadrática das Probabilidades:** Mede o erro quadrático médio entre a probabilidade prevista ($P_i$) e o resultado real binário ($y_i \in \{0, 1\}$). Varia de 0 (perfeito) a 1. Valores $< 0.15$ indicam probabilidades muito bem calibradas. |
| **Log Loss (Cross-Entropy)** | **`0.3578`** | **Penalidade de Incerteza:** Penaliza previsões que foram confiantes e erradas. Quanto menor o valor, menor a incerteza estatística gerada pelo ensemble. |
| **F1-Score (Limiar 0.32)** | **`0.7674`** | **Média Harmônica entre Precisão e Recall no Limiar Calibrado:** No ponto de corte ótimo de 32%, o modelo equilibra sensibilidade e precisão clínica. |
| **F1-Score (Limiar 0.50)** | **`0.7397`** | **Média Harmônica no Corte Tradicional:** Demonstra consistência mesmo sem ajuste fino de threshold. |
| **Acurácia Global** | **`83.33%`** | Taxa global de acertos (100 acertos em 120 atletas). |

### Matriz de Confusão no Limiar Calibrado (Corte em $32\%$):

$$\begin{array}{c|cc}
& \textbf{Previsto: Sem Lesão} & \textbf{Previsto: Com Lesão} \\
\hline
\textbf{Real: Sem Lesão (82)} & 67\text{ (81.7\%)} & 15\text{ (18.3\% - Falsos Alarmes)} \\
\textbf{Real: Com Lesão (38)} & 5\text{ (13.2\% - Lesões não vistas)} & 33\text{ (86.8\% - Sensibilidade)} \\
\end{array}$$

* **Sensibilidade (Recall) para Lesão:** **`86.84%`** (33 de 38 lesões reais identificadas).
* **Especificidade para Controle:** **`81.71%`** (67 de 82 controles saudáveis confirmados).

---

## 🧠 Pesos das Variáveis e Interpretabilidade com SHAP

Para compreender quais variáveis mais pesam na formação das estimativas, foram combinadas duas abordagens: a **Importância de Gini** das árvores e os **valores SHAP** baseados na teoria dos jogos cooperativos.

### 1. Importância Média de Gini (Random Forest)
Mede a redução média da impureza em todos os nós das 2.500 árvores treinadas na validação cruzada:

![Importância das Variáveis no Modelo](figures/feature_importance_rf.png)

1. **Assimetria Muscular:** Responsável por **17.9%** da capacidade explicativa do modelo.
2. **Recuperação Cardíaca:** **9.7%**.
3. **Horas de Sono:** **9.5%**.
4. **Nível de Estresse:** **9.2%**.
5. **Tempo de Aquecimento:** **9.1%**.
6. **Histórico de Lesões:** **7.2%**.

### 2. SHAP Beeswarm Plot (Direção do Impacto)
O gráfico *Beeswarm* revela não apenas o peso, mas a **direção do impacto** de cada variável:
* **Pontos Azuis:** Baixo valor do atributo.
* **Pontos Vermelhos:** Alto valor do atributo.
* **Eixo Horizontal:** Impacto no log-odds / risco (valores positivos aumentam o risco, negativos diminuem).

![SHAP Beeswarm Plot](figures/shap_summary_beeswarm.png)

![SHAP Bar Plot de Impacto Médio](figures/shap_summary_bar.png)

#### Interpretação das Interações SHAP:
* **Assimetria Muscular:** Pontos vermelhos (assimetria alta) estão maciçamente deslocados para a direita ($\text{SHAP} > 0$), elevando expressivamente o risco.
* **Horas de Sono:** Pontos azuis (sono insuficiente) situam-se fortemente à direita ($\text{SHAP} > 0$), comprovando que dormir pouco é um indutor estatístico de risco, enquanto pontos vermelores (sono prolongado) atuam como freio protetor à esquerda.
* **Tempo de Aquecimento:** Pontos azuis (aquecimento quase nulo) empurram as predições para a zona de lesão.
* **Acoplamento Sono $\times$ Estresse:** Observa-se um efeito sinérgico nos dados onde atletas sob alto estresse e sono degradado sofrem uma amplificação multiplicativa no risco acumulado.

---

## 🩺 Estratificação Teórica de Risco

Para fins puramente ilustrativos da variação contínua gerada pelo modelo (0% a 100%), foram avaliados três perfis teóricos contrastantes através da função `avaliar_atleta()`:

| Atleta Teórico | Perfil de Rotina e Fisiologia | Risco Calculado | Faixa | Fatores Relevantes Detectados |
| :--- | :--- | :---: | :---: | :--- |
| **Perfil A (Saudável)** | 8.5h sono, 25min aquecimento, 2.3% assimetria, 42s recuperação, estresse 3/10, 0 lesões prévias | **`0.8%`** | 🟢 Baixo Risco | Nenhum desvio crítico detectado |
| **Perfil B (Intermediário)**| 6.5h sono, 12min aquecimento, 5.5% assimetria, 85s recuperação, estresse 5/10, 1 lesão prévia | **`57.1%`** | 🟡 Moderado | Valores limítrofes de sono e recuperação |
| **Perfil C (Crítico/Fadiga)**| 4.8h sono, 5min aquecimento, 14.8% assimetria, 118s recuperação, estresse 9/10, 3 lesões prévias | **`99.7%`** | 🔴 Alto Risco | Sono $< 6.5\text{h}$, aquecimento $< 10\text{min}$, assimetria $> 8\%$, estresse $\ge 7$, histórico de recidiva |

---

## 🚫 Ressalvas e Limitações Metodológicas

> [!CAUTION]
> ### Por que este modelo NÃO deve ser considerado pronto para predição no mundo real?
> 
> 1. **Natureza Sintética e Estática do Dataset:** O dataset SIRP-600 é uma base estática projetada para benchmarks acadêmicos e estudos de interpretabilidade. Ele não captura a dinâmica temporal longitudinal (ex: oscilação diária de carga microciclo a microciclo).
> 2. **Ausência de Marcadores Biomecânicos Diretos:** Variáveis como cinemática de corrida, ângulos de valgo dinâmico de joelho e eletromiografia não estão presentes.
> 3. **Correlação $\neq$ Causalidade:** Ter assimetria muscular correlacionada com lesão não prova que a assimetria causou a lesão; ambas podem ser consequências de uma terceira variável não mensurada (confundidor).
> 4. **Necessidade de Ensaios Prospectivos:** Qualquer algoritmo que pretenda orientar afastamento ou prescrição de treino precisa ser validado prospectivamente em ambiente cego (*blinded prospective clinical trial*), comprovando eficácia médica e ausência de viés.
> 
> **Conclusão:** O presente trabalho serve como um **diagnóstico exploratório de correlações e um demonstrador técnico de calibração probabilística e explicabilidade SHAP**, não como dispositivo de predição médica.

---

## 💻 Estrutura do Repositório e Como Executar

```
📁 Scout/
├── 📄 README.md                            # Documentação científica e analítica do projeto
├── 📓 analise_risco_lesao_sirp600.ipynb     # Notebook Jupyter completo e documentado (7 células)
├── 📊 sirp600.csv                          # Dataset ML-Ready SIRP-600 (600 atletas)
├── 🐍 build_notebook.py                    # Script de automação e geração do notebook
└── 📁 figures/                             # Imagens e gráficos exportados em 300 DPI
    ├── 🖼️ matriz_correlacao_e_ranking.png
    ├── 🖼️ distribuicoes_kde_fatores_risco.png
    ├── 🖼️ metricas_avaliacao_calibracao.png
    ├── 🖼️ feature_importance_rf.png
    ├── 🖼️ shap_summary_beeswarm.png
    └── 🖼️ shap_summary_bar.png
```

### Como Executar Localmente:

1. **Clone ou navegue até o repositório:**
   ```bash
   cd Scout
   ```

2. **Instale as dependências analíticas:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn shap nbformat
   ```

3. **Execute o Jupyter Notebook:**
   ```bash
   jupyter notebook analise_risco_lesao_sirp600.ipynb
   ```
   *(Ou abra diretamente no VS Code / Cursor selecionando o kernel Python 3).*

Todas as 7 células são independentes, utilizam `n_jobs=-1` para aceleração em todos os núcleos da CPU e salvam os gráficos atualizados automaticamente na pasta `figures/`.

---

**Autor / Cientista de Dados:** Projeto Scout — Sports Analytics & Machine Learning Research.
