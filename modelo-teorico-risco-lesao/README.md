# Modelo Teórico de Risco de Lesão em Atletas de Futebol

> **Status: especificação conceitual; escopo inicial definido.**
>
> Este documento organiza uma hipótese de risco. Não apresenta uma estimativa individual de probabilidade nem uma recomendação clínica ou de manejo esportivo.

## Objetivo

Especificar como fatores estruturais do atleta, estado de recuperação e exposição à carga poderiam ser combinados numa hipótese de risco. A ideia central é que a carga não deve ser interpretada isoladamente: sua relação com o risco pode depender da recuperação e da vulnerabilidade do atleta.

Esta pasta documenta somente a hipótese e seu escopo. Os pesos numéricos não estão especificados e não há software de predição.

## Escopo inicial registrado

| Decisão | Definição conceitual |
|---|---|
| **População** | Atletas de futebol. Este escopo não presume um sexo, idade ou nível competitivo específico. |
| **Unidade de análise** | Um atleta em cada sessão esportiva monitorada. |
| **Momento da avaliação** | Antes da sessão, usando somente informações disponíveis até esse momento. |
| **Evento de interesse** | Nova lesão clinicamente confirmada dos músculos isquiotibiais (posterior da coxa), com data de início identificável. |
| **Horizonte** | Os sete dias após a avaliação: \((t, t + 7\text{ dias}]\). |
| **Saída conceitual** | Um sinal qualitativo de risco relativo, sem percentuais ou faixas clínicas. |

Os exemplos abaixo são **cenários conceituais**, não observações de atletas.

### Delimitação do evento

“Posterior da coxa” refere-se aqui a uma lesão dos isquiotibiais, e não a qualquer desconforto ou lesão genérica na região da coxa. Para identificar o evento de interesse, um registro clínico precisaria permitir reconhecer o diagnóstico, a região anatômica, a data de início e se se trata de um episódio novo.

O SoccerMon possui uma janela futura de sete dias, mas seu rótulo anatômico é “coxa” e não distingue os isquiotibiais de outras estruturas da região. Por isso, a definição de evento nesta hipótese é mais específica do que esse rótulo disponível.

## Hipótese fisiológica

O risco de lesão pode refletir o desacoplamento entre:

1. **Vulnerabilidade estrutural:** características relativamente persistentes, como assimetria muscular e histórico de lesões.
2. **Déficit de recuperação:** estado mais dinâmico, observado por indicadores de recuperação, sono, fadiga e estresse.
3. **Exposição à carga:** demanda recente, que pode incluir carga total e razão entre carga aguda e crônica (ACWR).

Em termos conceituais, uma exposição elevada poderia ser mais preocupante quando combinada com recuperação baixa ou vulnerabilidade estrutural elevada. Isso é uma hipótese, não uma relação causal demonstrada.

## Variáveis candidatas

| Eixo | Variáveis candidatas | Papel hipotético |
|---|---|---|---|
| Vulnerabilidade estrutural | Assimetria muscular; histórico de lesões | Contextualizar a suscetibilidade basal do atleta. |
| Recuperação | Escore de recuperação/HRV proxy e média móvel; qualidade e duração do sono | Representar o estado de recuperação recente. |
| Fadiga e estresse | Fadiga sistêmica; estresse psicofisiológico | Complementar a descrição do estado de recuperação. |
| Exposição à carga | Carga total; ACWR | Representar demanda recente em relação à carga crônica. |
| Exposição mecânica complementar | Desacelerações severas e outras métricas GPS | Descrever aspectos específicos da demanda mecânica. |

As variáveis são candidatas conceituais. Correlações, importâncias de Gini e valores SHAP não determinam, por si só, pesos ou uma forma matemática para a hipótese.

## Relação conceitual proposta

Esta especificação não fixa uma equação nem uma arquitetura matemática. Ela registra relações a considerar como hipótese:

1. Vulnerabilidade estrutural, como histórico de lesões e assimetria, pode contextualizar a suscetibilidade do atleta.
2. Recuperação, sono, fadiga e estresse descrevem um estado que pode variar entre sessões.
3. Carga recente pode estar associada ao risco, mas seu significado pode depender do estado de recuperação.
4. A combinação de carga elevada e recuperação baixa é uma interação de interesse, não uma regra ou relação causal estabelecida.

Essas proposições não especificam pesos, limiares, funções, interações exclusivas ou percentuais. Uma formulação matemática futura poderá expressar essas ideias ou ser modificada conforme a pergunta de pesquisa e os dados disponíveis.

## Cenários conceituais

| Cenário ilustrativo | Vulnerabilidade | Recuperação | Carga | Leitura que a hipótese propõe |
|---|---|---|---|---|
| A | Baixa | Adequada | Estável | Menor sinal relativo que nos cenários com múltiplos fatores desfavoráveis. |
| B | Baixa | Adequada | Elevada | A carga merece análise, mas a hipótese não conclui que ela, sozinha, implique risco alto. |
| C | Baixa | Baixa | Elevada | Combinação que a interação recuperação × carga trataria como mais preocupante para investigação. |
| D | Elevada | Baixa | Moderada ou elevada | Vulnerabilidade e recuperação baixa poderiam se combinar; a magnitude do efeito é desconhecida. |

“Menor” e “mais preocupante” são comparações qualitativas dentro da hipótese. Não são categorias clínicas nem faixas percentuais.

## Regras de interpretação propostas

1. Não interpretar ACWR, HRV, sono ou assimetria como diagnóstico ou causa isolada.
2. Interpretar o escore como qualitativo e relativo, não como uma probabilidade individual ou uma medida diretamente transferível entre populações.
3. Não tratar dados ausentes como valores seguros ou de risco. Se uma variável não estiver disponível, o sistema futuro deverá indicar a ausência e a cobertura dos dados.
4. Manter unidades, janelas temporais e definições explícitas para cada variável; não misturar medidas diárias e por sessão sem uma regra temporal previamente estabelecida.
5. Não tratar os cenários conceituais desta documentação como limites operacionais.

## Informações necessárias para caracterizar o evento

Para que o desfecho corresponda à definição desta hipótese, o registro clínico do evento deve permitir identificar:

1. **Diagnóstico e localização:** confirmação clínica de lesão dos isquiotibiais/posterior da coxa, distinguindo-a de outras lesões da coxa.
2. **Data de início:** data suficiente para determinar se o evento ocorreu dentro da janela definida.
3. **Episódio novo:** identificador ou informação clínica que diferencie uma nova lesão do acompanhamento de uma lesão já existente.
4. **Lateralidade:** lado afetado, quando disponível.
5. **Momento das variáveis:** horários ou datas que permitam separar informações disponíveis antes da sessão das medidas obtidas durante ou depois dela.

## Conclusão

O produto desta etapa é **uma especificação conceitual documentada** para atletas de futebol: população, unidade de análise, momento da avaliação, evento de interesse, horizonte, eixos de variáveis e limitações. Não é uma calculadora de risco ou ferramenta de decisão.
