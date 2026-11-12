# Resultados executados - University Football

**Classificação do rótulo publicado; não validação de lesão clínica.**

Linhas: 800; rótulo 1: 400; prevalência do rótulo: 50.00%.

| Modelo | ROC-AUC OOF | IC 95% condicional | AP | Brier | Log loss |
|---|---:|---|---:|---:|---:|
| prevalencia | 0.5000 | 0.500–0.500 | 0.5000 | 0.2500 | 0.6931 |
| logistica | 0.9900 | 0.985–0.994 | 0.9901 | 0.0389 | 0.1298 |
| rf_sigmoid | 0.9845 | 0.978–0.990 | 0.9843 | 0.0473 | 0.1634 |

## Correlações exploratórias

| Variável | Spearman |
|---|---:|
| Sleep_Hours_Per_Night | -0.516 |
| Balance_Test_Score | -0.499 |
| Sprint_Speed_10m_s | -0.476 |
| Knee_Strength_Score | -0.462 |
| Hamstring_Flexibility | -0.457 |
| Nutrition_Quality_Score | -0.422 |
| Agility_Score | -0.401 |
| Warmup_Routine_Adherence | -0.265 |
| Height_cm | -0.053 |
| Matches_Played_Past_Season | -0.045 |
| Weight_kg | +0.017 |
| Age | +0.021 |
| Training_Hours_Per_Week | +0.028 |
| BMI | +0.036 |
| Previous_Injury_Count | +0.381 |
| Reaction_Time_ms | +0.482 |
| Stress_Level_Score | +0.540 |
