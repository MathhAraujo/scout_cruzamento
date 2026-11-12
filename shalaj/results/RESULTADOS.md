# Resultados executados - Shalaj

Atletas: 143; com lesão: 40; prevalência: 27.97%.

| Modelo | ROC-AUC OOF | IC 95% condicional | AP | Brier | Log loss |
|---|---:|---|---:|---:|---:|
| prevalencia | 0.4942 | 0.402–0.582 | 0.2774 | 0.2015 | 0.5927 |
| logistica | 0.8053 | 0.727–0.876 | 0.5442 | 0.1617 | 0.4855 |
| rf_sigmoid | 0.8235 | 0.754–0.890 | 0.6266 | 0.1512 | 0.4600 |

Resultados internos exploratórios. Não são previsões validadas por clube nem risco em sete dias.

## Associações observadas

| Variável | Spearman | Média sem lesão | Média com lesão |
|---|---:|---:|---:|
| age | +0.474 | 22.146 | 26.050 |
| bmi | +0.250 | 22.626 | 23.629 |
| previous_hamstring_injury | +0.191 | 0.233 | 0.425 |
| nordic_failed | +0.286 | 0.291 | 0.600 |
| hamstring_relative_240_mean | -0.069 | 1.357 | 1.332 |
| hamstring_asymmetry_60_pct | -0.065 | 9.873 | 8.164 |
