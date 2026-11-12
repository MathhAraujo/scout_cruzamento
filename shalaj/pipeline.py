"""Shalaj: análise de pré-temporada, sem usar variáveis posteriores ao desfecho."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import requests
from scipy.stats import fisher_exact, spearmanr
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
URL = "https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0241127.s001"
ARTICLE = "https://doi.org/10.1371/journal.pone.0241127"
SEED = 42
FEATURES = ["age", "bmi", "previous_hamstring_injury", "nordic_failed",
            "hamstring_relative_240_mean", "hamstring_asymmetry_60_pct"]


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def download():
    path = ROOT / "data/raw/journal.pone.0241127.s001.sav"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        r = requests.get(URL, timeout=90)
        r.raise_for_status()
        if not r.content.startswith((b"$FL2", b"$FL3")):
            raise ValueError("A resposta não é um arquivo SPSS SAV.")
        path.write_bytes(r.content)
    return path


def prepare(raw, meta):
    assert raw.player_id.is_unique and raw.player_id.notna().all()
    assert set(raw.total_hamstring_injuries.unique()) == {1, 2}
    assert meta.variable_value_labels["total_hamstring_injuries"] == {1.0: "yes", 2.0: "no"}
    assert meta.variable_value_labels["nordic_hamstring_test"] == {1.0: "yes", 2.0: "no"}
    assert set(raw.previous_hamstring_injury.unique()) <= {0, 1}
    df = raw.copy()
    df["target_hamstring_season"] = raw.total_hamstring_injuries.eq(1).astype(int)
    df["nordic_failed"] = raw.nordic_hamstring_test.eq(2).astype(int)
    # O erro de grafia hamsrings faz parte do arquivo de origem.
    df["hamstring_relative_240_mean"] = (
        raw.hamstrings_concentric_240_dom + raw.hamsrings_concentric_240_nondom
    ) / (2 * raw.body_mass)
    a, b = raw.hamstrings_concentric_60_dom, raw.hamstrings_concentric_60_nondom
    assert (np.maximum(a, b) > 0).all()
    df["hamstring_asymmetry_60_pct"] = 100 * (a - b).abs() / np.maximum(a, b)
    assert np.isfinite(df[FEATURES].to_numpy()).all()
    # Allowlist fechada: exposição e taxas calculadas na temporada nunca entram em X.
    assert not any("rate" in c.lower() or "exposure" in c.lower() for c in FEATURES)
    return df


def metrics(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p))}


def evaluate(df):
    X, y = df[FEATURES], df.target_hamstring_season
    models = {
        "prevalencia": DummyClassifier(strategy="prior"),
        "logistica": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        "rf_sigmoid": CalibratedClassifierCV(
            make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
                n_estimators=200, max_depth=3, min_samples_leaf=6, n_jobs=-1, random_state=SEED)),
            method="sigmoid", cv=3, n_jobs=1),
    }
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    predictions = pd.DataFrame({"player_id": df.player_id.astype(int), "target": y,
                                "fold": -1})
    result, fold_results = {}, []
    coefficients = []
    for name, model in models.items():
        p = np.full(len(df), np.nan)
        for fold, (train, test) in enumerate(folds):
            assert not set(train) & set(test)
            fitted = clone(model).fit(X.iloc[train], y.iloc[train])
            p[test] = fitted.predict_proba(X.iloc[test])[:, 1]
            predictions.loc[test, "fold"] = fold
            fold_results.append({"model": name, "fold": fold, **metrics(y.iloc[test], p[test])})
            if name == "logistica":
                coefficients.append(fitted[-1].coef_[0])
        assert np.isfinite(p).all()
        predictions[name] = p
        result[name] = metrics(y, p)
        # Intervalo condicional às previsões OOF, não inclui toda incerteza de treino.
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(1500):
            ix = rng.integers(0, len(y), len(y))
            if y.iloc[ix].nunique() == 2:
                boot.append(roc_auc_score(y.iloc[ix], p[ix]))
        result[name]["auc_bootstrap_95_conditional"] = np.quantile(boot, [.025, .975]).tolist()
    coef = pd.DataFrame({"feature": FEATURES, "coefficient_mean": np.mean(coefficients, axis=0),
                         "coefficient_sd": np.std(coefficients, axis=0)})
    return result, predictions, pd.DataFrame(fold_results), coef


def associations(df):
    rows = []
    for name in FEATURES:
        x, y = df[name], df.target_hamstring_season
        rho, p = spearmanr(x, y)
        row = {"feature": name, "spearman": float(rho), "p_exploratory": float(p),
               "mean_no_injury": float(x[y == 0].mean()),
               "mean_injury": float(x[y == 1].mean())}
        if x.nunique() == 2:
            tab = pd.crosstab(x, y).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            odds, fp = fisher_exact(tab)
            row.update(odds_ratio=float(odds), fisher_p=float(fp), table=tab.values.tolist())
        rows.append(row)
    return rows


def plots(df, predictions, coef, figdir):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    strata = df.groupby(["previous_hamstring_injury", "nordic_failed"]).target_hamstring_season.agg(["sum", "count", "mean"])
    labels = [f"Histórico {int(a)} / Nordic falhou {int(b)}\n(n={int(r['count'])})" for (a,b), r in strata.iterrows()]
    axes[0].bar(range(len(strata)), 100 * strata["mean"], color="#237b87")
    axes[0].set_xticks(range(len(strata)), labels, rotation=18, ha="right")
    axes[0].set_ylabel("Atletas com lesão na temporada (%)")
    axes[0].set_title("Cruzamento observado: histórico × Nordic")
    for name in ["prevalencia", "logistica", "rf_sigmoid"]:
        observed, predicted = calibration_curve(predictions.target, predictions[name], n_bins=4, strategy="quantile")
        axes[1].plot(predicted, observed, "o-", label=name)
    axes[1].plot([0,1], [0,1], "k--", alpha=.4)
    axes[1].set(xlabel="Probabilidade OOF média", ylabel="Frequência observada", title="Confiabilidade exploratória (143 atletas)")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(figdir / "cruzamento_e_calibracao.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    c = coef.sort_values("coefficient_mean")
    ax.barh(c.feature, c.coefficient_mean, xerr=c.coefficient_sd, color="#237b87")
    ax.axvline(0, color="black", lw=.7)
    ax.set(xlabel="Coeficiente padronizado médio (± DP entre folds)", title="Regressão logística: associação no modelo")
    fig.tight_layout(); fig.savefig(figdir / "coeficientes_logistica.png", dpi=220); plt.close(fig)


def run():
    for name in ["results", "figures", "data/processed"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    source = download()
    raw, meta = pyreadstat.read_sav(str(source))
    df = prepare(raw, meta)
    result, predictions, folds, coef = evaluate(df)
    assoc = associations(df)
    dictionary = [{"column": c, "label": meta.column_names_to_labels.get(c),
                   "codes": meta.variable_value_labels.get(c, {}),
                   "missing": int(raw[c].isna().sum()), "unique": int(raw[c].nunique())} for c in raw]
    report = {"dataset": "Shalaj 2020", "analysis_date": "2026-11-12",
              "source_url": URL, "article": ARTICLE, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "raw_shape": list(raw.shape), "unique_players": int(raw.player_id.nunique()),
              "missing_cells": int(raw.isna().sum().sum()),
              "target_mapping": {"1=yes": 1, "2=no": 0}, "positive_players": int(df.target_hamstring_season.sum()),
              "prevalence": float(df.target_hamstring_season.mean()),
              "features": FEATURES, "validation": "5-fold stratified OOF; RF sigmoid calibration internal cv=3",
              "clinical_7d_eligible": False, "metrics_oof": result, "associations": assoc,
              "excluded_post_baseline": [c for c in raw if "exposure" in c.lower() or "rate" in c.lower()],
              "limitations": ["No event dates or daily repeated measurements in released supplement",
                              "No club identifiers: validation cannot hold out clubs",
                              "Conditional bootstrap intervals; no external validation",
                              "Article reports 43 episodes; released binary target identifies 40 players"]}
    save_json(ROOT / "results/summary.json", report)
    save_json(ROOT / "results/data_dictionary.json", dictionary)
    df[["player_id", *FEATURES, "target_hamstring_season"]].to_csv(ROOT / "data/processed/athlete_baseline.csv", index=False)
    predictions.to_csv(ROOT / "data/processed/oof_predictions.csv", index=False)
    folds.to_csv(ROOT / "data/processed/fold_metrics.csv", index=False)
    coef.to_csv(ROOT / "data/processed/logistic_coefficients.csv", index=False)
    plots(df, predictions, coef, ROOT / "figures")
    lines = ["# Resultados executados - Shalaj", "", f"Atletas: {len(df)}; com lesão: {int(df.target_hamstring_season.sum())}; prevalência: {df.target_hamstring_season.mean():.2%}.", "",
             "| Modelo | ROC-AUC OOF | IC 95% condicional | AP | Brier | Log loss |",
             "|---|---:|---|---:|---:|---:|"]
    for name, m in result.items():
        lo, hi = m["auc_bootstrap_95_conditional"]
        lines.append(f"| {name} | {m['roc_auc']:.4f} | {lo:.3f}–{hi:.3f} | {m['average_precision']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} |")
    lines += ["", "Resultados internos exploratórios. Não são previsões validadas por clube nem risco em sete dias.", "",
              "## Associações observadas", "", "| Variável | Spearman | Média sem lesão | Média com lesão |", "|---|---:|---:|---:|"]
    for a in assoc:
        lines.append(f"| {a['feature']} | {a['spearman']:+.3f} | {a['mean_no_injury']:.3f} | {a['mean_injury']:.3f} |")
    (ROOT / "results/RESULTADOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return report


if __name__ == "__main__":
    run()
