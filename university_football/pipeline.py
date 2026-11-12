"""Classificação exploratória de Injury_Next_Season; sem validade clínica presumida."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SLUG = "yuanchunhong/university-football-injury-prediction-dataset"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"
TARGET = "Injury_Next_Season"
NUMERIC = ["Age", "Height_cm", "Weight_kg", "Training_Hours_Per_Week",
           "Matches_Played_Past_Season", "Previous_Injury_Count", "Knee_Strength_Score",
           "Hamstring_Flexibility", "Reaction_Time_ms", "Balance_Test_Score",
           "Sprint_Speed_10m_s", "Agility_Score", "Sleep_Hours_Per_Night",
           "Stress_Level_Score", "Nutrition_Quality_Score", "Warmup_Routine_Adherence", "BMI"]
FEATURES = NUMERIC + ["Position"]
SEED = 42


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def download():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "dataset.zip"
    if not archive.exists():
        response = requests.get(URL, timeout=90)
        response.raise_for_status()
        if not response.content.startswith(b"PK"):
            raise ValueError("Kaggle não retornou um ZIP; não substituir por dados gerados.")
        archive.write_bytes(response.content)
    with zipfile.ZipFile(archive) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if csvs != ["data.csv"]:
            raise ValueError(f"Conteúdo do ZIP mudou: {csvs}")
        (raw / "data.csv").write_bytes(z.read("data.csv"))
    meta_path = raw / "kaggle_metadata.json"
    if not meta_path.exists():
        response = requests.get(f"https://www.kaggle.com/api/v1/datasets/view/{SLUG}", timeout=90)
        response.raise_for_status()
        save(meta_path, response.json())
    return raw / "data.csv", json.loads(meta_path.read_text(encoding="utf-8"))


def inspect(df):
    if set(df.columns) != set(FEATURES + [TARGET]):
        raise ValueError("O esquema não corresponde aos 18 atributos e ao alvo inspecionados.")
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[FEATURES].duplicated().sum() == 0, "Preditores idênticos exigem agrupamento na validação."
    assert df[NUMERIC].apply(pd.to_numeric, errors="coerce").notna().all().all()
    assert df.Position.notna().all()
    assert set(df.Warmup_Routine_Adherence.unique()) <= {0, 1}
    bmi_error = (df.BMI - df.Weight_kg / (df.Height_cm / 100) ** 2).abs()
    return {"rows": len(df), "columns": len(df.columns), "missing_cells": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()), "positive_labels": int(df[TARGET].sum()),
            "prevalence": float(df[TARGET].mean()), "athlete_id_available": False,
            "event_dates_available": False, "bmi_max_recalculation_error": float(bmi_error.max()),
            "stress_values_above_article_PSS10_max40": int(df.Stress_Level_Score.gt(40).sum()),
            "sprint_named_seconds_range": [float(df.Sprint_Speed_10m_s.min()), float(df.Sprint_Speed_10m_s.max())]}


def metric(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}


def evaluate(df):
    pre = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("position", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)), ["Position"])
    ])
    models = {"prevalencia": DummyClassifier(strategy="prior"),
              "logistica": make_pipeline(clone(pre), LogisticRegression(C=1, max_iter=2000, random_state=SEED)),
              "rf_sigmoid": CalibratedClassifierCV(
                  make_pipeline(clone(pre), RandomForestClassifier(n_estimators=200, max_depth=5,
                                min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
                  method="sigmoid", cv=3, n_jobs=1)}
    X, y = df[FEATURES], df[TARGET]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    predictions = pd.DataFrame({"row_id": np.arange(len(df)), "target": y, "fold": -1})
    metrics, per_fold, importance = {}, [], []
    for name, model in models.items():
        p = np.full(len(df), np.nan)
        for fold, (train, test) in enumerate(folds):
            assert not set(train) & set(test)
            fitted = clone(model).fit(X.iloc[train], y.iloc[train])
            p[test] = fitted.predict_proba(X.iloc[test])[:, 1]
            predictions.loc[test, "fold"] = fold
            per_fold.append({"model": name, "fold": fold, **metric(y.iloc[test], p[test])})
            if name == "logistica":
                imp = permutation_importance(fitted, X.iloc[test], y.iloc[test], scoring="roc_auc",
                                             n_repeats=8, random_state=SEED, n_jobs=1)
                importance.append(imp.importances_mean)
        assert np.isfinite(p).all()
        predictions[name] = p
        metrics[name] = metric(y, p)
        rng = np.random.default_rng(SEED)
        aucs = []
        for _ in range(1000):
            ix = rng.integers(0, len(y), len(y))
            if y.iloc[ix].nunique() == 2:
                aucs.append(roc_auc_score(y.iloc[ix], p[ix]))
        metrics[name]["auc_bootstrap_95_conditional"] = np.quantile(aucs, [.025, .975]).tolist()
    imp = pd.DataFrame({"feature": FEATURES, "auc_drop_mean": np.mean(importance, axis=0),
                        "auc_drop_sd_between_folds": np.std(importance, axis=0)})
    return metrics, predictions, pd.DataFrame(per_fold), imp


def figures(df, pred, corr, imp):
    out = ROOT / "figures"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ordered = corr.sort_values()
    axes[0].barh(ordered.index, ordered.values, color=["#ba4a42" if x > 0 else "#237b87" for x in ordered.values])
    axes[0].axvline(0, color="black", lw=.5)
    axes[0].set(xlabel="Spearman com o rótulo fornecido", title="Associações exploratórias (800 linhas)")
    for name in ["prevalencia", "logistica", "rf_sigmoid"]:
        obs, p = calibration_curve(pred.target, pred[name], n_bins=6, strategy="quantile")
        axes[1].plot(p, obs, "o-", label=name)
    axes[1].plot([0,1], [0,1], "k--", alpha=.4)
    axes[1].set(xlabel="Probabilidade OOF do rótulo 1", ylabel="Frequência do rótulo 1", title="Classificação do rótulo; sem validação clínica")
    axes[1].legend(); fig.tight_layout()
    fig.savefig(out / "associacoes_e_calibracao.png", dpi=220); plt.close(fig)
    # Tercis usados apenas para descrição; não são limiares de risco.
    descriptive = df.copy()
    descriptive["sleep_group"] = pd.qcut(df.Sleep_Hours_Per_Night, 3, labels=["Sono baixo", "Sono médio", "Sono alto"])
    descriptive["stress_group"] = pd.qcut(df.Stress_Level_Score, 3, labels=["Estresse baixo", "Estresse médio", "Estresse alto"])
    g = descriptive.groupby(["sleep_group", "stress_group"], observed=True)[TARGET].agg(["sum", "count", "mean"])
    g.to_csv(ROOT / "data/processed/sleep_stress_strata.csv")
    mat = g["mean"].unstack("stress_group")
    counts = g["count"].unstack("stress_group")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(mat, vmin=0, vmax=1, cmap="YlOrRd")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{mat.iloc[i,j]:.1%}\nn={counts.iloc[i,j]}", ha="center", va="center")
    axes[0].set_xticks(range(3), mat.columns, rotation=20)
    axes[0].set_yticks(range(3), mat.index)
    axes[0].set_title("Sono × estresse: frequência observada de rótulo 1")
    fig.colorbar(im, ax=axes[0], shrink=.7)
    top = imp.sort_values("auc_drop_mean").tail(10)
    axes[1].barh(top.feature, top.auc_drop_mean, xerr=top.auc_drop_sd_between_folds, color="#237b87")
    axes[1].set(xlabel="Queda de AUC por permutação no teste externo", title="Importância na regressão logística (± DP dos folds)")
    fig.tight_layout(); fig.savefig(out / "sono_estresse_e_importancia.png", dpi=220); plt.close(fig)
    return g.reset_index()


def run():
    for name in ["results", "figures", "data/processed"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    source, metadata = download()
    df = pd.read_csv(source)
    audit = inspect(df)
    metrics, pred, folds, importance = evaluate(df)
    corr = df[NUMERIC + [TARGET]].corr(method="spearman")[TARGET].drop(TARGET)
    strata = figures(df, pred, corr, importance)
    dictionary = [{"column": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
                   "unique": int(df[c].nunique()),
                   "minimum": float(df[c].min()) if c != "Position" else None,
                   "maximum": float(df[c].max()) if c != "Position" else None} for c in df]
    summary = {"dataset": SLUG, "analysis_date": "2026-11-12", "download_url": URL,
               "sha256_csv": hashlib.sha256(source.read_bytes()).hexdigest(),
               "sha256_zip": hashlib.sha256((source.parent / "dataset.zip").read_bytes()).hexdigest(),
               "kaggle_version": metadata.get("currentVersionNumber"), "license_reported": metadata.get("licenseName"),
               "audit": audit, "features": FEATURES, "target": TARGET,
               "label_semantics_verified": False, "clinical_7d_eligible": False,
               "validation": "5-fold stratified OOF by row; RF calibration internal cv=3; no athlete IDs to verify identities",
               "metrics_label_classification_oof": metrics, "spearman": corr.to_dict(),
               "sleep_stress_strata": strata.to_dict(orient="records"),
               "limitations": ["Article inconsistently describes observed injury and assessed high risk",
                               "Article states data are not a real-world clinical prospective cohort",
                               "Stress scale conflicts with PSS-10 range stated in article",
                               "Sprint column name and units require clarification",
                               "No athlete IDs, event dates, clinical diagnoses or external validation",
                               "Balanced labels are not a population injury incidence estimate"]}
    save(ROOT / "results/summary.json", summary)
    save(ROOT / "results/data_dictionary.json", dictionary)
    save(ROOT / "results/environment.json", {n:importlib.metadata.version(n) for n in ["numpy","pandas","scipy","scikit-learn","matplotlib","requests"]})
    pred.to_csv(ROOT / "data/processed/oof_predictions.csv", index=False)
    folds.to_csv(ROOT / "data/processed/fold_metrics.csv", index=False)
    importance.to_csv(ROOT / "data/processed/permutation_importance.csv", index=False)
    lines = ["# Resultados executados - University Football", "", "**Classificação do rótulo publicado; não validação de lesão clínica.**", "",
             f"Linhas: {len(df)}; rótulo 1: {df[TARGET].sum()}; prevalência do rótulo: {df[TARGET].mean():.2%}.", "",
             "| Modelo | ROC-AUC OOF | IC 95% condicional | AP | Brier | Log loss |", "|---|---:|---|---:|---:|---:|"]
    for name, m in metrics.items():
        lo, hi = m["auc_bootstrap_95_conditional"]
        lines.append(f"| {name} | {m['roc_auc']:.4f} | {lo:.3f}–{hi:.3f} | {m['average_precision']:.4f} | {m['brier']:.4f} | {m['log_loss']:.4f} |")
    lines += ["", "## Correlações exploratórias", "", "| Variável | Spearman |", "|---|---:|"]
    for name, value in corr.sort_values().items():
        lines.append(f"| {name} | {value:+.3f} |")
    (ROOT / "results/RESULTADOS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    run()
