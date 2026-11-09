#!/usr/bin/env python3
"""Run the fixed-setting Oldenlandia LOTO analyses specified in the manuscript.

The pipeline reads the deposited MTBLS13314 abundance matrices and sample metadata,
fits preprocessing within each relevant training fold, and writes sample-level
predictions, validation metrics, audit records, and manuscript figures.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_input"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
SEED = 20260828
TISSUES = ["root", "stem", "leaf", "flower", "fruit"]
SPECIES_NEG = "Oldenlandia corymbosa"
SPECIES_POS = "Oldenlandia diffusa"
MODES = ["negative", "positive"]

META_COLS = [
    "database_identifier", "chemical_formula", "smiles", "inchi",
    "metabolite_identification", "mass_to_charge", "fragmentation",
    "modifications", "charge", "retention_time", "taxid", "species",
    "database", "database_version", "reliability", "uri", "search_engine",
    "search_engine_score", "smallmolecule_abundance_sub",    "smallmolecule_abundance_stdev_sub", "smallmolecule_abundance_std_error_sub",
]
PANEL_DB_IDS = ["Lmyn002435", "Zbln008310", "Labn003679", "Zbln008344", "Wbmn004986"]
PANEL_CONDITIONS = {
    "Full five-marker panel": PANEL_DB_IDS,
    "Minus Zbln008310": [x for x in PANEL_DB_IDS if x != "Zbln008310"],
    "Minus Zbln008344": [x for x in PANEL_DB_IDS if x != "Zbln008344"],
    "Minus both missingness-heavy markers": [x for x in PANEL_DB_IDS if x not in {"Zbln008310", "Zbln008344"}],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    for p in (RESULTS, FIGURES, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def load_data():
    sample_meta = pd.read_csv(DATA / "s_MTBLS13314.txt", sep="\t", dtype=str)
    mode_matrices = []
    mode_metadata = []
    for mode in MODES:
        path = DATA / f"m_MTBLS13314_LC-MS_{mode}_reverse-phase_metabolite_profiling_v2_maf.tsv"
        raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True, na_values=[""])
        missing_meta = [c for c in META_COLS if c not in raw.columns]
        if missing_meta:
            raise ValueError(f"Missing expected metadata columns in {path.name}: {missing_meta}")
        sample_cols = [c for c in raw.columns if c not in META_COLS]
        meta = raw[META_COLS].copy()
        meta["ion_mode"] = mode
        meta["feature_id"] = [
            f"{mode[:3]}::{db}::{name}"
            for db, name in zip(meta["database_identifier"], meta["metabolite_identification"])
        ]
        meta["source_row"] = np.arange(len(meta), dtype=int)
        values = raw[sample_cols].apply(pd.to_numeric, errors="coerce")
        invalid_numeric = raw[sample_cols].notna() & values.isna()
        if invalid_numeric.to_numpy().any():
            raise ValueError(
                f"Non-blank abundance cells could not be parsed as numeric in {path.name}"
            )
        values.index = meta["feature_id"]
        mode_matrices.append(values)
        mode_metadata.append(meta)

    feature_meta = pd.concat(mode_metadata, ignore_index=True)
    if feature_meta["feature_id"].duplicated().any():
        raise ValueError("Feature IDs are not unique after mode/database/name encoding")
    combined = pd.concat(mode_matrices, axis=0).T
    if combined.index.duplicated().any():
        raise ValueError("Duplicate sample columns in assay matrices")

    sm = sample_meta.set_index("Sample Name").loc[combined.index].copy()
    sm.index.name = "Sample Name"
    sm = sm.reset_index()
    sm["is_qc"] = sm["Factor Value[Tissue]"].eq("Pooled Sample")
    sm["species"] = sm["Characteristics[Organism]"]
    sm["tissue"] = sm["Factor Value[Tissue]"]
    sm["replicate"] = pd.to_numeric(sm["Factor Value[Repetition]"], errors="coerce")

    bio_meta = sm.loc[~sm["is_qc"], ["Sample Name", "species", "tissue", "replicate", "is_qc"]].copy()
    bio = combined.loc[bio_meta["Sample Name"]].copy()
    bio.index = bio_meta["Sample Name"].to_numpy()
    X = bio.to_numpy(dtype=float)
    y = (bio_meta["species"].to_numpy() == SPECIES_POS).astype(int)
    tissue = bio_meta["tissue"].to_numpy(dtype=str)
    sample_ids = bio_meta["Sample Name"].to_numpy(dtype=str)
    feature_ids = feature_meta["feature_id"].to_numpy(dtype=str)
    return sm, feature_meta, combined, bio_meta.reset_index(drop=True), X, y, tissue, sample_ids, feature_ids


@dataclass
class FoldPreprocessor:
    fill_values: np.ndarray
    active_mask: np.ndarray
    means: np.ndarray | None
    scales: np.ndarray | None
    do_scale: bool

    def transform(self, X: np.ndarray) -> np.ndarray:
        z = np.asarray(X, dtype=float).copy()
        missing = ~np.isfinite(z)
        if missing.any():
            z[missing] = np.take(self.fill_values, np.where(missing)[1])
        z = np.log2(np.maximum(z, 0.0) + 1.0)
        z = z[:, self.active_mask]
        if self.do_scale:
            z = (z - self.means) / self.scales
        return z


def fit_preprocessor(X_train: np.ndarray, do_scale: bool) -> FoldPreprocessor:
    x = np.asarray(X_train, dtype=float)
    positive = np.where(np.isfinite(x) & (x > 0), x, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_positive = np.nanmin(positive, axis=0)
    fill = np.where(np.isfinite(min_positive), 0.5 * min_positive, 0.0)
    z = np.where(np.isfinite(x), x, fill[None, :])
    z = np.log2(np.maximum(z, 0.0) + 1.0)
    variance = np.var(z, axis=0, ddof=0)
    active = np.isfinite(variance) & (variance > 0.0)
    if not active.any():
        raise ValueError("All features are constant in this training fold")
    za = z[:, active]
    if do_scale:
        means = za.mean(axis=0)
        scales = za.std(axis=0, ddof=0)
        scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    else:
        means = scales = None
    return FoldPreprocessor(fill, active, means, scales, do_scale)


def model_settings(model_name: str, seed: int) -> dict[str, Any]:
    if model_name == "elastic_net":
        return {
            "class": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss", "penalty": "elasticnet", "alpha": 0.001,
            "l1_ratio": 0.15, "learning_rate": "optimal", "shuffle": True,
            "max_iter": 10000, "tol": 1e-4, "early_stopping": False,
            "class_weight": None, "random_state": int(seed),
        }
    if model_name == "pls_da":
        return {
            "class": "sklearn.cross_decomposition.PLSRegression",
            "n_components": 2, "scale": False, "max_iter": 500, "tol": 1e-6,
        }
    if model_name == "random_forest":
        return {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 60, "max_features": "sqrt", "min_samples_leaf": 1,
            "criterion": "gini", "max_depth": None, "class_weight": None,
            "bootstrap": True, "random_state": int(seed), "n_jobs": 1,
        }
    raise KeyError(model_name)


def build_model(model_name: str, seed: int):
    cfg = model_settings(model_name, seed)
    if model_name == "elastic_net":
        return SGDClassifier(
            loss=cfg["loss"], penalty=cfg["penalty"], alpha=cfg["alpha"],
            l1_ratio=cfg["l1_ratio"], learning_rate=cfg["learning_rate"],
            shuffle=cfg["shuffle"], max_iter=cfg["max_iter"], tol=cfg["tol"],
            early_stopping=cfg["early_stopping"], class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
        )
    if model_name == "pls_da":
        return PLSRegression(n_components=cfg["n_components"], scale=cfg["scale"],
                             max_iter=cfg["max_iter"], tol=cfg["tol"])
    return RandomForestClassifier(
        n_estimators=cfg["n_estimators"], max_features=cfg["max_features"],
        min_samples_leaf=cfg["min_samples_leaf"], criterion=cfg["criterion"],
        max_depth=cfg["max_depth"], class_weight=cfg["class_weight"],
        bootstrap=cfg["bootstrap"], random_state=cfg["random_state"], n_jobs=cfg["n_jobs"],
    )


def score_model(model_name: str, model, X: np.ndarray) -> np.ndarray:
    if model_name == "pls_da":
        return np.asarray(model.predict(X), dtype=float).reshape(-1)
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def metric_dict(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray, seed: int):
    do_scale = model_name in {"elastic_net", "pls_da"}
    prep = fit_preprocessor(X_train, do_scale=do_scale)
    Xtr = prep.transform(X_train)
    Xte = prep.transform(X_test)
    model = build_model(model_name, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Xtr, y_train)
    score = score_model(model_name, model, Xte)
    return score, prep, model, [str(w.message) for w in caught]


def save_split_table(path: Path, sample_ids, tissues, species, fold_name: str,
                     fold_index: int, train_idx: np.ndarray, test_idx: np.ndarray):
    test_set = set(test_idx.tolist())
    rows = []
    for i, sid in enumerate(sample_ids):
        rows.append({
            "validation": fold_name, "fold_index_zero_based": fold_index,
            "held_out_tissue_or_fold": fold_name,
            "sample_id": sid, "tissue": tissues[i], "species": species[i],
            "partition": "test" if i in test_set else "train",
        })
    return rows


def rank_training_features(X_train: np.ndarray, y_train: np.ndarray,
                           feature_ids: np.ndarray, top_k: int = 20):
    prep = fit_preprocessor(X_train, do_scale=True)
    Z = prep.transform(X_train)
    active_ids = feature_ids[prep.active_mask]
    difference = np.abs(Z[y_train == 1].mean(axis=0) - Z[y_train == 0].mean(axis=0))
    order = sorted(range(len(difference)), key=lambda j: (-float(difference[j]), str(active_ids[j])))
    out = []
    for rank, j in enumerate(order[:top_k], start=1):
        out.append({"rank": rank, "feature_id": str(active_ids[j]),
                    "absolute_standardized_mean_difference": float(difference[j])})
    return out


def run_loto(X, y, tissue, sample_ids, species, feature_ids):
    predictions, metrics, splits, ranking_rows, warnings_log = [], [], [], [], []
    for tissue_index, held_tissue in enumerate(TISSUES):
        test_idx = np.flatnonzero(tissue == held_tissue)
        train_idx = np.flatnonzero(tissue != held_tissue)
        if len(train_idx) != 48 or len(test_idx) != 12:
            raise AssertionError(f"Unexpected LOTO split size for {held_tissue}")
        splits.extend(save_split_table(None, sample_ids, tissue, species, held_tissue,
                                       tissue_index, train_idx, test_idx))
        ranks = rank_training_features(X[train_idx], y[train_idx], feature_ids, 20)
        for r in ranks:
            ranking_rows.append({"held_out_tissue": held_tissue, **r})
        for model_name in ("elastic_net", "pls_da", "random_forest"):
            seed = SEED + tissue_index
            score, prep, _, caught = fit_predict(
                model_name, X[train_idx], y[train_idx], X[test_idx], seed
            )
            warnings_log.extend({"validation": "LOTO", "fold": held_tissue,
                                 "model": model_name, "warning": w} for w in caught)
            pred = (score >= 0.5).astype(int)
            for local, global_i in enumerate(test_idx):
                predictions.append({
                    "sample_id": sample_ids[global_i], "tissue": tissue[global_i],
                    "species": species[global_i], "y_true": int(y[global_i]),
                    "model": model_name, "score": float(score[local]),
                    "y_pred": int(pred[local]), "correct": bool(pred[local] == y[global_i]),
                    "held_out_fold": held_tissue, "seed": seed,
                    "n_train": len(train_idx), "n_test": len(test_idx),
                    "n_features_after_training_filter": int(prep.active_mask.sum()),
                })
            md = metric_dict(y[test_idx], score)
            metrics.append({"model": model_name, "validation": "LOTO",
                            "held_out_tissue": held_tissue, "seed": seed,
                            "n_features_after_training_filter": int(prep.active_mask.sum()), **md})
    pred_df = pd.DataFrame(predictions)
    for model_name, group in pred_df.groupby("model", sort=False):
        md = metric_dict(group["y_true"].to_numpy(), group["score"].to_numpy())
        metrics.append({"model": model_name, "validation": "LOTO_pooled",
                        "held_out_tissue": "all five tissues", "seed": SEED, **md})
    return pred_df, pd.DataFrame(metrics), pd.DataFrame(splits), pd.DataFrame(ranking_rows), warnings_log


def run_random_cv(X, y, tissue, sample_ids, species, feature_ids):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    predictions, metrics, splits, ranking_rows, warnings_log = [], [], [], [], []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        fold_label = f"fold_{fold_index + 1}"
        splits.extend(save_split_table(None, sample_ids, tissue, species, fold_label,
                                       fold_index, train_idx, test_idx))
        for r in rank_training_features(X[train_idx], y[train_idx], feature_ids, 20):
            ranking_rows.append({"fold": fold_label, **r})
        seed = SEED + 100 + fold_index
        score, prep, _, caught = fit_predict(
            "elastic_net", X[train_idx], y[train_idx], X[test_idx], seed
        )
        warnings_log.extend({"validation": "random_CV", "fold": fold_label,
                             "model": "elastic_net", "warning": w} for w in caught)
        pred = (score >= 0.5).astype(int)
        for local, global_i in enumerate(test_idx):
            predictions.append({
                "sample_id": sample_ids[global_i], "tissue": tissue[global_i],
                "species": species[global_i], "y_true": int(y[global_i]),
                "fold": fold_label, "fold_index_zero_based": fold_index,
                "score": float(score[local]), "y_pred": int(pred[local]),
                "correct": bool(pred[local] == y[global_i]), "seed": seed,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "n_features_after_training_filter": int(prep.active_mask.sum()),
            })
        metrics.append({"model": "elastic_net", "validation": "random_CV_fold",
                        "fold": fold_label, "seed": seed,
                        "n_features_after_training_filter": int(prep.active_mask.sum()),
                        **metric_dict(y[test_idx], score)})
    pred_df = pd.DataFrame(predictions)
    metrics.append({"model": "elastic_net", "validation": "random_CV_pooled",
                    "fold": "all five folds", "seed": SEED,
                    **metric_dict(pred_df["y_true"].to_numpy(), pred_df["score"].to_numpy())})
    return pred_df, pd.DataFrame(metrics), pd.DataFrame(splits), pd.DataFrame(ranking_rows), warnings_log


def run_panel_ablation(X, y, tissue, sample_ids, species, feature_meta, feature_ids):
    id_to_idx = {fid: i for i, fid in enumerate(feature_ids)}
    db_to_feature = {}
    for _, row in feature_meta.iterrows():
        if row["ion_mode"] == "negative":
            db_to_feature[str(row["database_identifier"])] = str(row["feature_id"])
    missing_ids = [db for db in PANEL_DB_IDS if db not in db_to_feature]
    if missing_ids:
        raise ValueError(f"Panel database IDs missing from negative-mode matrix: {missing_ids}")
    prediction_rows, metric_rows, split_rows, warnings_log = [], [], [], []
    for condition, db_ids in PANEL_CONDITIONS.items():
        fids = [db_to_feature[x] for x in db_ids]
        idx = np.array([id_to_idx[x] for x in fids], dtype=int)
        scores_all = np.full(len(y), np.nan, dtype=float)
        for tissue_index, held_tissue in enumerate(TISSUES):
            test_idx = np.flatnonzero(tissue == held_tissue)
            train_idx = np.flatnonzero(tissue != held_tissue)
            split_rows.extend(save_split_table(None, sample_ids, tissue, species,
                                               held_tissue, tissue_index, train_idx, test_idx))
            score, prep, _, caught = fit_predict(
                "pls_da", X[train_idx][:, idx], y[train_idx], X[test_idx][:, idx], SEED + tissue_index
            )
            warnings_log.extend({"validation": "panel_ablation", "condition": condition,
                                 "fold": held_tissue, "model": "pls_da", "warning": w}
                                for w in caught)
            scores_all[test_idx] = score
            pred = (score >= 0.5).astype(int)
            for local, global_i in enumerate(test_idx):
                prediction_rows.append({
                    "condition": condition, "panel_size": len(db_ids),
                    "included_database_ids": ";".join(db_ids),
                    "sample_id": sample_ids[global_i], "tissue": tissue[global_i],
                    "species": species[global_i], "y_true": int(y[global_i]),
                    "score": float(score[local]), "y_pred": int(pred[local]),
                    "correct": bool(pred[local] == y[global_i]),
                    "held_out_fold": held_tissue,
                    "n_features_after_training_filter": int(prep.active_mask.sum()),
                })
            metric_rows.append({"condition": condition, "panel_size": len(db_ids),
                                "validation": "LOTO_fold", "held_out_tissue": held_tissue,
                                **metric_dict(y[test_idx], score)})
        metric_rows.append({"condition": condition, "panel_size": len(db_ids),
                            "validation": "LOTO_pooled", "held_out_tissue": "all five tissues",
                            **metric_dict(y, scores_all)})
    return (pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows),
            pd.DataFrame(split_rows), warnings_log)


def run_data_audit(sm, feature_meta, combined, bio_meta, X):
    bio_ids = bio_meta["Sample Name"].tolist()
    qc_ids = sm.loc[sm["is_qc"], "Sample Name"].tolist()
    Xbio = combined.loc[bio_ids].to_numpy(dtype=float)
    Xqc = combined.loc[qc_ids].to_numpy(dtype=float)
    qc_rows = []
    for mode in MODES:
        fids = feature_meta.loc[feature_meta["ion_mode"].eq(mode), "feature_id"].tolist()
        # QC summaries are grouped by the deposited Organism field (3 QCs per label).
        qc_meta = sm[sm["is_qc"]].set_index("Sample Name")
        for qc_species, grp in qc_meta.groupby("species", sort=True):
            vals = combined.loc[grp.index.tolist(), fids].to_numpy(dtype=float)
            cvs = []
            for j in range(vals.shape[1]):
                v = vals[:, j]
                v = v[np.isfinite(v)]
                cvs.append(100 * np.std(v, ddof=1) / np.mean(v) if len(v) >= 2 and np.mean(v) > 0 else np.nan)
            qc_rows.append({
                "ion_mode": mode, "qc_species_group": qc_species,
                "qc_replicates": len(grp), "features_total": len(fids),
                "features_cv_estimable_n": int(np.isfinite(cvs).sum()),
                "median_cv_pct": float(np.nanmedian(cvs)),
                "q25_cv_pct": float(np.nanpercentile(cvs, 25)),
                "q75_cv_pct": float(np.nanpercentile(cvs, 75)),
                "pct_features_cv_le_30": float(100 * np.nanmean(np.asarray(cvs) <= 30)),
            })
    qc_df = pd.DataFrame(qc_rows)

    # Descriptive audit only; this global imputation is not used by predictive models.
    positive = np.where(np.isfinite(Xbio) & (Xbio > 0), Xbio, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        minpos = np.nanmin(positive, axis=0)
    fill = np.where(np.isfinite(minpos), 0.5 * minpos, 0.0)
    z = np.where(np.isfinite(Xbio), Xbio, fill[None, :])
    z = np.log2(np.maximum(z, 0.0) + 1.0)
    var = np.var(z, axis=0, ddof=0)
    audit = {
        "total_samples": int(len(sm)), "biological_samples": int(len(bio_meta)),
        "qc_samples": int(sm["is_qc"].sum()),
        "species_counts_biological": bio_meta["species"].value_counts().to_dict(),
        "tissue_counts_biological": bio_meta["tissue"].value_counts().reindex(TISSUES).to_dict(),
        "replicates_per_species_tissue": int(bio_meta.groupby(["species", "tissue"]).size().min()),
        "negative_mode_features": int((feature_meta["ion_mode"] == "negative").sum()),
        "positive_mode_features": int((feature_meta["ion_mode"] == "positive").sum()),
        "combined_features": int(len(feature_meta)),
        "biological_missing_cells": int((~np.isfinite(Xbio)).sum()),
        "biological_missing_pct": float(100 * (~np.isfinite(Xbio)).mean()),
        "qc_missing_cells": int((~np.isfinite(Xqc)).sum()),
        "qc_missing_pct": float(100 * (~np.isfinite(Xqc)).mean()),
        "observed_numeric_zeros_biological": int((np.isfinite(Xbio) & (Xbio == 0)).sum()),
        "exact_constant_features_after_descriptive_imputation": int((var == 0).sum()),
        "duplicate_sample_names": int(sm["Sample Name"].duplicated().sum()),
        "duplicate_feature_ids": int(feature_meta["feature_id"].duplicated().sum()),
        "cross_mode_duplicate_metabolite_names": int(
            feature_meta.groupby("metabolite_identification")["ion_mode"].nunique().gt(1).sum()
        ),
        "assay_sample_order_matches_metadata": bool(list(combined.index) == list(sm["Sample Name"])),
        "qc_organism_labels": sm.loc[sm["is_qc"], ["Sample Name", "Characteristics[Organism]", "Characteristics[Variant]"]].to_dict(orient="records"),
        "qc_note": "QC01-QC03 have conflicting Organism and Variant labels; QC CVs are grouped by the deposited Organism field, as in the manuscript.",
        "feature_count_note": "Deposited matrices contain 629 negative-mode plus 714 positive-mode rows (1,343 total); the source investigation description reports 1,342.",
    }
    return audit, qc_df

