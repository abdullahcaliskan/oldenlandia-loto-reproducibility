import json
import platform
import time
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import scipy
import sklearn

from analysis_core import (
    FIGURES,
    LOGS,
    PANEL_CONDITIONS,
    PANEL_DB_IDS,
    RESULTS,
    SEED,
    SPECIES_NEG,
    SPECIES_POS,
    TISSUES,
    ensure_dirs,
    load_data,
    model_settings,
    run_data_audit,
    run_loto,
    run_panel_ablation,
    run_random_cv,
    sha256,
)


def make_figures(
    loto_pred,
    loto_metrics,
    random_metrics,
    panel_pred,
    panel_metrics,
    feature_meta,
    combined,
    bio_meta,
    feature_ids,
):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    # Figure 1: conceptual overview contrasting random splitting with tissue holdout.
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    ax.text(6, 5.08, "Biological design and LC–MS assay", ha="center",
            va="center", fontsize=13, weight="bold", color="#18344A")
    ax.text(6, 4.68,
            "O. diffusa  |  O. corymbosa  •  5 tissues  •  60 biological samples  •  1,343 assay features",
            ha="center", va="center", fontsize=9.5, color="#40566B")
    ax.plot([0.45, 11.55], [4.35, 4.35], color="#B7C4CE", lw=1.0)
    ax.text(3.0, 4.02, "Typical approach", ha="center", va="center",
            fontsize=11, weight="bold", color="#8B5E34")
    ax.text(9.0, 4.02, "This study", ha="center", va="center",
            fontsize=11, weight="bold", color="#1F6B57")

    def overview_box(x, y, w, h, title, body, face, edge):
        patch = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                              linewidth=1.1, joinstyle="round")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="center",
                fontsize=9.3, weight="bold", color="#18344A")
        ax.text(x + w / 2, y + 0.32, body, ha="center", va="center",
                fontsize=8.0, color="#263746", linespacing=1.25)

    overview_box(0.55, 2.55, 2.2, 0.95, "Random sample split",
                 "Samples from the same tissues\nmay cross the split",
                 "#FFF7ED", "#C28A55")
    overview_box(3.15, 2.55, 2.2, 0.95, "Many features",
                 "High-dimensional chemical\nfingerprints",
                 "#FFF7ED", "#C28A55")
    ax.annotate("", xy=(3.08, 3.02), xytext=(2.78, 3.02),
                arrowprops=dict(arrowstyle="->", color="#9A7048", lw=1.3))
    overview_box(1.05, 1.05, 4.0, 0.9, "Good internal classification",
                 "Tissue generalization remains uncertain",
                 "#FCE9D8", "#C28A55")

    overview_box(6.25, 2.55, 1.55, 0.95, "Strict LOTO",
                 "48 train /\n12 held out", "#EAF5F1", "#4D8B78")
    overview_box(8.05, 2.55, 1.55, 0.95, "Feature recurrence",
                 "Fold-specific\ntop-20 rankings", "#EAF5F1", "#4D8B78")
    overview_box(9.85, 2.55, 1.55, 0.95, "Panel robustness",
                 "Pre-specified\nfive-marker set", "#EAF5F1", "#4D8B78")
    for x1, x2 in [(7.80, 8.05), (9.60, 9.85)]:
        ax.annotate("", xy=(x2, 3.02), xytext=(x1, 3.02),
                    arrowprops=dict(arrowstyle="->", color="#4D8B78", lw=1.3))
    overview_box(7.0, 1.05, 4.6, 0.9, "Cross-tissue species discrimination",
                 "No independent biomarker validation is claimed",
                 "#DCEFE8", "#4D8B78")
    ax.text(6, 0.42,
            "Tissue-domain holdout  →  descriptive recurrence  →  interpretable chemical-marker leads",
            ha="center", va="center", fontsize=9, color="#40566B")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_overview.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_overview.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: strict LOTO design and fold-local preprocessing.
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    ax.text(5.75, 4.65, "60 biological samples  |  1,343 assay features",
            ha="center", va="center", fontsize=12, weight="bold", color="#18344A")

    def design_box(x, y, w, h, title, body, face, edge):
        rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="center",
                fontsize=9.5, weight="bold", color="#18344A")
        ax.text(x + w / 2, y + 0.34, body, ha="center", va="center",
                fontsize=8.2, color="#263746", linespacing=1.3)

    design_box(0.35, 2.75, 2.15, 1.15, "Training data",
               "Four tissues\n48 samples", "#EAF2F8", "#527A9A")
    design_box(0.35, 0.95, 2.15, 1.15, "Held-out data",
               "One tissue\n12 samples", "#FFF3E7", "#C28A55")
    design_box(3.05, 2.55, 3.15, 1.55, "Fit on training data only",
               "Half-min imputation  →  log₂(x+1)\nVariance filter  →  scaling*",
               "#EAF5F1", "#4D8B78")
    design_box(6.75, 2.75, 1.75, 1.15, "Fixed classifier",
               "Fit using\n48 samples", "#EAF2F8", "#527A9A")
    design_box(6.75, 0.95, 1.75, 1.15, "Apply fitted steps",
               "Transform held-out\n12 samples", "#FFF3E7", "#C28A55")
    design_box(9.0, 1.85, 2.15, 1.25, "Out-of-fold predictions",
               "One prediction per\nheld-out sample", "#EAF5F1", "#4D8B78")
    ax.annotate("", xy=(3.0, 3.32), xytext=(2.52, 3.32),
                arrowprops=dict(arrowstyle="->", color="#527A9A", lw=1.4))
    ax.annotate("", xy=(6.7, 3.32), xytext=(6.22, 3.32),
                arrowprops=dict(arrowstyle="->", color="#527A9A", lw=1.4))
    ax.annotate("", xy=(6.7, 1.52), xytext=(2.52, 1.52),
                arrowprops=dict(arrowstyle="->", color="#C28A55", lw=1.4))
    ax.annotate("", xy=(7.35, 2.05), xytext=(5.65, 2.52),
                arrowprops=dict(arrowstyle="->", color="#4D8B78", lw=1.2,
                                linestyle="--"))
    ax.annotate("", xy=(8.95, 2.48), xytext=(8.55, 3.0),
                arrowprops=dict(arrowstyle="->", color="#4D8B78", lw=1.4))
    ax.annotate("", xy=(8.95, 2.30), xytext=(8.55, 1.52),
                arrowprops=dict(arrowstyle="->", color="#C28A55", lw=1.4))
    ax.text(5.75, 0.38,
            "Repeat for root, stem, leaf, flower and fruit; pool the 60 held-out predictions.  *Scaling for linear models; QC samples excluded.",
            ha="center", va="center", fontsize=8.3, color="#40566B")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_loto_design.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_loto_design.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: tissue-specific balanced accuracy and pooled metrics.
    tissue_metrics = loto_metrics[loto_metrics["validation"] == "LOTO"].copy()
    model_order = ["elastic_net", "pls_da", "random_forest"]
    mat = np.array([[float(tissue_metrics.loc[(tissue_metrics.model == m) & (tissue_metrics.held_out_tissue == t), "balanced_accuracy"].iloc[0]) for t in TISSUES] for m in model_order])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    axes[0].imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(5), [x.title() for x in TISSUES])
    axes[0].set_yticks(range(3), ["Elastic net", "PLS-DA", "Random forest"])
    axes[0].set_title("Balanced accuracy by held-out tissue", pad=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[0].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="#FFFFFF" if mat[i, j] > .55 else "#102A43", weight="bold")
    pooled = loto_metrics[loto_metrics["validation"].isin(["LOTO_pooled"])].set_index("model")
    # Add the pooled random-CV elastic-net result (passed in from main()).
    random_pooled = random_metrics[random_metrics.validation == "random_CV_pooled"].iloc[0]
    labels = ["Elastic net\nLOTO", "PLS-DA\nLOTO", "Random forest\nLOTO", "Elastic net\nrandom CV"]
    bal = [pooled.loc[m, "balanced_accuracy"] for m in model_order] + [random_pooled["balanced_accuracy"]]
    auc = [pooled.loc[m, "roc_auc"] for m in model_order] + [random_pooled["roc_auc"]]
    xx = np.arange(4)
    width = 0.34
    axes[1].bar(xx - width / 2, bal, width, label="Balanced accuracy", color="#3D78A8")
    axes[1].bar(xx + width / 2, auc, width, label="ROC-AUC", color="#D38A3B")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_xticks(xx, labels)
    axes[1].set_ylabel("Pooled metric (0–1)")
    axes[1].set_title("Pooled validation performance")
    axes[1].legend(
        frameon=False, fontsize=8, loc="upper center",
        bbox_to_anchor=(0.5, -0.20), ncol=2
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(FIGURES / "fig3_performance.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_performance.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: all PLS-DA held-out scores.
    pls = loto_pred[loto_pred.model == "pls_da"].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    rng = np.random.default_rng(SEED)
    class_colors = {1: "#2A6F97", 0: "#C76D3A"}
    for yi, _ in [(1, SPECIES_POS), (0, SPECIES_NEG)]:
        sub = pls[pls.y_true == yi]
        for ti, t in enumerate(TISSUES):
            vals = sub[sub.tissue == t]["score"].to_numpy()
            jitter = rng.uniform(-.12, .12, len(vals))
            ax.scatter(np.full(len(vals), ti) + jitter, vals, s=34, alpha=.85,
                       color=class_colors[yi], label=("O. diffusa" if yi == 1 and ti == 0 else "O. corymbosa" if yi == 0 and ti == 0 else None),
                       edgecolor="white", linewidth=.45)
    ax.axhline(.5, color="#A23B3B", linestyle="--", linewidth=1.2, label="Decision threshold (0.5)")
    ax.set_xticks(range(5), [x.title() for x in TISSUES])
    ax.set_ylabel("Held-out PLS-DA score")
    ax.set_xlabel("Held-out tissue")
    ax.set_title("Full-feature PLS-DA scores for all 60 held-out samples")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIGURES / "fig4_plsda_scores.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig4_plsda_scores.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    # Figure 5: panel ablation matrix and observed abundance of retained markers.
    panel_order = list(PANEL_CONDITIONS.keys())
    ppooled = panel_metrics[panel_metrics.validation == "LOTO_pooled"].set_index("condition").loc[panel_order]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1, 1.8]})
    axes[0].axis("off")
    colnames = ["M1", "M2", "M3", "M4", "M5", "Correct", "AUC"]
    rows = []
    for cond, dbs in PANEL_CONDITIONS.items():
        metric_row = ppooled.loc[cond]
        correct = int(metric_row["tp"] + metric_row["tn"])
        rows.append(["Yes" if x in dbs else "—" for x in PANEL_DB_IDS] +
                    [f"{correct}/60", f"{float(metric_row['roc_auc']):.2f}"])
    table = axes[0].table(cellText=rows, colLabels=colnames, rowLabels=["Full five", "− Zbln008310", "− Zbln008344", "− both"], cellLoc="center", loc="center", colWidths=[.10, .10, .10, .10, .10, .18, .12])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    axes[0].set_title("Retrospective PLS-DA panel ablation", pad=14)

    # Retained markers M1, M3, M5; observed values only (no imputed values plotted).
    retained = ["Lmyn002435", "Labn003679", "Wbmn004986"]
    sample_names = bio_meta["Sample Name"].tolist()
    all_feature_index = {fid: i for i, fid in enumerate(feature_ids)}
    xraw = np.asarray(combined.loc[sample_names].to_numpy(dtype=float))
    for k, db in enumerate(retained):
        fid = next(row.feature_id for row in feature_meta.itertuples() if row.ion_mode == "negative" and row.database_identifier == db)
        j = all_feature_index[fid]
        subx = xraw[:, j]
        for yi, sp in [(1, SPECIES_POS), (0, SPECIES_NEG)]:
            for ti, t in enumerate(TISSUES):
                idx = np.flatnonzero((bio_meta.species.to_numpy() == sp) & (bio_meta.tissue.to_numpy() == t))
                vals = subx[idx]
                vals = vals[np.isfinite(vals)]
                xloc = k * 2.4 + ti * .28 + (0 if yi == 1 else .13)
                if len(vals):
                    jit = rng.uniform(-.045, .045, len(vals))
                    axes[1].scatter(np.full(len(vals), xloc) + jit, np.log2(vals + 1), s=13, alpha=.72,
                                    color=class_colors[yi], label=("O. diffusa" if yi == 1 and k == 0 and ti == 0 else "O. corymbosa" if yi == 0 and k == 0 and ti == 0 else None))
    axes[1].set_xticks(
        [k * 2.4 + 0.56 for k in range(3)],
        [
            "M1: Hydrangeifolin I",
            "M3: Caffeyl alcohol\n4-O-β-D-glucopyranoside",
            "M5: Lyoniresinol-3α-O-glucoside",
        ],
        rotation=12,
        ha="right",
    )
    axes[1].set_ylabel("Observed log₂(x+1) abundance")
    axes[1].set_title("Observed abundance of the three retained candidates")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_panel_ablation.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig5_panel_ablation.png", dpi=320, bbox_inches="tight")
    plt.close(fig)

    # Diagnostic-only plot: marker measurement presence; not a manuscript supplementary figure.
    fig, ax = plt.subplots(figsize=(10, 3.8))
    sample_order = sample_names
    matrix = []
    labels = []
    for db in PANEL_DB_IDS:
        fid = next(row.feature_id for row in feature_meta.itertuples() if row.ion_mode == "negative" and row.database_identifier == db)
        vals = combined.loc[sample_order, fid].to_numpy(dtype=float)
        matrix.append(np.isfinite(vals).astype(int))
        labels.append(db)
    ax.imshow(np.asarray(matrix), aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("Biological samples (grouped by deposited sample order)")
    ax.set_title("Observed measurement presence for the five inherited marker candidates")
    fig.tight_layout()
    fig.savefig(FIGURES / "marker_measurement_presence_audit.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "marker_measurement_presence_audit.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def verify_expected_outputs(loto_metrics, random_metrics, panel_metrics, ranking_loto, ranking_cv):
    pooled = loto_metrics[loto_metrics.validation == "LOTO_pooled"].set_index("model")
    expected = {
        "elastic_net": {"accuracy": 0.8, "balanced_accuracy": 0.8, "sensitivity": 1.0, "specificity": 0.6, "f1": 5 / 6, "roc_auc": 1.0},
        "pls_da": {"accuracy": 1.0, "balanced_accuracy": 1.0, "sensitivity": 1.0, "specificity": 1.0, "f1": 1.0, "roc_auc": 1.0},
        "random_forest": {"accuracy": 0.9, "balanced_accuracy": 0.9, "sensitivity": 1.0, "specificity": 0.8, "f1": 10 / 11, "roc_auc": 0.9988888888888888},
    }
    checks = []
    for model, vals in expected.items():
        for key, target in vals.items():
            observed = float(pooled.loc[model, key])
            ok = bool(np.isclose(observed, target, atol=1e-10))
            checks.append({"check": f"LOTO pooled {model} {key}", "observed": observed, "expected": target, "pass": ok})
    rand = random_metrics[random_metrics.validation == "random_CV_pooled"].iloc[0]
    for key in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "roc_auc"]:
        observed = float(rand[key])
        checks.append({
            "check": f"random-CV pooled elastic_net {key}",
            "observed": observed,
            "expected": 1.0,
            "pass": bool(np.isclose(observed, 1.0, atol=1e-10)),
        })
    pp = panel_metrics[panel_metrics.validation == "LOTO_pooled"]
    checks.append({"check": "all four panel ablations classify 60/60", "observed": int(pp["tp"].sum() + pp["tn"].sum()), "expected": 240, "pass": bool((pp["accuracy"] == 1.0).all())})
    c_loto = Counter(ranking_loto["feature_id"])
    c_cv = Counter(ranking_cv["feature_id"])
    checks.append({"check": "LOTO top-20 recurrence: 51 unique, 4 in all five folds", "observed": f"{len(c_loto)} unique; {sum(v == 5 for v in c_loto.values())} in all five", "expected": "51 unique; 4 in all five", "pass": len(c_loto) == 51 and sum(v == 5 for v in c_loto.values()) == 4})
    checks.append({"check": "random-CV top-20 recurrence: 27 unique, 16 in all five folds", "observed": f"{len(c_cv)} unique; {sum(v == 5 for v in c_cv.values())} in all five", "expected": "27 unique; 16 in all five", "pass": len(c_cv) == 27 and sum(v == 5 for v in c_cv.values()) == 16})
    return checks


def write_audit(sm, feature_meta, audit, qc_df):
    (RESULTS / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    qc_df.to_csv(RESULTS / "qc_cv_summary.csv", index=False)
    sm.to_csv(RESULTS / "sample_metadata.csv", index=False)
    feature_meta.to_csv(RESULTS / "feature_metadata.csv", index=False)


def write_results(
    loto_pred,
    loto_metrics,
    loto_splits,
    loto_rank,
    random_pred,
    random_metrics,
    random_splits,
    random_rank,
    panel_pred,
    panel_metrics,
    panel_splits,
):
    loto_pred.to_csv(RESULTS / "loto_predictions.csv", index=False)
    loto_metrics.to_csv(RESULTS / "loto_metrics.csv", index=False)
    loto_splits.to_csv(RESULTS / "loto_split_identifiers.csv", index=False)
    loto_rank.to_csv(RESULTS / "loto_top20_rankings.csv", index=False)
    random_pred.to_csv(RESULTS / "random_cv_predictions.csv", index=False)
    random_metrics.to_csv(RESULTS / "random_cv_metrics.csv", index=False)
    random_splits.to_csv(RESULTS / "random_cv_split_identifiers.csv", index=False)
    random_rank.to_csv(RESULTS / "random_cv_top20_rankings.csv", index=False)
    panel_pred.to_csv(RESULTS / "panel_ablation_predictions.csv", index=False)
    panel_metrics.to_csv(RESULTS / "panel_ablation_metrics.csv", index=False)
    panel_splits.to_csv(RESULTS / "panel_ablation_split_identifiers.csv", index=False)


def write_marker_metadata(feature_meta, combined, bio_meta, y):
    marker_rows = []
    bio_sample_names = bio_meta["Sample Name"].tolist()
    display_names = {
        "Lmyn002435": "Hydrangeifolin I",
        "Zbln008310": "Cannabidihydrophenanthrene",
        "Labn003679": "Caffeyl alcohol 4-O-β-D-glucopyranoside",
        "Zbln008344": "2,4-dihydroxy-7-methoxy-9,10-dihydrophenanthrene",
        "Wbmn004986": "Lyoniresinol-3α-O-glucoside",
    }
    for db in PANEL_DB_IDS:
        row = feature_meta[
            (feature_meta.ion_mode == "negative")
            & (feature_meta.database_identifier == db)
        ].iloc[0]
        fid = row.feature_id
        values = combined.loc[bio_sample_names, fid].to_numpy(dtype=float)
        vals_diff = values[y == 1]
        vals_cory = values[y == 0]
        marker_rows.append(
            {
                "database_identifier": db,
                "feature_id": fid,
                "metabolite_identification_deposited": row.metabolite_identification,
                "manuscript_display_name": display_names[db],
                "chemical_formula": row.chemical_formula,
                "mass_to_charge_field": row.mass_to_charge,
                "ion_mode": row.ion_mode,
                "adduct_annotation": row.modifications,
                "detected_O_diffusa_n": int(np.isfinite(vals_diff).sum()),
                "detected_O_corymbosa_n": int(np.isfinite(vals_cory).sum()),
                "missing_O_diffusa_n": int((~np.isfinite(vals_diff)).sum()),
                "missing_O_corymbosa_n": int((~np.isfinite(vals_cory)).sum()),
            }
        )
    marker_df = pd.DataFrame(marker_rows)
    marker_df.to_csv(RESULTS / "fixed_panel_marker_metadata.csv", index=False)
    return marker_rows


def build_checks(
    checks,
    loto_splits,
    random_splits,
    panel_splits,
    loto_pred,
    random_pred,
    panel_pred,
    audit,
    marker_rows,
    qc_df,
    random_metrics,
    panel_metrics,
):
    # Independent split-coverage checks: every sample is held out once in each validation scheme.
    loto_split_ok = True
    for tissue_name in TISSUES:
        fold = loto_splits[loto_splits["held_out_tissue_or_fold"] == tissue_name]
        loto_split_ok &= (
            int((fold["partition"] == "train").sum()) == 48
            and int((fold["partition"] == "test").sum()) == 12
            and bool(fold.loc[fold["partition"] == "test", "tissue"].eq(tissue_name).all())
            and bool(fold.loc[fold["partition"] == "train", "tissue"].ne(tissue_name).all())
            and fold["sample_id"].nunique() == 60
        )
    loto_test_counts = loto_splits[loto_splits["partition"] == "test"]["sample_id"].value_counts()
    loto_split_ok &= len(loto_test_counts) == 60 and bool(loto_test_counts.eq(1).all())
    checks.append(
        {
            "check": "LOTO split integrity: 5 folds, 48 train/12 test, each sample held out once",
            "observed": bool(loto_split_ok),
            "expected": True,
            "pass": bool(loto_split_ok),
        }
    )

    random_split_ok = True
    for fold_name in random_splits["held_out_tissue_or_fold"].unique():
        fold = random_splits[random_splits["held_out_tissue_or_fold"] == fold_name]
        random_split_ok &= (
            int((fold["partition"] == "train").sum()) == 48
            and int((fold["partition"] == "test").sum()) == 12
            and fold["sample_id"].nunique() == 60
        )
    random_test_counts = random_splits[random_splits["partition"] == "test"]["sample_id"].value_counts()
    random_split_ok &= len(random_test_counts) == 60 and bool(random_test_counts.eq(1).all())
    checks.append(
        {
            "check": "Random-CV split integrity: 5 folds, 48 train/12 test, each sample held out once",
            "observed": bool(random_split_ok),
            "expected": True,
            "pass": bool(random_split_ok),
        }
    )

    panel_repeat_counts = panel_splits.groupby(
        ["held_out_tissue_or_fold", "sample_id", "partition"]
    ).size()
    panel_split_ok = len(panel_repeat_counts) == 5 * 60 and bool(panel_repeat_counts.eq(4).all())
    checks.append(
        {
            "check": "Four panel conditions reuse identical LOTO split identifiers",
            "observed": bool(panel_split_ok),
            "expected": True,
            "pass": bool(panel_split_ok),
        }
    )

    coverage_ok = (
        all(
            len(group) == 60
            and group["sample_id"].nunique() == 60
            and not group.duplicated(["sample_id"]).any()
            for _, group in loto_pred.groupby("model")
        )
        and len(random_pred) == 60
        and random_pred["sample_id"].nunique() == 60
        and not random_pred.duplicated(["sample_id"]).any()
        and all(
            len(group) == 60
            and group["sample_id"].nunique() == 60
            and not group.duplicated(["sample_id"]).any()
            for _, group in panel_pred.groupby("condition")
        )
    )
    checks.append(
        {
            "check": "Prediction coverage: one out-of-fold prediction per sample and model/condition",
            "observed": bool(coverage_ok),
            "expected": True,
            "pass": bool(coverage_ok),
        }
    )

    checks.append(
        {
            "check": "Dataset structure matches manuscript: 60 biological + 6 QC; 1,343 features",
            "observed": [audit["biological_samples"], audit["qc_samples"], audit["combined_features"]],
            "expected": [60, 6, 1343],
            "pass": audit["biological_samples"] == 60
            and audit["qc_samples"] == 6
            and audit["combined_features"] == 1343,
        }
    )
    checks.append(
        {
            "check": "No duplicate metabolite names across ion modes",
            "observed": audit["cross_mode_duplicate_metabolite_names"],
            "expected": 0,
            "pass": audit["cross_mode_duplicate_metabolite_names"] == 0,
        }
    )

    marker_lookup = {row["database_identifier"]: row for row in marker_rows}
    expected_detection = {
        "Lmyn002435": (30, 30),
        "Zbln008310": (6, 30),
        "Labn003679": (30, 30),
        "Zbln008344": (6, 30),
        "Wbmn004986": (30, 30),
    }
    marker_ok = all(
        (
            marker_lookup[key]["detected_O_diffusa_n"],
            marker_lookup[key]["detected_O_corymbosa_n"],
        )
        == expected
        and marker_lookup[key]["ion_mode"] == "negative"
        and marker_lookup[key]["adduct_annotation"] == "[M-H]-"
        for key, expected in expected_detection.items()
    )
    checks.append(
        {
            "check": "Five marker identities, detection counts, ion mode and adduct match Table 3",
            "observed": bool(marker_ok),
            "expected": True,
            "pass": bool(marker_ok),
        }
    )

    # C1: manuscript-level descriptive and model-error checks from existing outputs.
    biological_cells = int(audit["biological_samples"]) * int(audit["combined_features"])
    biological_missing = int(audit["biological_missing_cells"])
    biological_missing_pct = round(float(audit["biological_missing_pct"]), 2)
    checks.append(
        {
            "check": "Biological missing cells and percentage",
            "observed": {
                "missing_cells": biological_missing,
                "total_cells": biological_cells,
                "percentage": biological_missing_pct,
            },
            "expected": {"missing_cells": 9241, "total_cells": 80580, "percentage": 11.47},
            "pass": biological_missing == 9241
            and biological_cells == 80580
            and biological_missing_pct == 11.47,
        }
    )

    qc_missing = int(audit["qc_missing_cells"])
    qc_cells = int(
        sum(
            int(group["qc_replicates"].sum()) * int(group["features_total"].iloc[0])
            for _, group in qc_df.groupby("ion_mode", sort=False)
        )
    )
    checks.append(
        {
            "check": "QC missing cells",
            "observed": {"missing_cells": qc_missing, "total_cells": qc_cells},
            "expected": {"missing_cells": 3, "total_cells": 8058},
            "pass": qc_missing == 3 and qc_cells == 8058,
        }
    )

    qc_expected = {
        ("negative", "Oldenlandia diffusa"): 5.49,
        ("negative", "Oldenlandia corymbosa"): 5.57,
        ("positive", "Oldenlandia diffusa"): 8.83,
        ("positive", "Oldenlandia corymbosa"): 8.22,
    }
    qc_observed = {}
    for (mode, species_label), expected in qc_expected.items():
        row = qc_df[
            (qc_df["ion_mode"] == mode)
            & (qc_df["qc_species_group"] == species_label)
        ].iloc[0]
        observed = round(float(row["median_cv_pct"]), 2)
        qc_observed[f"{mode} | {species_label}"] = observed
    checks.append(
        {
            "check": "Median QC CV by ion mode and deposited species label",
            "observed": qc_observed,
            "expected": {
                "negative | Oldenlandia diffusa": 5.49,
                "negative | Oldenlandia corymbosa": 5.57,
                "positive | Oldenlandia diffusa": 8.83,
                "positive | Oldenlandia corymbosa": 8.22,
            },
            "pass": all(
                qc_observed[f"{mode} | {species_label}"] == expected
                for (mode, species_label), expected in qc_expected.items()
            ),
        }
    )

    loto_errors = loto_pred.loc[~loto_pred["correct"]]
    rf_errors = loto_errors[loto_errors["model"] == "random_forest"]
    expected_rf_errors = {"OLC_R1", "OLC_R2", "OLC_R3", "OLC_R4", "OLC_R5", "OLC_R6"}
    observed_rf_errors = set(rf_errors["sample_id"])
    checks.append(
        {
            "check": "Random forest LOTO errors are exactly the six O. corymbosa root samples",
            "observed": sorted(observed_rf_errors),
            "expected": sorted(expected_rf_errors),
            "pass": observed_rf_errors == expected_rf_errors
            and bool(rf_errors["tissue"].eq("root").all())
            and bool(rf_errors["species"].eq(SPECIES_NEG).all()),
        }
    )

    en_errors = loto_errors[loto_errors["model"] == "elastic_net"]
    expected_en_errors = {
        "OLC_R1", "OLC_R2", "OLC_R3", "OLC_R4", "OLC_R5", "OLC_R6",
        "OLC_Fr1", "OLC_Fr2", "OLC_Fr3", "OLC_Fr4", "OLC_Fr5", "OLC_Fr6",
    }
    observed_en_errors = set(en_errors["sample_id"])
    checks.append(
        {
            "check": "Elastic net LOTO errors are six O. corymbosa root and six fruit samples",
            "observed": sorted(observed_en_errors),
            "expected": sorted(expected_en_errors),
            "pass": observed_en_errors == expected_en_errors
            and set(en_errors["tissue"]) == {"root", "fruit"}
            and bool(en_errors["species"].eq(SPECIES_NEG).all()),
        }
    )

    pls_errors = loto_errors[loto_errors["model"] == "pls_da"]
    checks.append(
        {
            "check": "PLS-DA LOTO has zero classification errors",
            "observed": int(len(pls_errors)),
            "expected": 0,
            "pass": len(pls_errors) == 0,
        }
    )

    auc_rows = []
    for model_name, model_group in loto_pred.groupby("model", sort=False):
        for tissue_name, tissue_group in model_group.groupby("tissue", sort=False):
            auc_rows.append(
                {
                    "model": model_name,
                    "tissue": tissue_name,
                    "roc_auc": float(roc_auc_score(tissue_group["y_true"], tissue_group["score"])),
                }
            )
    auc_ok = len(auc_rows) == 15 and all(np.isclose(row["roc_auc"], 1.0, atol=1e-12) for row in auc_rows)
    checks.append(
        {
            "check": "Within-tissue ROC-AUC equals 1.0 for all 15 model-tissue folds",
            "observed": auc_rows,
            "expected": "15 model-tissue folds, each ROC-AUC = 1.0",
            "pass": bool(auc_ok),
        }
    )

    panel_fold = panel_metrics[panel_metrics["validation"] == "LOTO_fold"]
    panel_accuracy = {
        f"{row.condition} | {row.held_out_tissue}": float(row.accuracy)
        for row in panel_fold.itertuples()
    }
    panel_ok = len(panel_accuracy) == 20 and all(
        np.isclose(value, 1.0, atol=1e-12) for value in panel_accuracy.values()
    )
    checks.append(
        {
            "check": "Each fixed-panel condition has accuracy 1.0 in each of five tissues",
            "observed": panel_accuracy,
            "expected": "20 condition-tissue folds, each accuracy = 1.0",
            "pass": bool(panel_ok),
        }
    )

    random_fold = random_metrics[random_metrics["validation"] == "random_CV_fold"]
    random_observed = {
        row.fold: {"accuracy": float(row.accuracy), "roc_auc": float(row.roc_auc)}
        for row in random_fold.itertuples()
    }
    random_ok = len(random_observed) == 5 and all(
        np.isclose(values["accuracy"], 1.0, atol=1e-12)
        and np.isclose(values["roc_auc"], 1.0, atol=1e-12)
        for values in random_observed.values()
    )
    checks.append(
        {
            "check": "Random-CV accuracy and ROC-AUC equal 1.0 in each of five folds",
            "observed": random_observed,
            "expected": "five folds, each accuracy = 1.0 and ROC-AUC = 1.0",
            "pass": bool(random_ok),
        }
    )
    return checks


def write_checks(checks, warnings_log):
    (RESULTS / "manuscript_checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS / "warnings.json").write_text(
        json.dumps(warnings_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_manifest():
    settings = {
        "master_seed": SEED,
        "tissue_order": TISSUES,
        "positive_class": SPECIES_POS,
        "decision_threshold": 0.5,
        "loto_model_seed_rule": "master_seed + zero_based_tissue_index (root=0, stem=1, leaf=2, flower=3, fruit=4)",
        "random_cv_split": "StratifiedKFold(n_splits=5, shuffle=True, random_state=master_seed)",
        "random_cv_estimator_seed_rule": "master_seed + 100 + zero_based_fold_index",
        "preprocessing": {
            "missing_value_fill": "one-half the smallest positive abundance in the relevant training subset; zero if no positive training abundance exists",
            "transformation": "log2(x+1)",
            "constant_feature_filter": "remove features with zero variance after training-set preprocessing",
            "scaling": "training mean and population SD for elastic net and PLS-DA; no scaling for random forest",
            "qc_in_classification": False,
        },
        "models": {
            name: model_settings(name, SEED)
            for name in ["elastic_net", "pls_da", "random_forest"]
        },
        "random_cv_model": "same fixed elastic-net classifier",
        "panel_ablation_conditions": PANEL_CONDITIONS,
        "panel_ablation_model": "fixed two-component PLS-DA, same LOTO splits and fold-local preprocessing",
        "fits": {
            "full_feature_loto": 15,
            "random_cv_baseline": 5,
            "fixed_panel_ablation": 20,
            "total": 40,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "verification_run_timestamp_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "analysis_script_sha256": {
            path.name: sha256(path)
            for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
        "input_file_sha256": {
            path.name: sha256(path)
            for path in sorted((Path(__file__).resolve().parents[1] / "data_input").iterdir())
            if path.is_file()
        },
        "runtime_note": "The manuscript reports Python 3.13.2, NumPy 2.0.2, pandas 3.0.3, SciPy 1.15.1, and scikit-learn 1.9.0. This verification run used the software versions listed above; the environment difference is recorded explicitly.",
    }
    (RESULTS / "model_settings_and_run_manifest.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_log(checks, warnings_log, start):
    all_pass = all(check["pass"] for check in checks)
    run_seconds = time.time() - start
    log = (
        f"Run completed: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"Runtime seconds: {run_seconds:.2f}\n"
        f"Python: {platform.python_version()} | NumPy: {np.__version__} | pandas: {pd.__version__} | SciPy: {scipy.__version__} | scikit-learn: {sklearn.__version__}\n"
        f"Fits: 15 LOTO + 5 random-CV + 20 PLS panel ablation = 40 total\n"
        f"Captured estimator warnings: {len(warnings_log)}\n"
        f"Manuscript checks: {'PASS' if all_pass else 'FAIL'} ({sum(check['pass'] for check in checks)}/{len(checks)})\n"
    )
    (LOGS / "run_log.txt").write_text(log, encoding="utf-8")
    if not all_pass:
        failed = [check for check in checks if not check["pass"]]
        raise AssertionError(f"Manuscript verification checks failed: {failed}")
    print(log)
    print(json.dumps(checks, indent=2, ensure_ascii=False))


def main():
    ensure_dirs()
    start = time.time()
    (
        sm,
        feature_meta,
        combined,
        bio_meta,
        X,
        y,
        tissue,
        sample_ids,
        feature_ids,
    ) = load_data()
    species = bio_meta["species"].to_numpy(dtype=str)

    audit, qc_df = run_data_audit(sm, feature_meta, combined, bio_meta, X)
    write_audit(sm, feature_meta, audit, qc_df)

    loto_pred, loto_metrics, loto_splits, loto_rank, warn_loto = run_loto(
        X, y, tissue, sample_ids, species, feature_ids
    )
    random_pred, random_metrics, random_splits, random_rank, warn_random = run_random_cv(
        X, y, tissue, sample_ids, species, feature_ids
    )
    panel_pred, panel_metrics, panel_splits, warn_panel = run_panel_ablation(
        X, y, tissue, sample_ids, species, feature_meta, feature_ids
    )

    write_results(
        loto_pred,
        loto_metrics,
        loto_splits,
        loto_rank,
        random_pred,
        random_metrics,
        random_splits,
        random_rank,
        panel_pred,
        panel_metrics,
        panel_splits,
    )
    marker_rows = write_marker_metadata(feature_meta, combined, bio_meta, y)

    warnings_log = warn_loto + warn_random + warn_panel
    checks = verify_expected_outputs(
        loto_metrics, random_metrics, panel_metrics, loto_rank, random_rank
    )
    checks = build_checks(
        checks,
        loto_splits,
        random_splits,
        panel_splits,
        loto_pred,
        random_pred,
        panel_pred,
        audit,
        marker_rows,
        qc_df,
        random_metrics,
        panel_metrics,
    )
    write_checks(checks, warnings_log)
    write_manifest()

    # Figures use the in-memory metrics and observed raw marker values; no fitting occurs here.
    make_figures(
        loto_pred,
        loto_metrics,
        random_metrics,
        panel_pred,
        panel_metrics,
        feature_meta,
        combined,
        bio_meta,
        feature_ids,
    )
    write_log(checks, warnings_log, start)


if __name__ == "__main__":
    main()
