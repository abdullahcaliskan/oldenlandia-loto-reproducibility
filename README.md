# Oldenlandia LOTO reproducibility repository

This repository contains the fixed-setting analysis code associated with the accepted IEEE Access article:

Abdullah Caliskan, “Machine Learning for Medicinal-Plant Authentication Across Unseen Tissues Using Compact LC-MS Fingerprints,” *IEEE Access*.

## Study data

The analysis uses the public MetaboLights study **MTBLS13314**, which contains LC–MS abundance matrices for *Oldenlandia diffusa* and *O. corymbosa*. The deposited data comprise 60 biological samples (two species × five tissues × six replicates), six pooled QC samples, and 1,343 assay features (629 negative-ion and 714 positive-ion). QC samples are included in the dataset audit and excluded from classifier fitting.

The repository does not redistribute the study data. Download the three files listed in [`data_input/README.md`](data_input/README.md) from the official [MetaboLights study record](https://www.ebi.ac.uk/metabolights/MTBLS13314) and place them in `data_input/`.

## Analysis implemented

1. **Fold-local preprocessing:** missing values are replaced by one-half the smallest positive abundance in the relevant training subset (zero if no positive value exists); abundances are transformed using log₂(x+1); zero-variance features are removed using the training subset; elastic net and PLS-DA use training means and population standard deviations, while random forest uses unscaled transformed values.
2. **Five-fold Leave-One-Tissue-Out (LOTO):** one complete tissue is held out per fold (48 training and 12 test samples). Each biological sample receives one out-of-fold prediction from each classifier.
3. **Fixed classifiers:** elastic-net logistic regression (`SGDClassifier`, logistic loss, α=0.001, l1 ratio 0.15, optimal learning-rate schedule, maximum 10,000 iterations, tolerance 1e-4); PLS-DA (`PLSRegression`, two components, scaling disabled after external standardization, maximum 500 iterations, tolerance 1e-6); and random forest (60 trees, square-root feature subsampling, minimum leaf size 1, Gini criterion, no depth limit or class weighting, bootstrap sampling).
4. **Random-CV baseline:** one shuffled, species-stratified five-fold random CV using the same fixed elastic-net classifier. This is a secondary comparison, not the primary validation.
5. **Descriptive feature ranking:** within each training fold, features are ranked by absolute standardized species mean difference. These rankings do not select features for the full-feature classifiers or the fixed-panel experiment.
6. **Retrospective fixed-panel ablation:** four pre-specified PLS-DA conditions (five markers, minus either missingness-heavy marker, and minus both), using the same LOTO splits and fold-local preprocessing.

There is **no hyperparameter search, inner tuning loop, repeated CV, bootstrap feature selection, or bootstrap uncertainty analysis** in this fixed-setting analysis.

Deposited metabolite names can contain encoding artefacts (for example, `Lyoniresinol-3¦Á-O-glucoside`). These deposited strings are retained unchanged in feature IDs; display names in the figures and tables follow the article.

## Run the analysis

The repository was verified with Python **3.13.5** (Python 3.13 is recommended). Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

The wrapper scripts below are the supported way to run the complete pipeline. A full run takes about 10 seconds in the verification environment.

**Linux/macOS**

```bash
bash run_all.sh
```

**Windows**

```bat
run_all_windows.bat
```

The pipeline writes dataset-audit files, fold identifiers, sample-level out-of-fold predictions, metrics, marker metadata, a run manifest, verification checks, and the figures listed below to `results/`, `logs/`, and `figures/`. It also creates a marker-measurement-presence audit plot; this is a diagnostic output and **not** a supplementary figure submitted with the article.

| Output file | Article item |
|---|---|
| `fig1_overview.pdf` / `.png` | Fig. 1 |
| `fig2_loto_design.pdf` / `.png` | Fig. 2 |
| `fig3_performance.pdf` / `.png` | Fig. 3 |
| `fig4_plsda_scores.pdf` / `.png` | Fig. 4 |
| `fig5_panel_ablation.pdf` / `.png` | Fig. 5 |

These figures are regenerated from the same outputs as the published figures; the plotted values are identical, but layout and styling differ from the typeset article.

The pipeline performs 40 fixed estimator fits (15 full-feature LOTO, five random-CV baseline, and 20 fixed-panel LOTO fits) and runs built-in checks against the manuscript-reported results and split integrity. The verification run completed with **43/43 checks passing (34 existing and nine additional checks) and zero captured estimator warnings**.

## Reproducibility details

- Master seed: `20260828`.
- LOTO fold order: root, stem, leaf, flower, fruit.
- Decision threshold: 0.5; *O. diffusa* is the positive class.
- All preprocessing parameters are learned from the relevant training subset and applied unchanged to held-out samples.
- The run manifest records the actual software versions and SHA-256 checksums of the input files used in that run.

The verification run for this repository used Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, and Matplotlib 3.10.8. The accepted manuscript reports Python 3.13.2, NumPy 2.0.2, pandas 3.0.3, SciPy 1.15.1, scikit-learn 1.9.0, and Matplotlib 3.10.0. These environments are not identical; the run manifest records the environment actually used for verification.

## Interpretation and limitations

The LOTO results assess tissue-domain holdout within this one deposited study. They do not establish donor independence, laboratory/geographic/seasonal/instrument generalization, or independent biomarker validation. The five-marker panel originated in an earlier exploratory analysis of the same dataset; its ablation is retrospective, not independent biomarker validation. External biological validation remains necessary before routine authentication.

## Citation

If you use this code, cite the associated article:

Abdullah Caliskan, “Machine Learning for Medicinal-Plant Authentication Across Unseen Tissues Using Compact LC-MS Fingerprints,” *IEEE Access*.

## Repository structure

See [`MANIFEST.md`](MANIFEST.md) for the canonical file inventory. The analysis implementation is maintained in `src/`; the single supported entry point is `src/run_analysis.py`.
