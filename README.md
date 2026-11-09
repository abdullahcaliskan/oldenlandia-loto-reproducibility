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

