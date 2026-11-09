# Input data

The analysis uses MetaboLights study **MTBLS13314**.

Download the following deposited files from the study and place them in this directory:

- `s_MTBLS13314.txt`
- `m_MTBLS13314_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv`
- `m_MTBLS13314_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv`

These are the three files read by `src/run_analysis.py`. The analysis uses the deposited abundance matrices and sample metadata; it does not reprocess raw instrument files. Six pooled QC samples are included in the audit but excluded from biological classification.

The pipeline was verified with files matching these SHA-256 checksums:

