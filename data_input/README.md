# Input data

The analysis uses MetaboLights study **MTBLS13314**.

Download the following deposited files from the study and place them in this directory:

- `s_MTBLS13314.txt`
- `m_MTBLS13314_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv`
- `m_MTBLS13314_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv`

These are the three files read by `src/run_analysis.py`. The analysis uses the deposited abundance matrices and sample metadata; it does not reprocess raw instrument files. Six pooled QC samples are included in the audit but excluded from biological classification.

The pipeline was verified with files matching these SHA-256 checksums:

```text
50e2ed8151625617bf8fa5ca95ac631a4a81c6c79baa72e39e92ad45010bb592  s_MTBLS13314.txt
ddb52362a44aba780d129b6606921a984431ba49e0ea84fd628d98724a26a5d6  m_MTBLS13314_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv
a1aae2d90d73d03edd79304d8dcc98f43fe21e9f24725ccf32a25d43f7165f90  m_MTBLS13314_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv
```

Check the downloaded files before running the pipeline.

**Windows PowerShell**

```powershell
Get-FileHash .\s_MTBLS13314.txt -Algorithm SHA256
Get-FileHash .\m_MTBLS13314_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv -Algorithm SHA256
Get-FileHash .\m_MTBLS13314_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv -Algorithm SHA256
```

**Linux/macOS**

```bash
sha256sum s_MTBLS13314.txt
sha256sum m_MTBLS13314_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv
sha256sum m_MTBLS13314_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv
```

Expected feature counts are 629 negative-ion and 714 positive-ion rows (1,343 combined). The source investigation description reports 1,342 metabolites; the analysis retains all 1,343 deposited feature rows, as stated in the article.

The raw dataset is not redistributed in this repository. Download it from the official MetaboLights record: https://www.ebi.ac.uk/metabolights/MTBLS13314
