# Data attribution and license

The original software is distributed under the MIT license in `LICENSE`.
The public datasets in `data/` are redistributed under the Creative Commons
Attribution 4.0 International license, as stated by the UCI Machine Learning
Repository: https://creativecommons.org/licenses/by/4.0/.

Dataset creators retain ownership. The original downloaded archives are preserved
as `uci_*.zip`; SHA-256 checksums and source URLs appear in `data/provenance.json`.
The `.npz` tables are conversions of these files. Identifiers are removed from the
breast diagnostic table; class labels are encoded as documented in `README.md`;
wine quality is thresholded at 6. Numeric features retain their original values.
Standardization is fitted within each experiment on its training subset.

| Dataset | Attribution | Persistent identifier |
|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | W. Wolberg, O. Mangasarian, N. Street, W. Street (1993) | https://doi.org/10.24432/C5DW2B |
| Ionosphere | V. Sigillito, S. Wing, L. Hutton, K. Baker (1989) | https://doi.org/10.24432/C5W01B |
| Connectionist Bench (Sonar, Mines vs. Rocks) | T. Sejnowski, R. Gorman (1988) | https://doi.org/10.24432/C5T01Q |
| Banknote Authentication | V. Lohweg (2012) | https://doi.org/10.24432/C55P57 |
| Spambase | M. Hopkins, E. Reeber, G. Forman, J. Suermondt (1999) | https://doi.org/10.24432/C53G6X |
| Wine Quality (red and white) | P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009) | https://doi.org/10.24432/C56S3T |
| MAGIC Gamma Telescope | R. Bock (2004) | https://doi.org/10.24432/C52C8B |

The license permits sharing and adaptation with attribution, a license link, and
an indication of changes. The transformations above describe the changes used here.
