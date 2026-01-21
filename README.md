# Side Information in Drug–Target Interaction Prediction

This repository contains the implementation for my Master of Science (Technology) Thesis in Biomedical Engineering and Health Technology at the University of Turku. The project focuses on developing a machine learning pipeline for predicting drug–target interactions (DTI) using factorization machines (FM).

## Project Background

In drug discovery, identifying new interactions experimentally is both costly and time-consuming. This model accelerates the process by predicting affinities (such as $K_d$ or $K_i$) non-experimentally by leveraging:

1. **Binary Identifiers** (One-Hot Encoding for drugs and targets)
2. **Side Information** (Chemical similarity matrices and genomic descriptors)

**Research Question:** Does incorporating side information improve drug–target interaction prediction compared to using only factorization of an interaction matrix?

## Key Features

- **Extensive Dataset Support:** Built-in loading functions for seven standardized benchmark datasets (Davis, Metz, Merget, KIBA, GPCR, IC, and E)
- **Flexible Feature Engineering:** Options to use binary identifiers only, side information only, or a concatenated combination of both
- **Robust Validation:** Implements two-tier Nested Cross-Validation (NCV) to ensure reliable hyperparameter tuning and performance estimation
- **Strict Data Integrity:** Ensures all drugs and targets in the test set are present in the training set, preventing unintended "cold-start" scenarios
- **Performance Metrics:** Evaluation based on C-index (Concordance Index), measuring the model's ability to correctly rank interaction strengths

## Technical Implementation

The model is implemented in Python, utilizing the `libFM` library, which is highly efficient for modeling interactions in sparse datasets.

**Dependencies:**
* `Python ≥ 3.9`
* `pywFM` (Python wrapper for libFM)
* `rlscore` (For C-index calculation)
* `scikit-learn` (For preprocessing and K-Fold splitting)
* `NumPy`

## Key Findings

The study revealed that **side information does not systematically improve predictions** across all datasets. Key results:

| Dataset | C-index (Binary Only) | C-index (Both) | Improvement |
|---------|----------------------|----------------|-------------|
| Davis KI | 0.855 | 0.856 | +0.05% |
| Metz KI | 0.830 | 0.815 | -1.76% |
| KIBA | 0.831 | 0.822 | -1.08% |
| Merget KW | 0.851 | 0.847 | -0.43% |
| **GPCR** | 0.894 | **0.921** | **+3.01%** ✓ |
| **Ion Channels** | 0.959 | **0.967** | **+0.76%** ✓ |
| Enzymes | 0.955 | 0.947 | -0.87% |

**Conclusion:** Binary identifiers alone achieve strong performance (C-index ≥ 0.83). Side information provides significant gains only for binary interaction datasets (GPCR, Ion Channels), while real-valued datasets show no systematic benefit.

## Configuration

Model hyperparameters in `settings.py`:

```python
task = 'regression'
num_iter = 100
learning_method = 'als'
r2_regularization = 1.0
candidate_k2_values = [4, 8, 16]
num_cv_folds_k = 3
num_cv_folds_split = 3
```

## Contact

**MSc (Tech) Mikko Happonen**  
Supervisors: Assoc. Prof. Antti Airola, MSc Riikka Numminen

---

**Note:** This implementation demonstrates that simpler models with binary identifiers often suffice for DTI prediction. The thesis explores why more complex side information doesn't consistently improve performance and discusses future directions for more sophisticated approaches.
