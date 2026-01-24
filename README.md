# Side Information in Drug–Target Interaction Prediction

This repository contains the implementation for my Master of Science (Technology) Thesis in Biomedical Engineering and Health Technology at the University of Turku. The project focuses on developing a machine learning pipeline for predicting drug–target interactions (DTI) using factorization machine (FM). Whole thesis is available in [UTUPub](https://www.utupub.fi/handle/10024/181501). The fundamental idea behind factorisation machines is to model interactions between variables using latent factors, which is highly effective for processing sparse data, such as the drug–target pairs in this thesis. In drug discovery, the C-index is preferred for model evaluation because ranking candidates—identifying which one is superior to another—is more critical than determining absolute values.

## Project Background

In drug discovery, identifying new interactions experimentally is both costly and time-consuming. This model accelerates the process by predicting affinities (such as $K_d$ or $K_i$) non-experimentally by leveraging:

1. **Binary Identifiers** (One-Hot Encoding for drugs and targets)
2. **Side Information** (Chemical similarity matrices and genomic descriptors)

**Research Question:** Does incorporating side information improve drug–target interaction prediction compared to using only factorization of an interaction matrix?

## Key Features

- **Extensive Dataset Support:** Built-in loading functions for seven standardized benchmark datasets (Davis, Metz, Merget, KIBA, GPCR, IC, and E)
- **Flexible Feature Engineering:** Options to use binary identifiers only, side information only, or a concatenated combination of both
- **Robust Validation:** Implements two-tier Nested Cross-Validation (NCV) to ensure reliable hyperparameter tuning and performance estimation
- **Strict Data Integrity:** Ensures all drugs and targets in the test set are present in the training set, preventing unintended cold-start scenarios
- **Performance Metrics:** Evaluation based on C-index (Concordance Index), measuring the model's ability to correctly rank interaction strengths

## Dependencies

* `Python ≥ 3.9`
* `pywFM` (Python wrapper for libFM)
* `rlscore` (For C-index calculation)
* `scikit-learn` (For preprocessing and K-Fold splitting)
* `NumPy`

## Key Findings

The study revealed that **side information does not systematically improve predictions** across all datasets.

| Dataset | C-index (Binary Only) | C-index (Both) | Improvement |
|---------|----------------------|----------------|-------------|
| Davis KI | 0.855 | 0.856 | +0.05% |
| Metz KI | 0.830 | 0.815 | -1.76% |
| KIBA | 0.831 | 0.822 | -1.08% |
| Merget KW | 0.851 | 0.847 | -0.43% |
| **GPCR** | 0.894 | **0.921** | **+3.01%** ✓ |
| **Ion Channels** | 0.959 | **0.967** | **+0.76%** ✓ |
| Enzymes | 0.955 | 0.947 | -0.87% |

**Conclusion:** Binary identifiers alone achieve strong performance (C-index ≥ 0.83). Side information provides significant gains only for binary interaction datasets (GPCR, Ion Channels), while real-valued datasets show no systematic benefit. For bigger improvement, one needs to have more powerful model (graph or network based).

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

## Methodology

The core of this project is based on **Factorization Machines (FM)** that capture second-order interactions between features using latent factors. The model equation is defined as:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$,

where $w_0$ is the global bias, $w_i$ represents the strength of the $i$-th variable, and $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ models the interaction between the $i$-th and $j$-th variable by calculating the dot product of their latent vectors of size $k$.

---

**Note:** This implementation demonstrates that simpler models with binary identifiers often suffice for DTI prediction. The thesis explores why more complex side information doesn't consistently improve performance and discusses future directions for more sophisticated approaches.

---

# Technical Implementation

Main code is in `fm.py`, and the instructions in `descriptions.txt`. The model is implemented in Python, utilising the `libFM` library.

## Running the Model (`fm.py`)

The main script `fm.py` executes the DTI prediction pipeline with Nested Cross-Validation.

**Command format:**

```bash
python fm.py [dataset] [feature_option]

```

* **`[dataset]`**: Choose from `davis`, `metz`, `tang`, `merget`, `gpcr`, `ic`, or `e`.
* **`[feature_option]`**:
* `bi`: Only binary identifiers (one-hot encoding) are used.
* `si`: Only side information (chemical/genomic similarities) is used.
* `both`: Both binary identifiers and side information are concatenated.

## Batch Execution and Specialized Scripts

* **Run all experiments:** To run the model across all seven datasets and all three feature options automatically:
```bash
python fmloop.py

```

* **Export predictions:** To save predicted interaction values to a CSV file (`Results/predictions_[file_name].csv`):
```bash
python fmpredictions.py [dataset] [feature_option] [file_name]

```

* **Save model weights:** To export trained model weights to a JSON file (`Results/weights_[file_name].json`):
```bash
python fmweights.py [dataset] [feature_option] [file_name]

```

## Complete Pipeline

1. **Data Loading:** Loads drug-drug similarities (), target-target similarities (), and the interaction matrix ().
2. **Scaling:** Features are normalized (e.g.,  scaled by 100,  converted to  values) to ensure optimal gradient descent/ALS performance.
3. **Cross-Validation:** Uses `KFold` (typically 10 folds) to ensure every drug-target pair is used for both training and testing.
4. **Feature Concatenation:** Merges drug and target features into a single sparse matrix , with optional one-hot encoding for biological entities.
5. **Validation & Training:** Performs hyperparameter tuning for  and trains the final FM model using Alternating Least Squares (ALS).
6. **Evaluation:** Calculates the **C-index** to measure the model's ability to rank interaction strengths correctly.

## Results

* **`Results/results.txt`**: Raw output of the model runs.
* **`Results/statistics.ipynb`**: Jupyter Notebook for visualizing and analyzing the performance metrics.
* **`Results/results.py`**: A utility script that parses `results.csv` and generates formatted LaTeX tables for thesis reporting.
