# Side Information in Drug–Target Interaction Prediction

This repository contains the implementation for my **Master of Science (Technology) Thesis** in Biomedical Engineering and Health Technology at the University of Turku. The project focuses on developing a machine learning pipeline for predicting drug–target interactions (DTI) using **Factorization Machines**.

### 🔍 Project Background

In drug discovery, identifying new interactions experimentally is both costly and time-consuming. This model accelerates the process by predicting affinities (such as  or ) non-experimentally by leveraging:

1. **Binary Identifiers** (One-Hot Encoding for drugs and targets).
2. **Side Information**, such as chemical similarity matrices and genomic descriptors.

### 🛠️ Key Features

* **Extensive Dataset Support:** Built-in loading functions for seven standardized benchmark datasets (*Davis, Metz, Merget, KIBA, GPCR, IC, and E*).
* **Flexible Feature Engineering:** Options to use binary identifiers only, side information only, or a concatenated combination of both.
* **Robust Validation:** Implements a two-tier **Nested Cross-Validation** (NCV) to ensure reliable hyperparameter tuning and performance estimation.
* **Strict Data Integrity:** Includes logic to ensure that all drugs and targets in the test set are present in the training set, preventing unintended "cold-start" scenarios during validation.
* **Performance Metrics:** Evaluation is based on the **C-index** (Concordance Index), measuring the model's ability to correctly rank the interaction strengths.

### 💻 Technical Implementation

The model is implemented in Python, utilizing the `libFM` library, which is highly efficient for modeling interactions in sparse datasets.

**Dependencies:**

* `Python ≥ 3.9`
* `pywFM` (Python wrapper for libFM)
* `rlscore` (For C-index calculation)
* `scikit-learn` (For preprocessing and K-Fold splitting)
* `NumPy`

### 🚀 Usage

The model can be executed via the command line by specifying the dataset and the desired feature option:

```bash
# Syntax: python dti_model.py [dataset] [feature_option]
# feature_option: bi (binary), si (side info), both (combined)

python dti_model.py davis both

```

### 📊 Results from the Thesis

The study demonstrated that integrating side information with binary identifiers significantly enhances prediction performance, particularly in sparse data scenarios. The pipeline automatically optimizes the factorization degree () in each fold to ensure peak model performance.
