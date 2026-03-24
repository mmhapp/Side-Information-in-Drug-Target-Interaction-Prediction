# Side Information in Drug–Target Interaction Prediction

This repository contains the implementation for my Master of Science (Technology) Thesis in Biomedical Engineering at the University of Turku. The project focuses on developing a machine learning (ML) pipeline for predicting drug–target interactions (DTI) using factorisation machine (FM). The thesis is available to read in [UTUPub](https://www.utupub.fi/handle/10024/181501).

Thesis presents a structured and scientific pipeline and provides clear, data-driven answers to its core research question. It follows a scientific process, provides a transparent analysis of its findings, and correctly identifies the boundaries of its chosen model. A key takeaway is that more data doesn't always mean a better model. It is possible to model interactions between two entities without needing a massive neural network. Also, simple binary identifiers sometimes outperform complex chemical descriptors, teaching the importance of well-designed feature engineering. Thesis highlights the handling of high-dimensional sparse data, a problem common in in recommendation engines. Unlike many machine learning black-box models, presented approach provides interpretability by explicitly modeling pairwise interactions through latent vectors, allowing for a clearer understanding of feature representations.

---

## Background

In drug discovery, identifying new interactions experimentally is both costly and time-consuming. Artifical Intelligence can accelerate the process. Presented model predicts affinities $K_d$ non-experimentally with the help of binary identifiers and side information.

1. *Binary Identifiers* (One-Hot Encodings of drugs and targets derived from drug–target interaction matrix $\mathbf{Y}$)
2. *Side Information* (Drug–drug similarity matrix $\mathbf{X}_D$ and target–target similarity matrix $\mathbf{X}_T$ representing chemical structure)

Seven benchmark datasets are supported. The datasets have different amounts of drugs and targets, as well as different types of drugs and targets. However, each dataset includes an interaction matrix $\mathbf{Y}$ for labels and binary identifiers, as well as drug and target similarity matrices $\mathbf{X}_D$ and $\mathbf{X}_T$ for side information. 

*Research Question:* Does the incorporation of side information improve drug–target interaction prediction compared to using only factorisation of an interaction matrix?

## Methodology

The fundamental idea behind factorisation machine is to model interactions between features using latent factors, which is highly effective for processing sparse data, such as the drug–target interactions. Factorisation machine can be presented as:

$$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

Model parameters: $w_0$ is the global bias, $w_i$ represents the strength of the $i$-th variable, $x_i$, and $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ models the interaction between the $i$-th and $j$-th variable by calculating the dot product of their latent vectors of size $d$. During learning, alternative least squares (ALS) is used for optimisation and L2 is used for regularisation. Also, following methodologies are present:

- *Feature engineering*: Options of using binary identifiers only, side information only, or a concatenation of both in the input data
- *Cold-Start Problem Prevention*: Ensures predictions are made only for known drugs and targets to ensure robustness of the model
- *Two-tier Nested Cross-Validation*: Used for hyperparameter tuning and train-test split and to avoid data leakage
- *Evaluation*: Based on C-index metric, measuring the model's ability to correctly rank interaction strengths

The complete representation of a drug–target pair is constructed by concatenating binary identifiers and side information into a single input vector. This modular structure allows the model to toggle between using only identifiers (BI), only side information (SI), or the full concatenated feature set (BI and SI). Drug–target pair can be formulated into a vector as:

$$\mathbf{x}^{(i,j)} = \left( \mathbf{e}_{d}^{(i)}, \mathbf{e}_{t}^{(j)}, \mathbf{x}_{d}^{(i)}, \mathbf{x}_{t}^{(j)} \right)$$

- $\mathbf{e}_{d}^{(i)}$ represents the binary identifiers (One-Hot Encoding) of drug $d_i$.
- $\mathbf{e}_{t}^{(j)}$ represents the binary identifiers (One-Hot Encoding) of target $t_j$.
- $\mathbf{x}_{d}^{(i)}$ represents the side information (similarity profile) of drug $d_i$.
- $\mathbf{x}_{t}^{(j)}$ represents the side information (similarity profile) of target $t_j$.

## Findings

The study revealed that side information does not systematically improve predictions across all datasets.

Table 5.2: Improvements of C-index values when using the proposed DTI prediction models in the scenarios with using only SI, and with using both BI and SI, relative to the scenario with using only BI.
| Dataset | Improvement on using SI | Improvement on using BI and SI |
| :--- | :--- | :--- |
| Davis et al. 2011 | -1.74 % | 0.05 % |
| KI (Metz et al. 2011) | -9.28 % | -1.76 % |
| KIBA (Tang et al. 2014) | -10.94 % | -1.08 % |
| KW (Merget et al. 2017) | -5.97 % | -0.43 % |
| GPCR (Yamanishi et al. 2008) | 2.19 % | 3.01 % ✓ |
| IC (Yamanishi et al. 2008) | 0.15 % | 0.76 % ✓ |
| E (Yamanishi et al. 2008) | -2.93 % | -0.87 % |

---

## Conclusions

- Binary identifiers alone achieve strong performance (C-index ≥ 0.83) and can be used as a fully-functioning predictive model.
- Side information provides significant improvements only for binary datasets (GPCR, Ion Channels) because these are less complex.
- Factorisation machine lacks capacity for complex non-linear interactions which causes the low impact of side information.

## Directions

- For greater improvements, more powerful machine learning models (graph, network, or deep learning) are needed.
- For greater improvements, more extensive data types (3D, 4D, SMILES, sequences, or physics-based features) are needed.

## Limitations

- Predictions made only inside the interaction matrix $\mathbf{Y}$ to prevent the cold-start problem.
- Focus in evaluating the performance of the model is solely in C-index metric since it evaluates ranking.

*Note:* In addition to improving this method by utilising different models or data types, it is also possible to do predictions outside the $\mathbf{Y}$ and thus move to Cold-Start or even to Out-of-Distribution scenarios to predict interactions for novel chemical scaffolds. In this scenario, it is a major challenge to maintain robustness and generalisability of the model. Current model supprots a transductive setting, predicting missing links between known entities, and not a inductive setting for predicting for entirely new molecules.

---

## Dependencies

The project is developed with Python ≥ 3.9. The following libraries and tools are required to run the model.

Core libraries for machine learning, scientific operations, and data processing:
* `NumPy` `Pandas` for matrix operations and structured data handling.
* `Scikit-learn` utilised for KFold cross-validation, MinMaxScaler preprocessing, and evaluation metrics (MAE, MSE, R2).
* `h5py` for loading and managing large-scale biological datasets stored in HDF5 format.
* `tqdm` provides progress bars for long-running Nested Cross-Validation loops.
* `Matplotlib` `Seaborn` for generating performance visualizations and statistical analysis within statistics.ipynb.

To install these, one can use the provided requirements.txt: `pip install -r requirements.txt`

Factorisation machine implementation:
* `LibFM` The core C++ engine for factorization machines, must be compiled and available in system's PATH ([Source](https://github.com/srendle/libfm)).
* `pywFM` A Python wrapper used to interface with the LibFM executable ([Source](https://github.com/jfloff/pywFM?tab=readme-ov-file)).
* `RLScore` Used for the C-index calculation, which is the primary metric for ranking-based DTI prediction ([Source](https://github.com/aatapa/RLScore)).

## Configuration

Model hyperparameters in `settings.py`:

```python
task = 'regression'
num_iter = 100
learning_method = 'als'
r2_regularisation = 1.0
candidate_k2_values = [4, 8, 16]
num_cv_folds_k = 3
num_cv_folds_split = 3
```

## Usage

Main code is in `fm.py`, and the instructions in `descriptions.txt` (`knn.py` is a basic version of model workflow but with kNN instead of FM). The model is implemented in Python, utilising the `libFM` library. Execute DTI predictions with:

```bash
python fm.py [dataset] [feature_option]
```

- `[dataset]`: Choose from `davis`, `metz`, `tang`, `merget`, `gpcr`, `ic`, `e`
- `[feature_option]`:
  - `bi` - Binary identifiers only (one-hot encoding)
  - `si` - Side information only (similarity matrices)
  - `both` - Concatenated combination of both

## Examples

Run the model:
```bash
# Binary identifiers only on Davis dataset
python fm.py davis bi

# Both features on GPCR dataset
python fm.py gpcr both

# Side information only on Ion Channels dataset
python fm.py ic si
```

Run all experiments automatically:
```bash
python fmloop.py
```

Save predicted interaction values (Output: `Results/predictions_[file_name].csv`):
```bash
python fmpredictions.py [dataset] [feature_option] [file_name]
```

Save trained model parameters (Output: `Results/weights_[file_name].json`):
```bash
python fmweights.py [dataset] [feature_option] [file_name]
```

## Project

```.
├── data/               # HDF5 benchmark datasets
├── Results/            # Output directory for weights and predictions
├── fm.py               # Main execution script
├── fmloop.py           # Automated experiment runner
├── settings.py         # Hyperparameter configurations
└── statistics.ipynb    # Visualization and analysis
```

## Pipeline

1. *Dataset loading*: Import $\mathbf{Y}$ (drug–target interaction matrix), $\mathbf{X}_D$ (drug similarities), $\mathbf{X}_T$ (target similarities)
2. *CV to replace train-test split of the dataset*: 3-fold CV to replace train-test split
3. *Feature concatenation*: Merge drug and target features with only binary identifiers, side information, or both
4. *CV for Model Validation*: Hyperparameter tuning $k∈{4,8,16}$
5. *Model training*: ALS-based training of the FM model
6. *Model evaluation*: Calculate C-index to measure ranking performance

## Analysis

Use the Jupyter notebook for visualizations:
```bash
jupyter notebook statistics.ipynb
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
