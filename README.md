# Side Information in Drug–Target Interaction Prediction

A machine learning (ML) pipeline for predicting drug–target interactions (DTI) using factorisation machines (FM). Used multimodal drug–target features and pairwise interaction modelling via latent vectors, and achieved **C-index ≥ 0.83** across all 7 benchmark datasets.

Implementation is done as a part of Master of Science (Technology) Thesis in Biomedical Engineering at the University of Turku. It presents a structured and scientific pipeline and provides clear, data-driven answers to its research question. It follows a scientific process, provides a transparent analysis of its findings, and correctly identifies the boundaries of chosen model. Read the full thesis on [UTUPub](https://www.utupub.fi/handle/10024/181501).

---

## Background

In drug discovery, identifying new interactions experimentally is both costly and time-consuming. Artifical Intelligence (AI) can accelerate the process. Presented model predicts affinities $K_d$ non-experimentally with the help of binary identifiers and side information.

*Binary Identifiers (BI)*: One-Hot Encodings of drugs and targets derived from drug–target interaction matrix $\mathbf{Y}$

*Side Information (SI)*: Drug–drug similarity matrix $\mathbf{X}_D$ and target–target similarity matrix $\mathbf{X}_T$ representing chemical structure

Seven benchmark datasets, each with different amounts and types of drugs and targets, are supported. Each dataset includes an interaction matrix $\mathbf{Y}$ for labels and binary identifiers, as well as drug and target similarity matrices $\mathbf{X}_D$ and $\mathbf{X}_T$ for side information. 

*Research Question:* Does the incorporation of SI  improve DTI prediction compared to using only factorisation of an interaction matrix?

## Model

The fundamental idea behind factorisation machine is to model interactions between features using latent factors, which is highly effective method for processing sparse data, such as the drug–target interactions. Mathematically, factorisation machine can be presented as

$\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^d w_i x_i + \sum_{i=1}^d \sum_{j=i+1}^d \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j.$

$w_0$ is the global bias.

$w_i$ represents the strength of the $i$-th variable, $x_i$.

$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ models the interaction between the $i$-th and $j$-th variable by calculating the dot product of their latent vectors of size $d$.

During learning, alternative least squares (ALS) is used for optimisation and L2 is used for regularisation.

## Data

The complete drug–target pair is constructed by concatenating BIs and SI into a single input vector. This modular structure is based on feature engineering – using only BI, only SI, or the full concatenated feature set (BI and SI). Drug–target pair can be presented as

$\mathbf{x}^{\left(i{,}j\right)}=\left(\mathbf{e}_d^{\left(i\right)}{,}\ \mathbf{e}_t^{\left(j\right)}{,}\ \mathbf{x}_d^{\left(i\right)}{,}\ \mathbf{x}_t^{\left(j\right)}\right).$

$\mathbf{e}_{d}^{(i)}$ represents the binary identifiers (One-Hot Encoding) of drug $d_i$.

$\mathbf{e}_{t}^{(j)}$ represents the binary identifiers (One-Hot Encoding) of target $t_j$.

$\mathbf{x}_{d}^{(i)}$ represents the side information (similarity profile) of drug $d_i$.

$\mathbf{x}_{t}^{(j)}$ represents the side information (similarity profile) of target $t_j$.

## Findings

The study revealed that using SI or BI and SI does not systematically improve predictions across all datasets compared to using only BI.

| Dataset | Improvement on using SI | Improvement on using BI and SI |
| :--- | :--- | :--- |
| Davis et al. 2011 | -1.74 % | 0.05 % |
| KI (Metz et al. 2011) | -9.28 % | -1.76 % |
| KIBA (Tang et al. 2014) | -10.94 % | -1.08 % |
| KW (Merget et al. 2017) | -5.97 % | -0.43 % |
| GPCR (Yamanishi et al. 2008) | 2.19 % | 3.01 % ✓ |
| IC (Yamanishi et al. 2008) | 0.15 % | 0.76 % ✓ |
| E (Yamanishi et al. 2008) | -2.93 % | -0.87 % |

## Discussion

*Conclusions*: Binary identifiers alone achieve strong performance (C-index ≥ 0.83) and can be used as a fully-functioning predictive model. Side information provides significant improvements only for binary datasets (GPCR, Ion Channels) because these are less complex. Factorisation machine lacks capacity for complex non-linear interactions which causes the low impact of side information.

*Directions*: For greater improvements, more powerful machine learning models (graph, network, or deep learning) are needed. For greater improvements, more extensive data types (3D, 4D, SMILES, sequences, or physics-based features) are needed.

*Limitations*: Predictions made only inside the interaction matrix $\mathbf{Y}$ to prevent the cold-start problem and to ensure robustness of the model.. Focus in evaluating the performance of the model is solely in C-index metric since it evaluates ranking.

*Note*: In addition to improving this method by utilising different models or data types, it is also possible to do predictions outside the $\mathbf{Y}$ and thus move to Cold-Start or even to Out-of-Distribution scenarios to predict interactions for novel chemical scaffolds. In this scenario, it is a major challenge to maintain robustness and generalisability of the model. Current model supprots a transductive setting, predicting missing links between known entities, and not a inductive setting for predicting for entirely new molecules.

## Pipeline

*Dataset loading*: Import $\mathbf{Y}$ (drug–target interaction matrix), $\mathbf{X}_D$ (drug similarities), $\mathbf{X}_T$ (target similarities).

*CV to replace train-test split of the dataset*: 3-fold CV to replace train-test split.

*Feature concatenation*: Merge drug and target features with only binary identifiers, side information, or both.

*CV for Model Validation*: Hyperparameter tuning $k∈{4,8,16}$ (completes a two-tier nested cross-validation).

*Model training*: ALS-based training of the FM model.

*Model evaluation*: Calculate C-index to measure ranking performance (measuring the model's ability to correctly rank interaction strengths).

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
├── data/               # Benchmark datasets
├── Results/            # Output directory for weights and predictions
├── fm.py               # Main execution script
├── fmloop.py           # Automated experiment runner
├── settings.py         # Hyperparameter configurations
└── statistics.ipynb    # Visualisation and analysis
```

## License
 
MIT — see [LICENSE](LICENSE) for details.
