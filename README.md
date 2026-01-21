# Side-Information-in-Drug-Target-Interaction-Prediction

This repository contains the implementation of a Drug–Target Interaction prediction pipeline based on Factorisation Machine, as developed for an MSc (Tech) thesis in Biomedical Engineering and Health Technology at the University of Turku. The pipeline uses both binary identifiers and side information to predict interactions between drugs and targets across multiple benchmark datasets.

## Overview
The goal of this project is to predict binding affinities or interactions between drugs and targets using a hybrid feature representation and Factorization Machines. The pipeline supports:
Multiple DTI datasets: Davis, Metz, KIBA, Merget, GPCR, IC, E.
Feature options:
Binary identifiers (one-hot encoding of drugs and targets)
Side-information (drug and target similarity matrices)
Combined features (both binary identifiers and side-information)
Cross-validation for hyperparameter tuning and model evaluation
C-index evaluation metric for ranking performance

project-root/
| Datasets
| LibFM
| Models
|—– fm.py
|—– fmloop.py
| RLScore


## Dependencies
The pipeline requires:
Python ≥ 3.9
NumPy
scikit-learn
rlscore
pywFM
A local installation of libFM
Install Python dependencies with:
pip install -r requirements.txt

## Settings
All hyperparameters and options are defined in settings.py:
task: "regression" or "classification" depending on dataset
num_iter: Number of iterations for FM training
k2: Latent factor dimension (selected via cross-validation)
learning_method: "sgd" or "als"
r2_regularization: Regularization parameter
candidate_k2_values: List of k2 values to test in cross-validation
num_cv_folds_k: Number of folds for hyperparameter tuning
num_cv_folds_split: Number of folds for train/test split simulation

## Running the Pipeline
Run the pipeline from the command line:
python src/main.py <dataset> <feature_option>
Where:
<dataset>: davis, metz, kiba, merget, gpcr, ic, e
<feature_option>:
bi → Binary identifiers only
si → Side-information only
both → Binary + Side-information
Example:
python src/main.py davis both
The script will:
Load the selected dataset
Preprocess features and scale similarity matrices
Split data using K-Fold cross-validation
Ensure all drugs and targets in the test set are present in the training set
Perform hyperparameter tuning to select the best k2
Train a Factorization Machine model
Evaluate using C-index and output fold-wise and overall performance

## Output
C-index values for each fold
Mean C-index across all folds
Optionally, predictions for each drug-target pair (can be extracted from model.predictions)
