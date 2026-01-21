# Number of folds when performing cross-validation to validate the model 
num_cv_folds_k = 3

# Number of folds when performing cross-validation to replace train-test split
num_cv_folds_split = 3

# Candidates for hyperparameter k2
candidate_k2_values = [4, 8, 16]

# Setting of factorisation machine
task = 'regression'
num_iter = 100
learning_method = 'als'
r2_regularization = 1

# Used datasets and feature options only for fmloop.py
datasets = ["davis", "metz", "kiba", "merget", "GPCR", "IC", "E"]
feature_options = ["bi", "si", "both"]