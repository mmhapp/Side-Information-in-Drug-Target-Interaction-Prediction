import numpy as np
from rlscore.measure import cindex
import pywFM
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import sys
import csv
import settings

np.random.seed(42)
os.environ["LIBFM_PATH"] = "/Users/mikkohapponen/Documents/Tiedostot/Opiskelu/DI/Diplomityö/Data/libfm/bin/"



"""
Function to load an incomplete data set introduced by Metz et al. (2011).
Returns the data matrices and lists of drug and target indices for the known pairs.
"""

def load_metz():

    Y = np.loadtxt("../Datasets/known_drug-target_interaction_affinities_pKi__Metz_et_al.2011.txt")
    XD = np.loadtxt("../Datasets/drug-drug_similarities_2D__Metz_et_al.2011.txt")
    XT = np.loadtxt("../Datasets/target-target_similarities_WS_normalized__Metz_et_al.2011.txt")
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]

    # Get OHE
    num_drugs = len(np.unique(drug_inds))
    num_targets = len(np.unique(target_inds))

    # Scaling of both XD and XT
    scaler = MinMaxScaler()
    XD = scaler.fit_transform(XD)
    XT = scaler.fit_transform(XT)

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32'), num_drugs, num_targets



"""
Function to load a complete data set introduced by Davis et al. (2011).
Returns the data matrices and lists of drug and target indices for the known pairs.
The matrix of drug similarities is multiplied by 100 in order to obtain the same
range as in the corresponding matrix of Metz data.
The returned continuous labels are natural logarithm of the Kd values so that the
range is again similar to the range of continuous labels in Metz data.
"""

def load_davis():

    Y = np.loadtxt("../Datasets/drug-target_interaction_affinities_Kd__Davis_et_al.2011.txt")
    XD = np.loadtxt("../Datasets/drug-drug_similarities_2D__Davis_et_al.2011.txt")
    XT = np.loadtxt("../Datasets/target-target_similarities_WS_normalized__Davis_et_al.2011.txt")
    XD = 100*XD
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    Y = -1*np.log10(Y/1e9)

    # Get OHE
    num_drugs = len(np.unique(drug_inds))
    num_targets = len(np.unique(target_inds))

    # Scaling of both XD and XT
    scaler = MinMaxScaler()
    XD = scaler.fit_transform(XD)
    XT = scaler.fit_transform(XT)

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32'), num_drugs, num_targets



"""
Function to load an incomplete data set introduced by Merget et al. (2017) and
updated by Cichonska et al (2018).
Returns the data matrices and lists of drug and target indices for the known pairs.
The matrices of drug similarities and target similaritied are multiplied by 100 in
order to obtain the same range as in the corresponding matrices of Metz data.
"""

def load_merget():

    Y = np.loadtxt("../Datasets/Merget_DTIs_2967com_226kin.txt")
    XD = np.loadtxt("../Datasets/Kd_Tanimoto-shortestpath.txt")
    XT = np.loadtxt("../Datasets/Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]

    # Get OHE
    num_drugs = len(np.unique(drug_inds))
    num_targets = len(np.unique(target_inds))

    # Scaling of both XD and XT
    scaler = MinMaxScaler()
    XD = scaler.fit_transform(XD)
    XT = scaler.fit_transform(XT)

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32'), num_drugs, num_targets



"""
Functions to load an incomplete data sets introduced by Tang et al. (2014) and Yamanishi et al. (2008).
Returns the data matrices and lists of drug and target indices for the known pairs.
The matrices of drug similarities and target similaritied are multiplied by 100 in
order to obtain the same range as in the corresponding matrices of Metz data.
Files kiba_binding_affinity_v2.txt and kiba_drug_sim.txt are slightly modifief in R
because their last columns were such that all values were "NA" and the numbers of
rows and columns did not match.
"""

def load_kiba():

    Y = np.loadtxt("../Datasets/kiba_binding_affinity_v2.txt")
    XD = np.loadtxt("../Datasets/kiba_drug_sim.txt")
    XT = np.loadtxt("../Datasets/kiba_target_sim.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]

    # Get OHE
    num_drugs = len(np.unique(drug_inds))
    num_targets = len(np.unique(target_inds))

    # Scaling of both XD and XT
    scaler = MinMaxScaler()
    XD = scaler.fit_transform(XD)
    XT = scaler.fit_transform(XT)

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32'), num_drugs, num_targets

def load_GPCR():

    Y = np.loadtxt("../Datasets/gpcr_admat_dgc.txt")
    XD = np.loadtxt("../Datasets/gpcr_simmat_dc.txt")
    XT = np.loadtxt("../Datasets/gpcr_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]

    # Get OHE
    num_drugs = len(np.unique(drug_inds))
    num_targets = len(np.unique(target_inds))

    # Scaling of both XD and XT
    scaler = MinMaxScaler()
    XD = scaler.fit_transform(XD)
    XT = scaler.fit_transform(XT)

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32'), num_drugs, num_targets

def load_IC():

    Y = np.loadtxt("../Datasets/ic_admat_dgc.txt")
    XD = np.loadtxt("../Datasets/ic_simmat_dc.txt")
    XT = np.loadtxt("../Datasets/ic_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]

    # Get OHE
    num_drugs = len(np.unique(drug_inds))
    num_targets = len(np.unique(target_inds))

    # Scaling of both XD and XT
    scaler = MinMaxScaler()
    XD = scaler.fit_transform(XD)
    XT = scaler.fit_transform(XT)

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32'), num_drugs, num_targets

def load_E():

    Y = np.loadtxt("../Datasets/e_admat_dgc.txt")
    XD = np.loadtxt("../Datasets/e_simmat_dc.txt")
    XT = np.loadtxt("../Datasets/e_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]

    # Get OHE
    num_drugs = len(np.unique(drug_inds))
    num_targets = len(np.unique(target_inds))

    # Scaling of both XD and XT
    scaler = MinMaxScaler()
    XD = scaler.fit_transform(XD)
    XT = scaler.fit_transform(XT)

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32'), num_drugs, num_targets



"""
Function to concatenate information to be used in the drug-target data.
A row in the matrix contains features related to a drug-target pair.
It includes binary identifiers, similarity matrix based side information, or both.
"""

def concatenate_features(X1, X2, inds1, inds2, drug_num_classes, target_num_classes, feature_option):

    features_X1 = X1[inds1,:]
    features_X2 = X2[inds2,:]

    one_hot_drugs = np.eye(drug_num_classes)[inds1]
    one_hot_targets =  np.eye(target_num_classes)[inds2]

    # Input data has one-hot encoding and if side information is used, similarity matrices
    if feature_option == 1:
        X = np.hstack((one_hot_drugs, one_hot_targets))
    elif feature_option == 2:
        X = np.hstack((features_X1, features_X2))
    elif feature_option == 3:
        X = np.hstack((one_hot_drugs, one_hot_targets, features_X1, features_X2))

    return X


"""
Function to ensure that all drugs and targets in the test set are also in the train set.
If a drug or target in the test set is missing from the train set, moves a corresponding
sample from test set to train set during cross-validation process.
"""

def ensure_drug_target_presence(drug_inds, target_inds, training_inds, test_inds):
    
    # Sets of drugs and targets related to training and test subsets on the indices.
    drugs_training = set(drug_inds[training_inds])
    targets_training = set(target_inds[training_inds])
    drugs_test = set(drug_inds[test_inds])
    targets_test = set(target_inds[test_inds])
    
    # Move drugs from test data to train data if missing
    missing_drugs = drugs_test - drugs_training
    for drug in missing_drugs:
        indices = np.where(drug_inds[test_inds] == drug)[0]
        if indices.size > 0:
            selected_idx = np.random.choice(indices)
            moved_index = test_inds.pop(selected_idx)
            training_inds.append(moved_index)
    
    # Update the sets of drugs and targets to match the current state
    drugs_training = set(drug_inds[training_inds])
    targets_training = set(target_inds[training_inds])
    drugs_test = set(drug_inds[test_inds])
    targets_test = set(target_inds[test_inds])
    
    # Move targets from test data to train data if missing
    missing_targets = targets_test - targets_training
    for target in missing_targets:
        indices = np.where(target_inds[test_inds] == target)[0]
        if indices.size > 0:
            selected_idx = np.random.choice(indices)
            moved_index = test_inds.pop(selected_idx)
            training_inds.append(moved_index)
    
    return training_inds, test_inds



"""
Function to perform cross-validation on the train set to choose best k2 for the model.
Inputs are the interaction matrix, the target values, list of candidate k2 values, and
number of folds for cross-validation, returns best performing k2.
"""

def cross_validation(X, Y, candidate_k2_values, num_folds):

    best_k2 = None
    best_performance = -np.inf
    kf = KFold(n_splits = num_folds, shuffle = True, random_state = 42)
    
    print("")
    for k2 in candidate_k2_values:

        print(f"Performing {num_cv_folds_k}-fold CV with k = {k2}...")
        fold_performances = []

        for train_index, val_index in kf.split(X):
            
            # Ensure drugs and targets in test set exist in train set
            train_index, val_index = ensure_drug_target_presence(drug_inds, target_inds, train_index.tolist(), val_index.tolist())
            train_index = np.array(train_index)
            val_index = np.array(val_index)

            X_train_cv, X_val_cv = X[train_index], X[val_index]
            Y_train_cv, Y_val_cv = Y[train_index], Y[val_index]
            
            fm = pywFM.FM(
                task = settings.task,
                num_iter = settings.num_iter,
                k2 = k2,
                learning_method = settings.learning_method,
                r2_regularization = settings.r2_regularization,
                silent = True)

            model = fm.run(X_train_cv, Y_train_cv, X_val_cv, Y_val_cv)
            P_val_cv = np.array(model.predictions)
            perf = cindex(Y_val_cv, P_val_cv)
            fold_performances.append(perf)
        
        avg_perf = np.mean(fold_performances)

        if avg_perf > best_performance:
            best_performance = avg_perf
            best_k2 = k2

    print(f"\nSelected k =", best_k2, f"as a result of {num_cv_folds_k}-fold CV\n")
    return best_k2



if __name__ == "__main__":

    dataset_str = sys.argv[1]
    feature_map_1 = {"davis": ["davis"], "metz": ["metz"], "tang": ["kiba"], "merget": ["merget"], "gpcr": ["GPCR"], "ic": ["IC"], "e": ["E"]}
    dataset = feature_map_1[dataset_str]
    feature_option_str = sys.argv[2]
    feature_map_2 = {"bi": 1, "si": 2, "both": 3}
    feature_option = feature_map_2[feature_option_str]
    file_base_name = sys.argv[3]
    predictions_filename = f"Results/predictions_{file_base_name}.csv"

    if dataset == ["davis"]:
        print("\nKI dataset from Davis et al. (2011) is used")
    elif dataset == ["metz"]:
        print("\nKI dataset from Metz et al. (2011) is used")
    elif dataset == ["kiba"]:
        print("\nKIBA dataset from Tang et al. (2014) is used")
    elif dataset == ["merget"]:
        print("\nKW dataset from Merget et al. (2017) is used")
    elif dataset == ["GPCR"]:
        print("\nGPCR dataset from Yamanishi et al. (2008) is used")
    elif dataset == ["IC"]:
        print("\nIC dataset from Yamanishi et al. (2008) is used")
    elif dataset == ["E"]:
        print("\nE dataset from Yamanishi et al. (2008) is used")

    if feature_option == 1:
        print("\nOnly binary identifiers are used")
    elif feature_option == 2:
        print("\nOnly side information is used")
    elif feature_option == 3:
        print("\nBoth binary identifiers and side information are used")

    # CV to validate model, choose best hyperparameter k
    candidate_k2_values = settings.candidate_k2_values
    num_cv_folds_k = settings.num_cv_folds_k

    # CV to replace train-test split, 5 means 80–20 split ratio
    num_cv_folds_split = settings.num_cv_folds_split
    
    for ds in dataset:

        XD, XT, Y, drug_inds, target_inds, num_drugs, num_targets = eval("load_" + ds + "()")
        
        # Implementation of cross-validation to replace train-test split
        cv_split = KFold(n_splits = num_cv_folds_split, shuffle = True, random_state = 42)
        fold_results = []
        
        for fold_num, (train_index, test_index) in enumerate(cv_split.split(Y), 1):

            print("")
            print("-" * 40)

            # Ensure all drugs and targets in test set are also in training set
            train_index, test_index = ensure_drug_target_presence(drug_inds, target_inds, train_index.tolist(), test_index.tolist())
            train_index = np.array(train_index)
            test_index = np.array(test_index)

            # Concatenation of features
            X_train = concatenate_features(XD, XT, drug_inds[train_index], target_inds[train_index], num_drugs, num_targets, feature_option)
            X_test = concatenate_features(XD, XT, drug_inds[test_index], target_inds[test_index], num_drugs, num_targets, feature_option)
            Y_train, Y_test = Y[train_index], Y[test_index]

            # Cross-validation to validate model
            best_k2 = cross_validation(X_train, Y_train, candidate_k2_values, num_cv_folds_k)

            # Train factorisation machine model            
            fm = pywFM.FM(
                task = settings.task,
                num_iter = settings.num_iter,
                k2 = best_k2,
                learning_method = settings.learning_method,
                r2_regularization = settings.r2_regularization,
                silent = True)

            # Evaluate the factorisation machine model with C-index
            model = fm.run(X_train, Y_train, X_test, Y_test)
            P_test = np.array(model.predictions)
            performance = cindex(Y_test, P_test)
            fold_results.append(performance)
            print(f"Fold C-index: {performance}")

            # Function to save the predictions            
            def save_predictions_to_file(Y_test, P_test, drug_ids, target_ids, fold, dataset, feature_option, filename):
                file_exists = os.path.isfile(filename)
                with open(filename, "a", newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "Fold", "Dataset", "Feature option", 
                            "Drug ID", "Target ID", "True values", "Predictions"
                        ])
                    for drug_id, target_id, true_val, pred_val in zip(drug_ids, target_ids, Y_test, P_test):
                        writer.writerow([fold, dataset, feature_option, drug_id, target_id, true_val, pred_val])

            # Save the predictions
            save_predictions_to_file(Y_test, P_test, drug_inds[test_index], target_inds[test_index], fold_num, ds, feature_option, predictions_filename)

            

        print("")
        print("-" * 40)
        print("")
        print("*" * 40, f"\n\nC-index = {np.mean(fold_results)}\n")
        print("*" * 40)
        print("")