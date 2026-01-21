import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from rlscore.measure import cindex

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
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

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

    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

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
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

"""
Function to load an incomplete data set introduced by
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
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

def load_GPCR():
    Y = np.loadtxt("../Datasets/gpcr_admat_dgc.txt")
    XD = np.loadtxt("../Datasets/gpcr_simmat_dc.txt")
    XT = np.loadtxt("../Datasets/gpcr_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

def load_IC():
    Y = np.loadtxt("../Datasets/ic_admat_dgc.txt")
    XD = np.loadtxt("../Datasets/ic_simmat_dc.txt")
    XT = np.loadtxt("../Datasets/ic_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')

def load_E():
    Y = np.loadtxt("../Datasets/e_admat_dgc.txt")
    XD = np.loadtxt("../Datasets/e_simmat_dc.txt")
    XT = np.loadtxt("../Datasets/e_simmat_dg.txt")
    XD = 100*XD
    XT = 100*XT
    drug_inds, target_inds = np.where(np.isnan(Y)==False)
    Y = Y[drug_inds, target_inds]
    return XD, XT, Y, drug_inds.astype('int32'), target_inds.astype('int32')


"""
Function for splitting a data set into training and test sets so that split_percentage
represent the relative size of the training set. E.g. split_percentage = 0.7 means that 70 %
of the data is assigned to the training set. 
After randomly dividing the data points, it is made sure that all the drugs and targets in
the test set exist also in the training set. Some data points are moved from test set to 
training set in order to fulfill this condition. 
"""
def train_test_splits(drug_inds, target_inds, split_percentage, random_seed):
    n_sample = len(drug_inds)
    np.random.seed(random_seed)
    # Randomly select which subset each data point is assigned to. 
    train_test_sequence = np.random.choice(a = ["training", "test"], 
                                          size = n_sample, 
                                          replace = True, 
                                          p = [split_percentage, 1-split_percentage])
    training_inds= list(np.where(np.asarray(train_test_sequence) == "training")[0])
    test_inds= list(np.where(np.asarray(train_test_sequence) == "test")[0])

    # Sets of drugs and targets related to training and test subsets on the indices.
    drugs_training = set(drug_inds[training_inds])
    targets_training = set(target_inds[training_inds])
    drugs_test = set(drug_inds[test_inds])
    targets_test = set(target_inds[test_inds])

    # If there are some elements in test set that are not present in the training data, 
    # move elements from test data to training data so that the condition is fulfilled.
    if not drugs_test.issubset(drugs_training):
        in_test_not_training = drugs_test.difference(drugs_training)
        for d in in_test_not_training:
            indices = list(np.where(drug_inds[test_inds] == d)[0])
            # Randomly select one of the indices among the indices of the pairs having drug d as a drug component.
            selected_ind = np.random.choice(indices, 1)[0]
            popped_ind = test_inds.pop(selected_ind)
            training_inds.append(popped_ind)
        # Update the sets of drugs and targets to match the current state.
        drugs_training = set(drug_inds[training_inds])
        targets_training = set(target_inds[training_inds])
        drugs_test = set(drug_inds[test_inds])
        targets_test = set(target_inds[test_inds])

    # Do also the same related to the targets. 
    if not targets_test.issubset(targets_training):
        in_test_not_training = targets_test.difference(targets_training)
        for t in in_test_not_training:
            indices = list(np.where(target_inds[test_inds] == t)[0])
            # Randomly select one of the indices among the indices of the pairs having target t as a target component.
            selected_ind = np.random.choice(indices, 1)[0]
            popped_ind = test_inds.pop(selected_ind)
            training_inds.append(popped_ind)

    return training_inds, test_inds

"""
Function to concatenate information from two feature matrices. 
A row contains features related to a drug-target pair: 
first the features related to the drug and then the features related to the target,
if X1 and inds1 are related to the drugs and X2 and inds2 to the targets.
"""
def concatenate_features(X1, X2, inds1, inds2):
    features_X1 = X1[inds1,:]
    features_X2 = X2[inds2,:]
    X = np.hstack((features_X1, features_X2))
    return X

if __name__ == "__main__":
    # Select a seed or multiple seeds for controlling the randomness in creating the splits.
    random_seeds = [2688385916]
    split_percentage = 1.0/5.0 # About 20 % of the data are used as training data.
    datasets = ["davis", "metz", "kiba", "merget", "GPCR", "IC", "E"]
    for random_seed in random_seeds:
        for ds in datasets:
            # Load the data set in the wanted form.
            XD, XT, Y, drug_inds, target_inds = eval('load_'+ds+'()')
            training_inds, test_inds = train_test_splits(drug_inds, target_inds, split_percentage, random_seed)
            
            # Labels and drug and target indices in the training and test subsets.
            Y_train = Y[training_inds]
            train_drug_inds = drug_inds[training_inds]
            train_target_inds = target_inds[training_inds]
            Y_test = Y[test_inds]
            test_drug_inds = drug_inds[test_inds]
            test_target_inds = target_inds[test_inds]

            # Create such feature representation that it is suitable for algorithms in the library sklearn. 
            X_train = concatenate_features(XD, XT, train_drug_inds, train_target_inds)
            X_test = concatenate_features(XD, XT, test_drug_inds, test_target_inds)

            # Example model: kNN with 30 neighbours.
            regressor = KNeighborsRegressor(n_neighbors=30)
            model = regressor.fit(X_train, Y_train)
            P_test = model.predict(X_test)

            performance = cindex(Y_test, P_test)
            print(performance)