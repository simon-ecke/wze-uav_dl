# Author: Simon Ecke
# Date: 05.04.2023
# Code Repository: https://github.com/simon-ecke/wze-uav_dl
#
# Description: This script consisting of several functions. The first three functions (nbv_to_sst_3classes, nbv_to_sst_4classes, and nbv_to_sst_5classes) take a NumPy array of label values and map them to a smaller set of discrete labels (3, 4, or 5 classes) based on a set of predefined ranges. The fourth function (data_split) takes four NumPy arrays (representing image data, label data, hash IDs, and species data) as inputs, and uses train_test_split from scikit-learn to split the data into training and testing sets. The split is stratified by label class, and the function outputs several NumPy arrays representing the training and testing data, as well as some information about the split.


import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# nbv to 3 classes
def nbv_to_sst_3classes(label_set: np.array):
    
    label_list = []
    for label in label_set:
        if label >= 0 and label <= 25:
            label = 0
            label_list.append(label)
        elif label >= 30 and label <= 95:
            label = 1
            label_list.append(label)
        else:
            label = 2
            label_list.append(label)
            
    label_set = np.array(label_list)
    
    return label_set


# nbv to 4 classes
def nbv_to_sst_4classes(label_set: np.array):
    
    label_list = []
    for label in label_set:
        if label >= 0 and label <= 25:
            label = 0
            label_list.append(label)
        elif label >= 30 and label <= 60:
            label = 1
            label_list.append(label)
        elif label >= 65 and label <= 95:
            label = 2
            label_list.append(label)
        else:
            label = 3
            label_list.append(label)
            
    label_set = np.array(label_list)
    
    return label_set


# nbv to 5 classes
def nbv_to_sst_5classes(label_set: np.array):
    
    label_list = []
    for label in label_set:
        if label >= 0 and label <= 10:
            label = 0
            label_list.append(label)
        elif label >= 15 and label <= 25:
            label = 1
            label_list.append(label)
        elif label >= 30 and label <= 60:
            label = 2
            label_list.append(label)
        elif label >= 65 and label <= 95:
            label = 3
            label_list.append(label)
        else:
            label = 4
            label_list.append(label)
            
    label_set = np.array(label_list)
    
    return label_set


# species IDs to classes
def species_to_classes(species_set: np.array):
    
    species_list = []
    oak_species = [48, 51, 98]
    other_conifers = [103, 104, 106, 116, 117, 119, 120, 123, 128, 129, 133, 135, 136, 137]
    for tree in species_set:
        if tree == 118: # spruces = species ID 118
            tree = 0
            species_list.append(tree)
        elif tree == 134: # pines
            tree = 1
            species_list.append(tree)
        elif tree == 100: # firs
            tree = 2
            species_list.append(tree)
        elif tree == 20: # beeches
            tree = 3
            species_list.append(tree)
        elif tree in oak_species: # oaks
            tree = 4
            species_list.append(tree)
        elif tree in other_conifers: # other conifers
            tree = 5
            species_list.append(tree)
        else:
            tree = 6
            species_list.append(tree) # other broad-leaved trees
            
    species_set = np.array(species_list)
    return species_set


def data_split_TEMP(image_set: np.array, label_set: np.array, hash_id: np.array, species_set: np.array, test_size=0.15, random_state=42):
    
    unique_values = np.unique(hash_id)
    num_unique = len(unique_values)
    print(f"There are {num_unique} unique values within hash_id.")
    
    # Split unique hash_id values into train and test sets
    train_ids, test_ids = train_test_split(unique_values, test_size=test_size, random_state=random_state)
    
    # Use np.isin() to create boolean arrays indicating which indices belong to train or test sets
    train_indices = np.isin(hash_id, train_ids)
    test_indices = np.isin(hash_id, test_ids)
    
    # Reshape boolean arrays to match shape of image_set and label_set
    train_indices = train_indices.reshape(-1, 1)
    test_indices = test_indices.reshape(-1, 1)
    
    # Select images and labels for train and test sets
    train_image_set = image_set[train_indices[:, 0]]
    train_label_set = label_set[train_indices[:, 0]]
    train_hash_id = hash_id[train_indices[:, 0]]
    train_species_set = species_set[train_indices[:, 0]]
    
    test_image_set = image_set[test_indices[:, 0]]
    test_label_set = label_set[test_indices[:, 0]]
    test_hash_id = hash_id[test_indices[:, 0]]
    test_species_set = species_set[test_indices[:, 0]]
    
    train_label_set = train_label_set.reshape(-1, 1)
    test_label_set = test_label_set.reshape(-1, 1)
    train_hash_id = train_hash_id.reshape(-1, 1)
    test_hash_id = test_hash_id.reshape(-1, 1)
    train_species_set = train_species_set.reshape(-1, 1)
    test_species_set = test_species_set.reshape(-1, 1)
    
    print("Check shapes:\n")
    print(f"Images train dataset: {train_image_set.shape}")
    print(f"Labels train dataset: {train_label_set.shape}\n")
    
    print(f"Images test dataset: {test_image_set.shape}")
    print(f"Labels test dataset: {test_label_set.shape}\n")
    print('-'*50)
    print (f"Check if the split was stratified: (random_state={random_state})")
    print(f"Healthy trees in train dataset: {np.count_nonzero(train_label_set == 0)}")
    print(f"Stressed trees in train dataset: {np.count_nonzero(train_label_set == 1)}")
    print(f"Dead trees in train dataset: {np.count_nonzero(train_label_set == 2)}")
    print(f"Healthy trees in test dataset: {np.count_nonzero(test_label_set == 0)}")
    print(f"Stressed trees in test dataset: {np.count_nonzero(test_label_set == 1)}")
    print(f"Dead trees in test dataset: {np.count_nonzero(test_label_set == 2)}")
    print(f"Ratio health trees in test dataset: {np.count_nonzero(test_label_set == 0)/np.count_nonzero(label_set == 0)}")
    print(f"Ratio stressed trees in test dataset: {np.count_nonzero(test_label_set == 1)/np.count_nonzero(label_set == 1)}")
    print(f"Ratio dead trees in test dataset: {np.count_nonzero(test_label_set == 2)/np.count_nonzero(label_set == 2)}")
    
    del image_set, label_set, hash_id, species_set
    return train_image_set, train_label_set, train_hash_id, train_species_set, test_image_set, test_label_set, test_hash_id, test_species_set


def data_split_TEMP(image_set: np.array, label_set: np.array, hash_id: np.array, species_set: np.array, test_size=0.1667, random_state=42, seed=2):
    
    unique_values = np.unique(hash_id[:,0])
    num_unique = len(unique_values)
    print(f"There are {num_unique} unique values within hash_id.")
    
    # get image IDs for each unique hash ID and year
    image_ids_2020 = hash_id[hash_id[:, 1] == '2020'][:, 0]
    image_ids_2021 = hash_id[hash_id[:, 1] == '2021'][:, 0]
    image_ids_2022 = hash_id[hash_id[:, 1] == '2022'][:, 0]
    
    # split unique hash IDs into a subset and test datasets
    sub_ids, test_ids = train_test_split(unique_values, test_size=test_size, random_state=random_state)
    
    # create boolean arrays indicating which indices belong to train or test sets
    # 1. combined subset of training and validation data
    sub_indices = np.isin(hash_id[:, 0], sub_ids)
    num_true = np.count_nonzero(sub_indices)
    num_false = np.count_nonzero(~sub_indices)
    print("Number of True in sub_indices:", num_true)
    print("Number of False in sub_indices:", num_false)
    
    # 2. test dataset
    np.random.seed(seed) # set the seed value for randomly choosing an image of one of the years
    test_indices = np.zeros_like(sub_indices)  # initialize to all False
    for hash_id_test in test_ids:
        # select one image ID randomly from either 2020 or 2021 or 2022 for each unique hash ID in the test set
        year = np.random.choice(['2020', '2021', '2022'])
        image_ids = hash_id[(hash_id[:, 0] == hash_id_test) & (hash_id[:, 1] == year), 0]
        
        # mark the index corresponding to the selected image ID and hash ID as True in the test indices array
        test_indices[(hash_id[:, 0] == hash_id_test) & (hash_id[:, 1] == year) & (np.isin(hash_id[:, 0], image_ids))] = True
  
    num_true = np.count_nonzero(test_indices)
    num_false = np.count_nonzero(~test_indices)
    print("Number of True in test_indices:", num_true)
    print("Number of False in test_indices:", num_false)
    
    # Reshape boolean arrays to match shape of image_set and label_set
    sub_indices = sub_indices.reshape(-1, 1)
    test_indices = test_indices.reshape(-1, 1)
    
    # Select images and labels for train_val and test sets
    sub_image_set = image_set[sub_indices[:, 0]]
    sub_label_set = label_set[sub_indices[:, 0]]
    sub_hash_id = hash_id[sub_indices[:, 0]]
    sub_species_set = species_set[sub_indices[:, 0]]
    
    test_image_set = image_set[test_indices[:, 0]]
    test_label_set = label_set[test_indices[:, 0]]
    test_hash_id = hash_id[test_indices[:, 0]]
    test_species_set = species_set[test_indices[:, 0]]
    
    sub_label_set = sub_label_set.reshape(-1, 1)
    test_label_set = test_label_set.reshape(-1, 1)
    sub_species_set = sub_species_set.reshape(-1, 1)
    test_species_set = test_species_set.reshape(-1, 1)

    print("Check shapes:\n")
    print(f"Images sub dataset: {sub_image_set.shape}")
    print(f"Labels sub dataset: {sub_label_set.shape}\n")
    print(f"Images test dataset: {test_image_set.shape}")
    print(f"Labels test dataset: {test_label_set.shape}\n")
    print('-'*50)
    print (f"Check if the split was stratified: (random_state={random_state})")
    print(f"Healthy trees in sub dataset: {np.count_nonzero(sub_label_set == 0)}")
    print(f"Stressed trees in sub dataset: {np.count_nonzero(sub_label_set == 1)}")
    print(f"Dead trees in sub dataset: {np.count_nonzero(sub_label_set == 2)}")
    print(f"Healthy trees in test dataset: {np.count_nonzero(test_label_set == 0)}")
    print(f"Stressed trees in test dataset: {np.count_nonzero(test_label_set == 1)}")
    print(f"Dead trees in test dataset: {np.count_nonzero(test_label_set == 2)}")
    print(f"Ratio health trees in test dataset: {np.count_nonzero(test_label_set == 0)/np.count_nonzero(sub_label_set == 0)}")
    print(f"Ratio stressed trees in test dataset: {np.count_nonzero(test_label_set == 1)/np.count_nonzero(sub_label_set == 1)}")
    print(f"Ratio dead trees in test dataset: {np.count_nonzero(test_label_set == 2)/np.count_nonzero(sub_label_set == 2)}")
    
    del image_set, label_set, hash_id, species_set
    return sub_image_set, sub_label_set, sub_hash_id, sub_species_set, test_image_set, test_label_set, test_hash_id, test_species_set


def data_split(image_set: np.array, label_set: np.array, hash_id: np.array, species_set: np.array, test_size=0.1667, random_state=42, seed=2):
     
    # group the hashIDs to get the unique values of hashIDs    
    groups = hash_id[:, 0] 
    print("ORIGINAL POSITIVE RATIO:", label_set.mean())
    # create a StratifiedGroupKFold instance
    cv = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=random_state)
    # loop through the first fold
    for fold, (sub_ids, test_ids) in enumerate(cv.split(image_set, label_set, groups)):
        print("Fold :", fold)
        print("TRAIN POSITIVE RATIO:", label_set[sub_ids].mean())
        print("TEST POSITIVE RATIO :", label_set[test_ids].mean())
        print("LENGTH TRAIN GROUPS :", len(set(groups[sub_ids])))
        print("LENGTH TEST GROUPS  :", len(set(groups[test_ids])))
        break # we only need one fold

    sub_ids = hash_id[:, 0][sub_ids]
    test_ids = np.unique(hash_id[:, 0][test_ids])
    
    # create boolean arrays indicating which indices belong to train or test sets
    # 1. combined subset of training and validation data
    sub_indices = np.isin(hash_id[:, 0], sub_ids)
    num_true = np.count_nonzero(sub_indices)
    num_false = np.count_nonzero(~sub_indices)
    print("Number of True in sub_indices:", num_true)
    print("Number of False in sub_indices:", num_false)
    
    # 2. test dataset
    np.random.seed(seed) # set the seed value for randomly choosing an image of one of the years
    test_indices = np.zeros_like(sub_indices)  # initialize to all False
    for hash_id_test in test_ids:
        # select one image ID randomly from either 2020 or 2021 or 2022 for each unique hash ID in the test set
        temp = np.unique(hash_id[(hash_id[:, 0] == hash_id_test), 1]) # check how many years are available per hashID
        if len(temp) == 1:
            year = temp[0]
        elif len(temp) == 2:
            year = np.random.choice(temp)
        else:
            year = np.random.choice(temp)
        
        image_ids = hash_id[(hash_id[:, 0] == hash_id_test) & (hash_id[:, 1] == year), 0]
        
        # mark the index corresponding to the selected image ID and hash ID as True in the test indices array
        test_indices[(hash_id[:, 0] == hash_id_test) & (hash_id[:, 1] == year) & (np.isin(hash_id[:, 0], image_ids))] = True
  
    num_true = np.count_nonzero(test_indices)
    num_false = np.count_nonzero(~test_indices)
    print("Number of True in test_indices:", num_true)
    print("Number of False in test_indices:", num_false)
    
    # Reshape boolean arrays to match shape of image_set and label_set
    sub_indices = sub_indices.reshape(-1, 1)
    test_indices = test_indices.reshape(-1, 1)
    
    # Select images and labels for train_val and test sets
    sub_image_set = image_set[sub_indices[:, 0]]
    sub_label_set = label_set[sub_indices[:, 0]]
    sub_hash_id = hash_id[sub_indices[:, 0]]
    sub_species_set = species_set[sub_indices[:, 0]]
    
    test_image_set = image_set[test_indices[:, 0]]
    test_label_set = label_set[test_indices[:, 0]]
    test_hash_id = hash_id[test_indices[:, 0]]
    test_species_set = species_set[test_indices[:, 0]]
    
    sub_label_set = sub_label_set.reshape(-1, 1)
    test_label_set = test_label_set.reshape(-1, 1)
    sub_species_set = sub_species_set.reshape(-1, 1)
    test_species_set = test_species_set.reshape(-1, 1)

    print("Check shapes:\n")
    print(f"Images sub dataset: {sub_image_set.shape}")
    print(f"Labels sub dataset: {sub_label_set.shape}\n")
    print(f"Images test dataset: {test_image_set.shape}")
    print(f"Labels test dataset: {test_label_set.shape}\n")
    print('-'*50)
    print (f"Check if the split was stratified: (random_state={random_state})")
    print(f"Healthy trees in sub dataset: {np.count_nonzero(sub_label_set == 0)}")
    print(f"Stressed trees in sub dataset: {np.count_nonzero(sub_label_set == 1)}")
    print(f"Dead trees in sub dataset: {np.count_nonzero(sub_label_set == 2)}")
    print(f"Healthy trees in test dataset: {np.count_nonzero(test_label_set == 0)}")
    print(f"Stressed trees in test dataset: {np.count_nonzero(test_label_set == 1)}")
    print(f"Dead trees in test dataset: {np.count_nonzero(test_label_set == 2)}")
    print(f"Ratio health trees in test dataset: {np.count_nonzero(test_label_set == 0)/np.count_nonzero(sub_label_set == 0)}")
    print(f"Ratio stressed trees in test dataset: {np.count_nonzero(test_label_set == 1)/np.count_nonzero(sub_label_set == 1)}")
    print(f"Ratio dead trees in test dataset: {np.count_nonzero(test_label_set == 2)/np.count_nonzero(sub_label_set == 2)}")
    
    del image_set, label_set, hash_id, species_set
    return sub_image_set, sub_label_set, sub_hash_id, sub_species_set, test_image_set, test_label_set, test_hash_id, test_species_set