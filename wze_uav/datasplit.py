# Author: Simon Ecke
# Date: 05.04.2023
# Code Repository: https://github.com/simon-ecke/wze-uav_dl
#
# Description: This script consisting of several functions. The first three functions (nbv_to_sst_3classes, nbv_to_sst_4classes, and nbv_to_sst_5classes) take a NumPy array of label values and map them to a smaller set of discrete labels (3, 4, or 5 classes) based on a set of predefined ranges. The fourth function (data_split) takes four NumPy arrays (representing image data, label data, hash IDs, and species data) as inputs, and uses train_test_split from scikit-learn to split the data into training and testing sets. The split is stratified by label class, and the function outputs several NumPy arrays representing the training and testing data, as well as some information about the split.


import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# nbv to 3 classes
def nbv_to_sst_3classes(image_set: np.array, label_set: np.array, species_set: np.array, kkl_set: np.array, bk_set: np.array, hash_id: np.array):
    
    label_list = []
    species_list = [20, 48, 51, 100, 118, 134]
    for i in range(0, len(label_set)):
        if label_set[i] >= 99: # all dead trees
            label = 2
            label_list.append(label)
        elif species_set[i] in species_list and label_set[i] >= 0 and label_set[i] <= 25: # healthy trees
            label = 0
            label_list.append(label)
        elif species_set[i] in species_list and label_set[i] >= 30 and label_set[i] <= 95: # stressed trees
            label = 1
            label_list.append(label)
        else:
            label = 999
            label_list.append(label)
            
    label_set = np.array(label_list)
    
    np_filter = []
    for i in range(0, len(bk_set)):
        if label_set[i] in [0,1,2]:
            np_filter.append(True)
        else:
            np_filter.append(False)
         
    image_set = image_set[np_filter]
    label_set = label_set[np_filter]
    species_set = species_set[np_filter]
    kkl_set = kkl_set[np_filter]
    bk_set = bk_set[np_filter]
    hash_id = hash_id[np_filter]
    
    return image_set, label_set, species_set, kkl_set, bk_set, hash_id


# nbv to 3 classes
def nbv_to_sst_6classes_combined(image_set: np.array, label_set: np.array, species_set: np.array, kkl_set: np.array, bk_set: np.array, hash_id: np.array):
    
    label_list = []
    species_list = [20, 48, 51, 100, 118, 134]
    acer_species = [5, 4, 1]
    larix_species = [116, 117]
    for label in label_set:
        if label >= 99: # all dead trees
            label = 5
            label_list.append(label)
        elif species_set[i] in species_list and label_set[i] >= 0 and label_set[i] <= 25: # healthy trees
            label = 0
            label_list.append(label)
        elif species_set[i] in species_list and label_set[i] >= 30 and label_set[i] <= 95: # stressed trees
            label = 1
            label_list.append(label)
        elif species_set[i] in larix_species:
            label = 2
            label_list.append(label)
        elif species_set[i] in acer_species:
            label = 3
            label_list.append(label)
        elif species_set[i] == 10: #Betula pendula
            label = 4
            label_list.append(label)
        else:
            label = 999
            
    label_set = np.array(label_list)
    
    np_filter = []
    for i in range(0, len(bk_set)):
        if label_set[i] in [0,1,2]:
            np_filter.append(True)
        else:
            np_filter.append(False)
         
    image_set = image_set[np_filter]
    label_set = label_set[np_filter]
    species_set = species_set[np_filter]
    kkl_set = kkl_set[np_filter]
    bk_set = bk_set[np_filter]
    hash_id = hash_id[np_filter]
    
    return image_set, label_set, species_set, kkl_set, bk_set, hash_id


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


# species and stress level to classes
def combined_to_classes(species_set: np.array, nbv_set: np.array, image_set: np.array, kkl_set: np.array, bk_set: np.array, hash_id: np.array):
    
    species_list = []
    oak_species = [48, 51]
    acer_species = [5, 4, 1]
    larix_species = [116, 117]
    #other_conifers = [103, 104, 106, 116, 117, 119, 120, 123, 128, 129, 133, 135, 136, 137]
    for i in range(len(species_set)):
        if nbv_set[i] >= 99: # dead trees
            tree = 13
            species_list.append(tree)
        elif species_set[i] == 118 and nbv_set[i] >= 0 and nbv_set[i] <= 25: # healthy spruces
            tree = 0
            species_list.append(tree)
        elif species_set[i] == 118 and nbv_set[i] >= 30 and nbv_set[i] <= 95: # stressed spruces
            tree = 1
            species_list.append(tree)
        elif species_set[i] == 134 and nbv_set[i] >= 0 and nbv_set[i] <= 25: # healthy pines
            tree = 2
            species_list.append(tree)
        elif species_set[i] == 134 and nbv_set[i] >= 30 and nbv_set[i] <= 95: # stressed pines
            tree = 3
            species_list.append(tree)
        elif species_set[i] == 100 and nbv_set[i] >= 0 and nbv_set[i] <= 25: # healthy firs
            tree = 4
            species_list.append(tree)
        elif species_set[i] == 100 and nbv_set[i] >= 30 and nbv_set[i] <= 95: # stressed firs
            tree = 5
            species_list.append(tree)
        elif species_set[i] == 20 and nbv_set[i] >= 0 and nbv_set[i] <= 25: # healthy beeches
            tree = 6
            species_list.append(tree)
        elif species_set[i] == 20 and nbv_set[i] >= 30 and nbv_set[i] <= 95: # stressed beeches
            tree = 7
            species_list.append(tree)
        elif species_set[i] in oak_species and nbv_set[i] >= 0 and nbv_set[i] <= 25: # healthy oaks
            tree = 8
            species_list.append(tree)
        elif species_set[i] in oak_species and nbv_set[i] >= 30 and nbv_set[i] <= 95: # stressed oaks
            tree = 9
            species_list.append(tree)
        elif species_set[i] in larix_species and nbv_set[i] >=0: # Larix decidua/kaempferi
            tree = 10
            species_list.append(tree)
        elif species_set[i] in acer_species and nbv_set[i] >=0: # acer spp.
            tree = 11
            species_list.append(tree)
        elif species_set[i] == 10 and nbv_set[i] >= 0: # Betula pendula
            tree = 12
            species_list.append(tree)
        else:
            tree = 999
            species_list.append(tree) # other broad-leaved trees
            
    species_setv2 = np.array(species_list)
    label_set = species_setv2
    
    np_filter = []
    for i in range(0, len(bk_set)):
        if label_set[i] in [0,1,2,3,4,5,6,7,8,9,10,11,12,13]:
            np_filter.append(True)
        else:
            np_filter.append(False)
         
    image_set = image_set[np_filter]
    label_set = label_set[np_filter]
    species_set = species_set[np_filter]
    kkl_set = kkl_set[np_filter]
    bk_set = bk_set[np_filter]
    hash_id = hash_id[np_filter]
    
    return image_set, label_set, species_set, kkl_set, bk_set, hash_id


def data_split(image_set: np.array, label_set: np.array, hash_id: np.array, species_set: np.array, n_splits=6, random_state=42, seed=2):
     
    # group the hashIDs to get the unique values of hashIDs    
    groups = hash_id[:, 0] 
    print("ORIGINAL POSITIVE RATIO:", label_set.mean())
    # create a StratifiedGroupKFold instance
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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