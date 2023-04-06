import numpy as np
from sklearn.model_selection import train_test_split

def nbv_to_sst_3classes(label_set: np.array):
    
    label_list = []
    for label in label_set:
        if label >= 0 and label <= 45:
            label = 0
            label_list.append(label)
        elif label >= 50 and label <= 95:
            label = 1
            label_list.append(label)
        else:
            label = 2
            label_list.append(label)
            
    label_set = np.array(label_list)
    
    return label_set


def data_split(image_set: np.array, label_set: np.array, hash_id: np.array, species_set: np.array, test_size=0.15, random_state=42):
    
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