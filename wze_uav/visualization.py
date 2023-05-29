# Author: Simon Ecke
# 
# Description: Visualizing random images from PyTorch Datasets.
#

import torch     
import random
import matplotlib.pyplot as plt
import copy

def display_filtered_images(dataset: torch.utils.data.dataset.Dataset,
                              label=None,
                              species=None,
                              n=10,
                              display_shape=True,
                              seed=None, 
                              export_path=None):
    # Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print("For display purposes, n shouldn't be larger than 10. Setting n to 10 and removing shape display.")

    # Set random seed
    if seed is not None:
        random.seed(seed)

    # Make a copy of the dataset
    shuffled_dataset = copy.deepcopy(dataset)

    # Shuffle and select random samples from filtered dataset
    random.shuffle(shuffled_dataset)
    random_samples = shuffled_dataset[:n]

    # Setup plot
    plt.figure(figsize=(30, 20))

    # Loop through samples and display random samples
    for i, (targ_image, targ_label, targ_species) in enumerate(random_samples):
        # Adjust image tensor shape for plotting: [color_channels, height, width] -> [height, width, color_channels]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        #if display_shape:
        #    title = f"Label: {targ_label} | Species: {targ_species}\n ID: {}\n"
        #    #title = f"Label: {targ_label} | Species: {targ_species}\nShape: {targ_image_adjust.shape}"
        #else:
        #    title = f"Label: {targ_label} | Species: {targ_species}"
        #plt.title(title)

        # Export the image as PNG
        if export_path:
            image_filename = f"{species}_{label}_{seed}.png"
            export_filepath = f"{export_path}/{image_filename}"
            plt.savefig(export_filepath)

    # Show the plot
    plt.show()



     