"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import yaml
from termcolor import colored
from utils.common_config import get_val_dataset, get_val_transformations, get_val_dataloader,\
                                get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank
from PIL import Image
from utils.CustomDataset import CustomDataset

FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--config_exp', help='Location of config file')
FLAGS.add_argument('--model', help='Location where model is saved')
FLAGS.add_argument('--visualize_prototypes', action='store_true', 
                    help='Show the prototpye for each cluster')
args = FLAGS.parse_args()

from data.custom_dataset import AugmentedDataset
def main():
    
    # Read config file
    print(colored('Read config file {} ...'.format(args.config_exp), 'blue'))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config)

    # Get dataset
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    # dataset = get_val_dataset(config, transforms)
    dataset = CustomDataset('/home/fwojciak/projects/Unsupervised-Classification/costarica_2017_32x32', transforms)

    dataloader = get_val_dataloader(config, dataset)
    print('Number of samples: {}'.format(len(dataset)))

    # Get model
    print(colored('Get model ...', 'blue'))
    model = get_model(config)
    print(model)

    # Read model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict = torch.load(args.model, map_location='cpu')

    if config['setup'] in ['simclr', 'moco', 'selflabel']:
        model.load_state_dict(state_dict)

    elif config['setup'] == 'scan':
        model.load_state_dict(state_dict['model'])

    else:
        raise NotImplementedError
        
    # CUDA
    model.cuda()

    # Perform evaluation
    if config['setup'] in ['simclr', 'moco']:
        print(colored('Perform evaluation of the pretext task (setup={}).'.format(config['setup']), 'blue'))
        print('Create Memory Bank')
        if config['setup'] == 'simclr': # Mine neighbors after MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'],
                                    config['num_classes'], config['criterion_kwargs']['temperature'])

        else: # Mine neighbors before MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'], 
                                    config['num_classes'], config['temperature'])
        memory_bank.cuda()

        print('Fill Memory Bank')
        fill_memory_bank(dataloader, model, memory_bank)

        print('Mine the nearest neighbors')
        for topk in [1, 5, 20]: # Similar to Fig 2 in paper 
            _, acc = memory_bank.mine_nearest_neighbors(topk)
            print('Accuracy of top-{} nearest neighbors on validation set is {:.2f}'.format(topk, 100*acc))


    elif config['setup'] in ['scan', 'selflabel']:
        print(colored('Perform evaluation of the clustering model (setup={}).'.format(config['setup']), 'blue'))
        head = state_dict['head'] if config['setup'] == 'scan' else 0

        # TODO Here use a custom dataset 
        predictions, features = get_predictions(config, dataloader, model, return_features=True)

        # clustering_stats = hungarian_evaluate(head, predictions, dataset.classes, 
        #                                         compute_confusion_matrix=True)
        # print(clustering_stats)
        if args.visualize_prototypes:
            prototype_indices = get_prototypes(config, predictions[head], features, model)
            visualize_indices(prototype_indices, dataset)
    else:
        raise NotImplementedError

@torch.no_grad()
def get_prototypes(config, predictions, features, model, threshold=0.9):
    import torch.nn.functional as F

    # Get indices of features with predicted probability above threshold for each class, sorted by probability
    print('Get indices')
    probs = predictions['probabilities']
    n_classes = probs.shape[1]
    indices_list = []
    for pred_id in range(n_classes):
        pred_probs = probs[:, pred_id]
        pred_indices = torch.nonzero(pred_probs > threshold).squeeze(1)
        _, sorted_indices = torch.sort(pred_probs[pred_indices], descending=True)
        sorted_indices = pred_indices[sorted_indices]
        indices_list.append(sorted_indices.tolist())

    print(indices_list)

    return indices_list

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random

def visualize_indices(indices, dataset):
    fig, axes = plt.subplots(ncols=len(indices), nrows=5, figsize=(len(indices), 5))

    # iterate over the indices and add each image as a subplot
    for i, list in enumerate(indices):
        for j in range(5): #enumerate(list)
            axes[j][i].axis('off')
            if len(list) <= j:
                continue
            img_path = dataset.get_image(list[j]) # path to the 32x32 dataset
            filename = os.path.basename(img_path)
            new_path = '/media/datasets/flickr-familly/familly_Aukett/costarica_2017/' + filename # path to the og dataset
            img = Image.open(new_path).convert('RGB')

            axes[j][i].imshow(img)
            # axes[i][j].set_title(f"Index: {i}")

    # adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.3)
    fig.tight_layout() 

    # save the figure to a file
    plt.savefig("out/proto.png")

# def update_labels(indices, dataset):
#     # iterate over the indices and add each image as a subplot, i can be seen as the 'label class'
#     for i, list in enumerate(indices):
#         for j, image in enumerate(list):
#             img_path = dataset.get_image(list[j]) # path to the 32x32 dataset
#             filename = os.path.basename(img_path)             
#             base_name, extension = os.path.splitext(filename)

def update_labels(input_list, dataset, csv_filename): # TODO TESTING !
    # Open the CSV file for reading and writing
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Loop through each sublist in the input list
    for i, sublist in enumerate(input_list):
        # Loop through each item in the sublist
        for j, item in enumerate(sublist):
            # Find the row in the CSV file with a matching 'name' field
            img_path = dataset.get_image(item) # path to the 32x32 dataset
            filename = os.path.basename(img_path)        
            for row in rows:
                if row['name'] == filename:
                    # Set the 'label' field to the index of the sublist
                    row['label'] = i

    # Write the updated rows back to the CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['id', 'name', 'label', 'upload_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def move_images_to_label_subdirs(csv_filename, images_dir):
    # Read the CSV file to get the label of each file
    with open(csv_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # Create subdirectories for each label
    for row in rows:
        label = row['label']
        os.makedirs(os.path.join(images_dir, label), exist_ok=True)

    # Move each file to the appropriate subdirectory based on its label
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'): 
            full_path = os.path.join(images_dir, filename)
            base_name, extension = os.path.splitext(filename)

            # Find the row in the CSV file with a matching 'name' field
            for row in rows:
                if row['name'] == base_name:
                    # Move the file to the appropriate subdirectory based on its label
                    label = row['label']
                    dest_dir = os.path.join(images_dir, label)
                    shutil.move(full_path, os.path.join(dest_dir, filename))
                    break

def select_random_image(image_dir):
    image_list = [f for f in os.listdir(image_dir) if f.endswith('.jpg')] 
    if not image_list:
        return None
    else:
        return os.path.join(image_dir, random.choice(image_list))


# TODO TEST THE METHODS

if __name__ == "__main__":
    main() 
