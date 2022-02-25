import tensorflow as tf
from os import listdir
import numpy as np
from imageio import imread
import json
import random





def SHIDC_B_Ki67(batch_size, path_preproc_data):
    trainData = [path_preproc_data+"/Train/"+f for f in listdir(path_preproc_data+"/Train/") if '.jpg' in f] 
    valData = [path_preproc_data+"/Val/"+f for f in listdir(path_preproc_data+"/Val/") if '.jpg' in f] 
    testData = [path_preproc_data+"/Test/"+f for f in listdir(path_preproc_data+"/Test/") if '.jpg' in f]

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    
    # Read training data
    for i in trainData:
        # read images
        x_train.append(imread(i))
        # read labels (Gaussian distributions)
        y_train.append(np.load(i.replace(".jpg",".npy")))
    x_train = np.array(x_train).astype("float32") / 255.0 #In pathonet DataLoader, normalization is done only on imgs
    y_train = np.array(y_train)
    
    # Read validation data
    for i in valData:
        # read images
        x_val.append(imread(i))
        # read labels (Gaussian distributions)
        y_val.append(np.load(i.replace(".jpg",".npy")))
    x_val = np.array(x_val).astype("float32") / 255.0 #In pathonet DataLoader, normalization is done only on imgs
    y_val = np.array(y_val)

    # Read Test data
    for i in testData:
        # read images
        x_test.append(imread(i))
        # read labels (Gaussian distributions)
        y_test.append(np.load(i.replace(".jpg",".npy")))
    x_test = np.array(x_test).astype("float32") / 255.0 #In pathonet DataLoader, normalization is done only on imgs
    y_test = np.array(y_test)

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(1000).batch(batch_size)

    return dataset, x_train, y_train, x_val, y_val, x_test, y_test


# Example of function call
# dataset, x_train, y_train, x_val, y_val, x_test, y_test = SHIDC_B_Ki67(batch_size=100, path_preproc_data="./preprocessed_data")


