import numpy as np
import os
import random
import json
import click
import shutil


def read_lbl(json_file):
    with open(json_file,'r') as f:
        js = json.load(f)
        labels = []
        for k in js:
            l = int(k['label_id'])
            labels.append(l)
        labels = np.array(labels)
    return labels


def check_AvgIMG(data_split_list):
    split_labels = []
    
    for p in data_split_list:
        json_file = p.split('.j')[0] + '.json'
        labels = read_lbl(json_file)
        split_labels.append(labels.tolist())
    
    flattened_labels = [val for sublist in split_labels for val in sublist]
    print("Avg./IMG for Immunopositive:", round(flattened_labels.count(1)/len(data_split_list), 2), ", total cells: ",  flattened_labels.count(1))
    print("Avg./IMG for Immunonegative:", round(flattened_labels.count(2)/len(data_split_list), 2), ", total cells: ",  flattened_labels.count(2))
    print("Avg./IMG for TIL:", round(flattened_labels.count(3)/len(data_split_list), 2), ", total cells: ",  flattened_labels.count(3))


# This function splits the provided training set in a train and validation sets.
@click.command()
@click.option('--inputPath', help='Path to SHIDC-B-Ki-67 dataset')
@click.option('--outputPath', help='Path to folder where data splits (i.e. train, validation) are saved')
def main(inputpath, outputpath):
    train_list = [inputpath+"/Train/"+f for f in os.listdir(inputpath+"/Train/") if '.jpg' in f]
    train_list = sorted(train_list)
    testData = [inputpath+"/Test/"+f for f in os.listdir(inputpath+"/Test/") if '.jpg' in f] # Test set is not augmented (700 imgs)
    testData = sorted(testData)
    
    # Split training data in training and validation sets
    data = train_list.copy()
    random.Random(4).shuffle(data)
    thr = int(len(data)* 0.2)
    trainData=data[thr:]
    valData=data[:thr]
    
    # Check Avg./IMG for each set
    print("TRAINING SET data - ", len(trainData), " images\n")
    check_AvgIMG(trainData)
    print("\nVALIDATION SET data - ", len(valData), " images\n")
    check_AvgIMG(valData)
    print("\nTRAIN + VALIDATION data - ", len(data), "images\n")
    check_AvgIMG(data)
    print("\nTEST SET data - ", len(testData), "images\n")
    check_AvgIMG(testData)

    # Create new training and validation folders (train_split, val_split)
    if not os.path.isdir(os.path.join(outputpath, 'train_split')):
        os.mkdir(os.path.join(outputpath, 'train_split'))
    if not os.path.isdir(os.path.join(outputpath, 'val_split')):
        os.mkdir(os.path.join(outputpath, 'val_split'))

    for i in trainData:
        shutil.copy2(i, os.path.join(outputpath, 'train_split'))
        shutil.copy2(i.split('.j')[0] + '.json', os.path.join(outputpath, 'train_split'))
    
    for i in valData:
        shutil.copy2(i, os.path.join(outputpath, 'val_split'))
        shutil.copy2(i.split('.j')[0] + '.json', os.path.join(outputpath, 'val_split'))


if __name__ == '__main__':
    main()


#TRAINIG SET data -  1325  images

#Avg./IMG for Immunopositive: 20.88 , total cells:  27664
#Avg./IMG for Immunonegative: 45.04 , total cells:  59683
#Avg./IMG for TIL: 1.81 , total cells:  2402

#VALIDATION SET data -  331  images

#Avg./IMG for Immunopositive: 22.48 , total cells:  7442
#Avg./IMG for Immunonegative: 46.31 , total cells:  15327
#Avg./IMG for TIL: 2.15 , total cells:  710

#TRAIN + VALIDATION data -  1656 images

#Avg./IMG for Immunopositive: 21.2 , total cells:  35106
#Avg./IMG for Immunonegative: 45.3 , total cells:  75010
#Avg./IMG for TIL: 1.88 , total cells:  3112

#TEST SET data -  700 images

#Avg./IMG for Immunopositive: 22.51 , total cells:  15755
#Avg./IMG for Immunonegative: 46.63 , total cells:  32643
#Avg./IMG for TIL: 1.97 , total cells:  1380

