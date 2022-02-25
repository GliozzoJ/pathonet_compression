import pandas as pd
import numpy as np
import sys
import json
import time
from numpy.random import seed
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

sys.path.append('./PathoNet/')
import compression
import evaluation



# Set seed to make results reproducible
SEED = 1
seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Function to read and modify json file for configs (modify path to compressed model)
def updateJsonFile(compr_model_path, config_path):
    jsonFile = open(config_path, "r") # Open the JSON file for reading
    data = json.load(jsonFile) # Read the JSON into the buffer
    jsonFile.close() # Close the JSON file

    ## Modify json
    data["pretrainedModel"] = compr_model_path

    ## Save our changes to new JSON file (temp.json)
    jsonFile = open("temp.json", "w+")
    jsonFile.write(json.dumps(data))
    jsonFile.close()
    

# Paths
config_path = "./PathoNet/configs/eval.json"
out='./pathonet_compressed_models/results' #NOTE: results are saved here, change the path to save them in another position

# filename for results
res_name = "results_compression_model_selection"

# Arguments for exps (k and lr cumulative to tune)
param1 = [256, 1024, 4096]
param2 = [0.0001, 0.00001, 0.000001]

parameters = []    
for i in param1:        
    for j in param2:            
        parameters.append(( i, j ))

        
print("The grid of evaluated hyperparameters is:\n")
print(parameters)


# Create dataframe to store results
column_names = ["compression", "lr", "lr_cum", "minibatch", "clustercnn", "patience", "pre_compr_train", "post_compr_train", "acc_train","pre_compr_test", "post_compr_test", "acc_test", "elapsed_time", "prec_ki67+", "prec_ki67-", "prec_TIL", "rec_ki67+", "rec_ki67-", "rec_TIL", "f1_ki67+", "f1_ki67-", "f1_TIL"]

res = pd.DataFrame(columns = column_names)

# Compute and store precision, recall and F1 for the original network
print("Computing Prec, Rec and F1 using the original network on test set... \n")

pre_or, rec_or, F1_or = evaluation.eval(['-i','./SHIDC-B-Ki-67/Test', '-c', config_path])
res = res.append({"compression": float("NaN"), "lr": float("NaN"), "lr_cum": float("NaN"), "minibatch": float("NaN"), "clustercnn": float("NaN"), "patience": float("NaN"), "pre_compr_train": float("NaN"), "post_compr_train": float("NaN"), "acc_train": float("NaN"), "pre_compr_test": float("NaN"), "post_compr_test": float("NaN"), "acc_test": float("NaN"),"elapsed_time": float("NaN"), "prec_ki67+": pre_or[0], "prec_ki67-": pre_or[1], "prec_TIL": pre_or[2], "rec_ki67+": rec_or[0], "rec_ki67-": rec_or[1], "rec_TIL": rec_or[2], "f1_ki67+": F1_or[0], "f1_ki67-": F1_or[1], "f1_TIL": F1_or[2]}, ignore_index=True)

#Compute metrics for experiments using grid search
grid_res = []

for k in range(len(parameters)):
    print("Used hyperparameters are")
    exps = {'compression':'uUQ', 'net':'./original_nets/PathoNet.hdf5', 'data_path':'./preprocessed_data', 'output_path':out,'learning_rate':0.0001, 'epochs':30,
        'lr_cumulative':parameters[k][1], 'minibatch':8, 'prfc':0, 'prcnn':0, 'clusterfc':0, 'clustercnn':parameters[k][0], 'tr':0.001, 'lambd':0., 'logger':False, 
        'ptnc':5, 'internal_ho':True} # NOTE: This dictionary contains the hyper-parameters of the model used for UQ quantization (change it to perform other experiments)
    print(exps)
    start_time = time.time()
    TRAIN_RES, TEST_RES, compression_param, compr_model_path = compression.run(**exps) #Internal holdout=True
    elapsed_time = time.time() - start_time
    grid_res.append(TEST_RES[-1])
    print("\nEvaluated hyper-param combination ", k+1, " out of ", len(parameters), "with elapsed time ", round(elapsed_time, 1))

#Detect which is the best combination of hyperparams
print("\nResults of grid search are: ", grid_res, "\n")
idx_min = grid_res.index(min(grid_res))
print("\nBest model is number: ", idx_min+1, "\n")

#Retrain model with best combination of hyperparameters
SEED = 123
seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

#Update dictionary with best hyperparameters
exps['clustercnn'] = parameters[idx_min][0]
exps['lr_cumulative'] = parameters[idx_min][1]
exps['epochs'] = 75
exps['internal_ho'] = False #Internal holdout=False (training on train+val and test on test set)

print("Used hyperparameters are:")
print(exps)

start_time = time.time()
TRAIN_RES, TEST_RES, compression_param, compr_model_path = compression.run(**exps) #Internal holdout=False    
elapsed_time = time.time() - start_time
updateJsonFile(compr_model_path, config_path) #use compressed model for evaluation
pre, rec, F1 = evaluation.eval(['-i','./SHIDC-B-Ki-67/Test', '-c','./temp.json'])
    
# Save results in dataframe
res = res.append({"compression":exps["compression"], "lr":exps["learning_rate"], "lr_cum": exps["lr_cumulative"], 
                  "minibatch": exps["minibatch"], "clustercnn": float(compression_param.split("-")[3]), "patience": exps["ptnc"],
                  "pre_compr_train": TRAIN_RES[0], "post_compr_train": TRAIN_RES[1], "acc_train": TRAIN_RES[-1], "pre_compr_test": TEST_RES[0], 
                  "post_compr_test": TEST_RES[1], "acc_test":TEST_RES[-1], "elapsed_time": elapsed_time, "prec_ki67+": pre[0],
                  "prec_ki67-": pre[1], "prec_TIL": pre[2], "rec_ki67+": rec[0], "rec_ki67-": rec[1], "rec_TIL": rec[2], "f1_ki67+": F1[0], 
                  "f1_ki67-": F1[1], "f1_TIL": F1[2]}, ignore_index=True)
    
#Delete temp.json
os.remove("./temp.json")
    
#Delete best model checkpoint
dir_content = os.listdir("./")
for item in dir_content:
    if item.endswith("_check.h5"):
        os.remove(os.path.join("./", item))
    
#Save pandas dataframe (pickle and csv)
df_save_path = os.path.join(out, res_name)
res.to_pickle(df_save_path+".pkl")
res.to_csv(df_save_path+".csv")

#Save TRAIN_RES e TEST_RES
np.savez(df_save_path+"_TRAIN_RES", np.array(TRAIN_RES))
np.savez(df_save_path+"_TEST_RES", np.array(TEST_RES))
