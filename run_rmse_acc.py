import pandas as pd
import sys
import os
sys.path.append('./PathoNet/')
from numpy.random import seed
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #to use CPU
import tensorflow as tf

from update_json import updateJsonFile
import evaluation

# Set seed to make results reproducible
SEED = 1
seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# Iterate across compressed networks and compute RMSE, cutoff accuracy for ki67-score and TIL-score
# Change the paths listed in "list_comprNets" to evaluate other compressed networks (leave the first item
# as it is)
list_comprNets=["./original_nets/PathoNet.hdf5",
"./pathonet_compression/pathonet_compressed_models/experiments/CWS_k256/0-0-0-256-0.0001-1e-05-5-75-147.1456_save_weights.h5",
"./pathonet_compression/pathonet_compressed_models/experiments/PWS_k256/0-0-0-256-0.0001-0.0001-5-75-159.29054_save_weights.h5",
"./pathonet_compression/pathonet_compressed_models/experiments/ECSQ_k256/0-0-0-256-0.0001-1e-05-5-75-152.7148_save_weights.h5", 
"./pathonet_compression/pathonet_compressed_models/experiments/UQ_k256/0-0-0-249-0.0001-0.0001-5-75-146.19026_save_weights.h5"]


# Paths
config_path = "./PathoNet/configs/eval.json"
res_name = "results_RMSE_cutoffAccuracy_comprNets"


# Create dataframe to store results
column_names = ["exp", "rmse_ki67", "rmse_TIL", "acc_ki67_pt", "acc_TIL_pt"]

res = pd.DataFrame(columns = column_names)

for i,p in enumerate(list_comprNets):
    print("Evaluated network path: ", p)
    updateJsonFile(p, config_path, "temp")
    rmse_ki67, rmse_TIL, acc_ki67_pt, acc_TIL_pt = evaluation.eval_pts(['-i','./SHIDC-B-Ki-67/Test', '-c', './temp.json'])
    print() #Print empty line
    os.remove("temp.json")
    if i==0:
        res = res.append({"exp":"Original", "rmse_ki67":rmse_ki67, "rmse_TIL":rmse_TIL, "acc_ki67_pt":acc_ki67_pt, 
                          "acc_TIL_pt":acc_TIL_pt}, ignore_index=True)
    else:
        res = res.append({"exp":p.split("/")[6], "rmse_ki67":rmse_ki67, "rmse_TIL":rmse_TIL, "acc_ki67_pt":acc_ki67_pt, 
                          "acc_TIL_pt":acc_TIL_pt}, ignore_index=True)

# Save results
res.to_pickle(res_name+".pkl")
res.to_csv(res_name+".csv")
