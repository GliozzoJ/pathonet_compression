import os
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import click

from datasets import SHIDC_B_Ki67
from sHAM import pruning, uCWS, uPWS
from sHAM import uUQ, uECSQ, pruning_uCWS, pruning_uPWS
from sHAM import pruning_uUQ, pruning_uECSQ
exec(open("../GPU.py").read())

import sys
sys.path.append('./PathoNet/')
import utils


# Set seed to make results reproducible
# SEED = 1
# from numpy.random import seed
# seed(SEED)
# tf.random.set_seed(SEED)

from tensorflow.python import _pywrap_util_port
print("MKL enabled:", _pywrap_util_port.IsMklEnabled())


# This script does not excercise old non-unified methods. Check https://github.com/giosumarin/ICPR2020_sHAM for those
def run(compression, net, data_path, output_path, learning_rate, epochs, lr_cumulative, minibatch, prfc, prcnn, clusterfc, clustercnn, tr, lambd, logger, ptnc, internal_ho):

    # Load model
    model = tf.keras.models.load_model(net)

    # Compression definition
    if compression == 'pr':
        compression_model = pruning.pruning(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn)
    elif compression == 'uCWS':
        compression_model = uCWS.uCWS(model=model, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'uPWS':
        compression_model = uPWS.uPWS(model=model, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'uUQ':
        compression_model = uUQ.uUQ(model=model, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'uECSQ':
        compression_model = uECSQ.uECSQ(model=model, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc, wanted_clusters_cnn=clustercnn, wanted_clusters_fc=clusterfc, tr=tr, lamb=lambd)
    elif compression == 'pruCWS':
        compression_model = pruning_uCWS.pruning_uCWS(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'pruPWS':
        compression_model = pruning_uPWS.pruning_uPWS(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'pruUQ':
        compression_model = pruning_uUQ.pruning_uUQ(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc)
    elif compression == 'pruECSQ':
        compression_model = pruning_uECSQ.pruning_uECSQ(model=model, perc_prun_for_dense=prfc, perc_prun_for_cnn=prcnn, clusters_for_conv_layers=clustercnn, clusters_for_dense_layers=clusterfc, wanted_clusters_cnn=clustercnn, wanted_clusters_fc=clusterfc, tr=tr, lamb=lambd)

    # Load dataset
    print("\nLoading dataset (it takes several minutes)... \n")
    if internal_ho == True:
        # Evaluation on the validation set
        dataset, x_train, y_train, x_val, y_val, _, _ = SHIDC_B_Ki67(minibatch, data_path)
        x_train = tf.convert_to_tensor(x_train)
        y_train = tf.convert_to_tensor(y_train)
        x_test = tf.convert_to_tensor(x_val)
        y_test = tf.convert_to_tensor(y_val)
    else:
        # Training on train+validation set and evaluation on test set
        _, x_train, y_train, x_val, y_val, x_test, y_test = SHIDC_B_Ki67(minibatch, data_path) #34.3 GB
        x_test = tf.convert_to_tensor(x_test)
        y_test = tf.convert_to_tensor(y_test)
        x_val_aug, y_val_aug = utils.dataAugmentation(x_val, y_val)  #data augmentation validation set
        x_train = tf.convert_to_tensor(np.concatenate((x_train, x_val_aug)))
        y_train = tf.convert_to_tensor(np.concatenate((y_train, y_val_aug)))
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(1000).batch(minibatch)

    print("\nTraining set has shape: ", x_train.shape, "; with dtype: ", x_train.dtype, "\n")
    print("\nTest set has shape: ", x_test.shape, "; with dtype: ", x_test.dtype, "\n")
    
    # Compile the model
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=learning_rate), loss = 'mse', metrics=['MeanSquaredError']) # taken from PathoNet (except metric)
    
    # Pre-compression prediction assessment
    pre_compr_train = compression_model.model.evaluate(x_train, y_train, verbose = 0)[1] #batch size default 32
    pre_compr_test = compression_model.model.evaluate(x_test, y_test, verbose = 0)[1] #batch size default 32
    print("before compression, performance on train -->", pre_compr_train)
    print("before compression, performance on test -->", pre_compr_test) 

    # Model compression
    if compression == 'pr':
        compression_model.apply_pruning()
    elif compression == 'uCWS':
        compression_model.apply_uCWS()
    elif compression == 'uPWS':
        compression_model.apply_uPWS()
    elif compression == 'uUQ':
        compression_model.apply_uUQ()
    elif compression == 'uECSQ':
        # lambdas = [1e-10, 5e-9, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 7.5e-6, 1e-5, 2e-5][::-1]
        # lambdas = [1e-12, 2.5e-12, 5e-12, 7.5e-12, 1e-11, 2.5e-11, 5e-11, 7.5e-11, 1e-10][::-1]
        # compression_model.tune_lambda(lambdas)
        compression_model.apply_uECSQ()
        print("Lambda value is set to:", getattr(compression_model, 'lamb_cnn'))
    elif compression == 'pruCWS':
        compression_model.apply_pr_uCWS()
    elif compression == 'pruPWS':
        compression_model.apply_pr_uPWS()
    elif compression == 'pruUQ':
        compression_model.apply_pr_uUQ()
    elif compression == 'pruECSQ':
        # lambdas = [1e-10, 5e-9, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 7.5e-6, 1e-5, 2e-5][::-1]
        # lambdas = [1e-12, 2.5e-12, 5e-12, 7.5e-12, 1e-11, 2.5e-11, 5e-11, 7.5e-11, 1e-10][::-1]
        # compression_model.tune_lambda(lambdas)
        compression_model.apply_pr_uECSQ()
        print("Lambda value is set to:", getattr(compression_model, 'lamb_cnn'))

    # Post-compression prediction assessment
    compression_model.set_loss(tf.keras.losses.MeanSquaredError())
    compression_model.set_optimizer(tf.keras.optimizers.Adam(lr=learning_rate))
    post_compr_train = compression_model.model.evaluate(x_train, y_train, verbose = 0)[1]
    post_compr_test = compression_model.model.evaluate(x_test, y_test, verbose = 0)[1]
    print("Applying initial compression setting before retrain, performance on train -->" , post_compr_train)
    print("Applying initial compression setting before retrain, performance on test -->" , post_compr_test)
    
    if compression == "pr":
        compression_model.train_pr(epochs=epochs, dataset=dataset, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, step_per_epoch = 10000000, patience=0)
    else:
        compression_model.train_ws(epochs=epochs, lr=lr_cumulative, dataset=dataset, X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, patience=ptnc, min_is_better=True, threshold=0.001, best_model=True)

    # Model save
    name_net = (net.split("/")[-1])[:-5] #Adapted to format ".hdf5"
    TRAIN_RES = ([pre_compr_train] + [post_compr_train] + compression_model.acc_train)
    TEST_RES = ([pre_compr_test] + [post_compr_test] + compression_model.acc_test)

    if compression in ['uUQ', 'pruUQ', 'uECSQ', 'pruECSQ']:
        temp_centers_fc  = len(compression_model.centers_fc)  if hasattr(compression_model, 'centers_fc')  else 0
        temp_centers_cnn = len(compression_model.centers_cnn) if hasattr(compression_model, 'centers_cnn') else 0
        compression_param = "-".join([str(x) for x in [prfc, prcnn, temp_centers_fc, temp_centers_cnn, learning_rate, lr_cumulative, ptnc, epochs]]) + '-'
    else:
        compression_param = "-".join([str(x) for x in [prfc, prcnn, clusterfc, clustercnn, learning_rate, lr_cumulative, ptnc, epochs]]) + '-'

    if logger:
        with open("{}_{}.txt".format(name_net, compression), "a+") as tex:
            tex.write("lr {} {} -->\n {}\n , {}\n\n".format(learning_rate, compression_param, TRAIN_RES, TEST_RES))
 
    if internal_ho == False:
        DIR="{}/{}".format(output_path, compression)
        TO_SAVE = "{}{}".format(compression_param, round(TEST_RES[-1],5))
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        if not os.path.isdir(DIR):
            os.mkdir(DIR)

        end_weights = compression_model.model.get_weights()

        with open(DIR+"/"+TO_SAVE+".h5", "wb") as file:
            pickle.dump(end_weights, file)

        # Save model weights using keras
        compression_model.model.save_weights(DIR+"/"+TO_SAVE+"_save_weights.h5")
        
        compr_model_path = DIR+"/"+TO_SAVE+"_save_weights.h5"
        
        # Save model weights+structure
        compression_model.model.save(DIR+"/"+TO_SAVE+"_save_weights-struct.h5")
        
    else:
        compr_model_path = None
        
    return TRAIN_RES, TEST_RES, compression_param, compr_model_path

@click.command()
@click.option('--compression', help='Type of compression')
@click.option('--net', help='original network datapath (path to .hdf5 file)')
@click.option('--data_path', help='path to preprocessed data (obtained using script preprocessing.py)')
@click.option('--output_path', help='path to folder to save reults')
@click.option('--learning_rate', default=0.001, help='learning rate')
@click.option('--epochs', default=30, help='epochs')
@click.option('--lr_cumulative', default=0.001, help='learning rate for cumulative gradient descent')
@click.option('--minibatch', default=8, help='size of minibatch')
@click.option('--prfc', default=0, help='percentage of pruned connection (dense layers)')
@click.option('--prcnn', default=0, help='percentage of pruned connection (convolutional layers)')
@click.option('--clusterfc', default=0, help='different values for all dense layers')
@click.option('--clustercnn', default=0, help='different values for all convolutional layers')
@click.option('--tr', default=0.001, help='treshold for ECSQ')
@click.option('--lambd', default=0., help='coefficient for entropy with ECSQ')
@click.option('--logger', default=False, help='set True for logging train into txt')
@click.option('--ptnc', default=0, help='patience (default 0)')
@click.option('--internal_ho', default=False, help='Test on validation set not on test set')
def main(compression, net, data_path, output_path, learning_rate, epochs, lr_cumulative, minibatch, prfc, prcnn, clusterfc, clustercnn, tr, lambd, logger, ptnc, internal_ho):
    run(compression, net, data_path, output_path, learning_rate, epochs, lr_cumulative, minibatch, prfc, prcnn, clusterfc, clustercnn, tr, lambd, logger, ptnc, internal_ho)
    

if __name__ == '__main__':
    main()

    
