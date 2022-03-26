# Reducing the Complexity of Deep Learning Models for Medical Applications in Resource-limited Contexts: A Use Case


This repository contains the code to:

- Compress the deep neural network PathoNet [1] using different quantization strategies (i.e. CWS, PWS, UQ, ECSQ) [2];

- Reproduce the experiments comparing the original PathoNet model with our compressed models, which are described in detail in a paper submitted to the journal "Computers in Biology and Medicine" [3];

- A Jupyter notebook showing (I) how to create and load compressed models using the "Index Map"  method [4] and (II) how to compute the compression ratios and time ratios and for a given quantized network.


Moreover, we provide our best compressed PathoNet models using *k*=256 which can be exploited by an interested user having limited computational resources available.

## Getting started

### Requirements and installation

* Install `python3`, `python3-pip` and `python3-venv`.
* Make sure that `python --version` starts by 3 or execute `alias python='python3'` in the shell.
* Create a virtual environment and activate it: 

  ```
  python3 -m venv /path/to/new/virtual/environment
  source /path/to/new/virtual/environment/bin/activate
  ```

  If you want to easily activate the environment, you can add a permanent 
  alias to your `.bashrc`:
  
  ```
  gedit ~/.bashrc
  ```

  Add the alias `alias patho="source /path/to/new/virtual/environment/bin/activate"` at the end of the file to be able to activate the environment by simply typing `patho`.
  
* Activate the environment typing `patho`.

* Install the [package sHAM](https://github.com/AnacletoLAB/sHAM), which contains the implementation of the considered quantization strategies:

  ```
  git clone https://github.com/AnacletoLAB/sHAM.git
  cd sHAM
  pip install ./sHAM_package
  ```
  Then compile the megaDot as explained [here](https://github.com/AnacletoLAB/sHAM#compiling-megadot).


* Then, with the environment active, clone this repository in the indicated position inside the sHAM package and install the required depedencies:

  ```
  cd ./experiments/performance_eval
  git clone https://github.com/GliozzoJ/pathonet_compression.git
  cd pathonet_compression
  pip3 install -r ./requirements.txt
  ```

  The set of suggested dependencies, and in general all the code, have been tested with Python 3.6.8 on Ubuntu 18.10 (Cosmic Cuttlefish) OS.


### Dataset

The dataset SHIDC-B-Ki-67-V1.0 can be requested to the authors of PathoNet [here](https://shiraz-hidc.com/service/ki-67-dataset/).

Once you have the dataset, please copy its content in the folder `SHIDC-B-Ki-67`. The content of this folder should look like this:

```
ls ./SHIDC-B-Ki-67/
notes.txt
Test
Train
```

### PathoNet pre-trained model (not compressed)

The PathoNet pre-trained model (not compressed) is available in the folder `original_nets` o it can be downloaded from the original [PathoNet github repository](https://github.com/SHIDCenter/PathoNet/blob/master/README.md#pretrained-models). In the latter case, please copy the downloaded model in the folder `original_nets`.


### Quantized PathoNet models (*k*=256)

In the folder `./pathonet_compressed_models/experiments/`, we provide our PathoNet quantized models using a number of representatives equal to 256, which showed good perfomance in terms of generalization metrics, compression and time ratios (see paper [3] for further details).


## Code usage

### Create the validation set

Run the following command to create the validation set used in our experiments by randomly selecting 20% of the labeled images from the training data available in the SHIDC-B-Ki-76 (version 1.0) dataset:

```
python3 split_train_val.py --inputPath ./SHIDC-B-Ki-67 --outputPath ./preprocessed_data
```

The newly created training and validation sets are saved in the folder `./preprocessed_data`. 

### Data preprocessing

To perform the data preprocessing which computes the data augmentation step for training data and the density maps (Gaussian labels) for training, validation and test data, run:

```
mkdir ./preprocessed_data/Train
python3 ./PathoNet/preprocessing.py -i ./preprocessed_data/train_split -o ./preprocessed_data/Train -a True

mkdir ./preprocessed_data/Val
python3 ./PathoNet/preprocessing.py -i ./preprocessed_data/val_split -o ./preprocessed_data/Val -a False

mkdir ./preprocessed_data/Test
python3 ./PathoNet/preprocessing.py -i ./SHIDC-B-Ki-67/Test -o ./preprocessed_data/Test -a False
```

### Run quantization on PathoNet pre-trained model

The function `compression.py` performs the quantization and retraining of the PathoNet pre-trained model given as input. To show the help for the function:

```
python3 compression.py --help
``` 


Following some examples performing the quantization of PathoNet using different approaches (i.e. CWS, PWS, UQ and ECSQ):

```
# CWS
python3 compression.py --compression uCWS --net ./original_nets/PathoNet.hdf5 --data_path ./preprocessed_data --output_path ./pathonet_compressed_models/results --minibatch 8 --clustercnn 256 

# PWS
python3 compression.py --compression uPWS --net ./original_nets/PathoNet.hdf5 --data_path ./preprocessed_data --output_path ./pathonet_compressed_models/results --minibatch 8 --clustercnn 256

# UQ
python3 compression.py --compression uUQ --net ./original_nets/PathoNet.hdf5 --data_path ./preprocessed_data --output_path ./pathonet_compressed_models/results --minibatch 8 --clustercnn 256

# ECSQ
python3 compression.py --compression uECSQ --net ./original_nets/PathoNet.hdf5 --data_path ./preprocessed_data --output_path ./pathonet_compressed_models/results --minibatch 8 --clustercnn 256

```

### Reproduce our results 

To reproduce our results we provide the following scripts/functions:


1. **pathonet_compressed_k256.py**

	This script performsthe quantization of the PathoNet model using the hyper-parameters set at [line 46](https://github.com/GliozzoJ/pathonet_compression/blob/284d503da1bb7fe6cae6bf5247c94ed05422a9d2/pathonet_compressed_k256.py#L46). We used this script to quantize a model with *k=256* and the best hyper-parameters previously obtained by grid search. By default, the hyper-parameters used in the experiment with quantization method UQ are set, but you can uncomment lines [50](https://github.com/GliozzoJ/pathonet_compression/blob/284d503da1bb7fe6cae6bf5247c94ed05422a9d2/pathonet_compressed_k256.py#L50), [55](https://github.com/GliozzoJ/pathonet_compression/blob/284d503da1bb7fe6cae6bf5247c94ed05422a9d2/pathonet_compressed_k256.py#L55) or [60](https://github.com/GliozzoJ/pathonet_compression/blob/284d503da1bb7fe6cae6bf5247c94ed05422a9d2/pathonet_compressed_k256.py#L60) to reproduce the experiment using another compression approach.
	You can run the script in background with the following command:
	
	```
	nohup python3 pathonet_compressed_k256.py > output.log &
	```

	which saves the results in the folder `./pathonet_compressed_models/results` and a text 	log file in the current working directory. In particular, the  script saves:
	- the quantized model (using three different saving methods); 
	- a csv file with the hyper-parameters used to obtain the quantized model and the Precision, Recall and F1-score for Ki67 and TIL (divided in Ki67-positive and Ki67-negative),
	- a pickle file which content is identical to the previous one;
	- two .npz objects containing the MSE (Mean Squared Error) values obtained before quantization, after quantization and after the re-training phase of the best model (one .npz file for the training set and one for the test set).
	
2. **run_rmse_acc.py**

	This function computes a series of patient-level metrics (i.e. RMSE, Ki67 cut-off accuracy and TIL cut-off accuracy) on a PathoNet model given in input. By default the metrics are computed on the original PathoNet network and on the quantized networks we provide in the folder `./pathonet_compression/pathonet_compressed_models/experiments/`. You can run the script typing:
	
	```
	python3 run_rmse_acc.py
	```
	
	The outputs are a csv and a pickle file containing the above-mentioned metrics for each tested model.

3. **time_space_eval_pathonet.py**

	Run this function to compute the space and time ratios for a given quantized network. Moreover, it computes the space on disk (MB) for the quantized and original PathoNet model.
	
	The following command shows the help of this function:
	
	```
	python3 time_space_eval_pathonet.py --help
	```
	
	Considering the quantized networks provided in this repository, you can run:
	
	```
	#CWS
	python3 time_space_eval_pathonet.py --compression CWS --net ./original_nets/PathoNet.hdf5 --testset ./SHIDC-B-Ki-67/Test --compr_weights ./pathonet_compressed_models/experiments/CWS_k256/0-0-0-256-0.0001-1e-05-5-75-147.1456.h5

	#PWS
	python3 time_space_eval_pathonet.py --compression PWS --net ./original_nets/PathoNet.hdf5 --testset ./SHIDC-B-Ki-67/Test --compr_weights ./pathonet_compressed_models/experiments/PWS_k256/0-0-0-256-0.0001-0.0001-5-75-159.29054.h5

	#ECSQ
	python3 time_space_eval_pathonet.py --compression ECSQ --net ./original_nets/PathoNet.hdf5 --testset ./SHIDC-B-Ki-67/Test --compr_weights ./pathonet_compressed_models/experiments/ECSQ_k256/0-0-0-256-0.0001-1e-05-5-75-152.7148.h5

	#UQ
	python3 time_space_eval_pathonet.py --compression UQ --net ./original_nets/PathoNet.hdf5 --testset ./SHIDC-B-Ki-67/Test --compr_weights ./pathonet_compressed_models/experiments/UQ_k256/0-0-0-249-0.0001-0.0001-5-75-146.19026.h5

	
	```

	The results are appended in a text file called `time_space_patho.txt`, which is automatically created at first call of the function.



4. We also provide a Jupyter notebook `Run_PathoNet_iMap.ipynb` showing how to compute the compression and time ratio for a given quantized network. Moreover, it shows how to create a compressed PathoNet model using the IndexMap method and how to load it.


### References

[1] Negahbani, F., Sabzi, R., Pakniyat Jahromi, B. et al. PathoNet introduced as a deep neural network backend for evaluation of Ki-67 and tumor-infiltrating lymphocytes in breast cancer. Sci Rep 11, 8489 (2021). https://doi.org/10.1038/s41598-021-86912-w

[2] Marinò, Giosuè Cataldo, et al. "Compression strategies and space-conscious representations for deep neural networks." 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021.

[3] Gliozzo, Jessica et al. "Compressing Deep Learning Models for Medical Applications in Resource-limited Contexts: A Use Case", Computers in Biology and Medicine (2022) [**submitted**].

[4] Marinò, Giosuè Cataldo, et al. "Compact representations of convolutional neural networks via weight pruning and quantization." arXiv preprint arXiv:2108.12704 (2021).




