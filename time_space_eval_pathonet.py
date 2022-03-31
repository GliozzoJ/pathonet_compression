##### USAGE EXAMPLE: python time-space_eval_pathonet.py --compression uCWS --net PathoNet.hdf5 --testset Test --compr_weights 0-0-0-256-0.0001-1e-05-5-75-147.1456.h5
#####       --compression = considered quantization method (used only in the output file)
#####       --net = original pre-trained PathoNet network (i.e. uncompressed network)
#####       --testset = name of folder containing the test set images+json files (NOTE: do no put the final /)
#####       --compr_weights = quantized weights saved using pickle (i.e. the compressed network)


import click
import lzma
import tensorflow as tf
import pickle
import numpy as np
import imageio 
import json
from scipy import misc
#from PIL import Image 
import timeit
import math
import os
from tensorflow.keras.layers import (Input,Add,add,concatenate,Activation,concatenate,
                        Concatenate,Dropout,BatchNormalization,Reshape,Permute,
                        Dense,UpSampling2D,Flatten,Lambda,Activation,Conv2D,
                        DepthwiseConv2D,ZeroPadding2D,GlobalAveragePooling2D,
                        MaxPooling2D,AveragePooling2D,LeakyReLU,Conv2DTranspose)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import SGD, Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #disable all logging output (info, warning, error)


# Extract output of the layer preceding a "quantized" one
def make_submodel(model, index):
    submodel = tf.keras.Model(inputs=model.input,
                            outputs=model.layers[index-1].output)
    return submodel

# Extract indices of "quantized" layers and the original tensor dimension
def extract_conv_info(model):
    indexes = []
    weights = []
    for i, layer in enumerate(model.layers):
        if "conv" in layer.name:
            indexes += [i]
            weights += [layer.get_weights()]
    return indexes, weights

def extract_compressed_weights(model):
    indexes, weights = extract_conv_info(model)

    last_bias = weights[-1][1]
    weights[-1] = [weights[-1][0]]
    weights = [w[0] for w in weights]
    vect_weights = [np.hstack(weight).reshape(-1,1) for weight in weights]
    all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
    uniques = np.unique(all_vect_weights)
    dict_ass = {x:i for i,x in enumerate(uniques)}
    indexes_weights = [np.vectorize(lambda x: dict_ass[x])(weight) for weight in weights]
    dict_index_centers = {v:k for k,v in dict_ass.items()}
    vect_centers = np.array([dict_index_centers[k] for k in dict_index_centers.keys()]).reshape(-1, 1)
    
    return indexes, weights, indexes_weights, vect_centers, last_bias

def read_labels(name,inputShape,imageShape):
    with open(name,'r') as f:
        temp = json.load(f)
        labels=[]
        for d in temp:
            if imageShape[0] ==255 and imageShape[1]==255:
                x=int(d['x'])
                y=int(d['y'])
            else:
                x=min(max(int(int(d['x'])*(inputShape[0]/imageShape[0])),0),inputShape[0])
                y=min(max(int(int(d['y'])*(inputShape[1]/imageShape[1])),0),inputShape[1])
            c=int(d['label_id'])-1
            labels.append([x,y,c])
        labels=np.array(labels)
    return labels


# Compute space of original and compressed model
def space_calc(model):
    
    space_per_indexes = lambda x: 8 if x <= 256 else 16

    indexes, weights, indexes_weights, vect_centers, last_bias = extract_compressed_weights(model)
    k = len(vect_centers)
    space_original = 0
    space_compressed = 0
    
    for i in range(len(indexes)):
        space_original += weights[i].size*32
        space_compressed += weights[i].size*space_per_indexes(k)+32*k
    return space_original, space_compressed


# Compute time of original and compressed model
def product_time(model, data, times=5):
    indexes, weights, indexes_weights, vect_centers, last_bias = extract_compressed_weights(model)
    time_original = 0
    time_compressed = 0
    
    imageShape = [1228, 1228, 3]
    inputShape = [256, 256, 3]
    
    for i in range(len(indexes)):
        sm = make_submodel(model, indexes[i])
        for d in data:
            img=imageio.imread(d)
            labels=read_labels(d.replace(".jpg",".json"),inputShape,imageShape).reshape((-1,3))
            #img=np.expand_dims(misc.imresize(img,inputShape), axis=0)
            #img=np.expand_dims(np.array(Image.fromarray(img).resize(size=(inputShape[0], inputShape[1])))/255, axis=0)
            img=np.expand_dims(np.array(misc.imresize(img,size=(inputShape[0], inputShape[1])))/255, axis=0)

            actual_input = sm.predict(img)
            timeit.timeit(lambda: tf.nn.conv2d(actual_input, weights[i], strides=(1, 1), padding='SAME'), number=times, globals=globals())
            timeit.timeit(lambda: tf.nn.conv2d(actual_input, vect_centers[indexes_weights[i]].reshape(weights[i].shape), strides=(1, 1), padding='SAME'), number=times, globals=globals())
            time_original += timeit.timeit(lambda: tf.nn.conv2d(actual_input, weights[i], strides=(1, 1), padding='SAME'), number=times, globals=globals())
            time_compressed += timeit.timeit(lambda: tf.nn.conv2d(actual_input, vect_centers[indexes_weights[i]].reshape(weights[i].shape), strides=(1, 1), padding='SAME'), number=times, globals=globals())
        print(time_original, time_compressed)
    
    return time_original, time_compressed


#New code of Pathonet with iMap rappresentation
def residualDilatedInceptionModule_iMap(y, nb_channels, indexes_weights_temp, vect_centers, _strides=(1, 1),t="e"):
    if t=="d":
        w = indexes_weights_temp.pop(0)
        y = tf.nn.conv2d(y,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        w = indexes_weights_temp.pop(0)
        y = tf.nn.conv2d(y,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)


    w = indexes_weights_temp.pop(0)
    A1 = tf.nn.conv2d(y,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))
    A1 = BatchNormalization()(A1)
    A1 = LeakyReLU()(A1)
    w_4 = indexes_weights_temp.pop(0)
    w = indexes_weights_temp.pop(0)
    A1 = tf.nn.conv2d(A1,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))
    A1 = BatchNormalization()(A1)
    A1 = LeakyReLU()(A1)

    

    A4 = tf.nn.conv2d(y,vect_centers[w_4].reshape(w_4.shape), padding="SAME", strides=(1, 1),dilations=4)
    A4 = BatchNormalization()(A4)
    A4 = LeakyReLU()(A4)
    w = indexes_weights_temp.pop(0)
    A4 = tf.nn.conv2d(A4,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1), dilations=4)
    A4 = BatchNormalization()(A4)
    A4 = LeakyReLU()(A4)

    if (t=="e"):
        y=concatenate([y,y])
    y=add([A1,A4,y])
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)

    return y




def PathoNet_iMaps(last_bias, indexes_weights, vect_centers, input_size = (256,256,3), classes=3, res_weights = None):
    indexes_weights_temp = indexes_weights.copy()
    tt = "uint8" if len(vect_centers) <= 256 else "uint16"
    indexes_weights_temp = [i.astype(tt) for i in indexes_weights_temp]

    inputs = Input(input_size) 
    w = indexes_weights_temp.pop(0)
    block1 = tf.nn.conv2d(inputs,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU()(block1)
    w = indexes_weights_temp.pop(0)
    block1 = tf.nn.conv2d(block1,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))
    block1 = BatchNormalization()(block1)
    block1 = LeakyReLU()(block1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(block1)


    block2= residualDilatedInceptionModule_iMap(pool1,32,t="e", indexes_weights_temp=indexes_weights_temp, vect_centers=vect_centers)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block2)

    block3= residualDilatedInceptionModule_iMap(pool2,64,t="e",indexes_weights_temp=indexes_weights_temp, vect_centers=vect_centers)
    pool3 = MaxPooling2D(pool_size=(2, 2))(block3)

    block4= residualDilatedInceptionModule_iMap(pool3,128,t="e", indexes_weights_temp=indexes_weights_temp, vect_centers=vect_centers)
    pool4 = MaxPooling2D(pool_size=(2, 2))(block4)
    drop4 = Dropout(0.1)(pool4)

    block5= residualDilatedInceptionModule_iMap(drop4,256,t="e", indexes_weights_temp=indexes_weights_temp, vect_centers=vect_centers)
    drop5 = Dropout(0.1)(block5)

    up6 = residualDilatedInceptionModule_iMap((UpSampling2D(size = (2,2))(drop5)),128,t="d", indexes_weights_temp=indexes_weights_temp, vect_centers=vect_centers)
    merge6 = concatenate([block4,up6], axis = 3)

    up7 = residualDilatedInceptionModule_iMap((UpSampling2D(size = (2,2))(merge6)),64,t="d", indexes_weights_temp=indexes_weights_temp, vect_centers=vect_centers)
    merge7 = concatenate([block3,up7], axis = 3)

    up8 = residualDilatedInceptionModule_iMap((UpSampling2D(size = (2,2))(merge7)),32,t="d", indexes_weights_temp=indexes_weights_temp, vect_centers=vect_centers)
    merge8 = concatenate([block2,up8], axis = 3)

    up9 = residualDilatedInceptionModule_iMap((UpSampling2D(size = (2,2))(merge8)),16,t="d", indexes_weights_temp=indexes_weights_temp, vect_centers=vect_centers)
    merge9 = concatenate([block1,up9], axis = 3)
    
    w = indexes_weights_temp.pop(0)
    block9 = tf.nn.conv2d(merge9,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))
    
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)
    
    w = indexes_weights_temp.pop(0)
    block9 = tf.nn.conv2d(block9,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))
    
    
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)

    w = indexes_weights_temp.pop(0)
    block9 = tf.nn.conv2d(block9,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))

    
    block9 = BatchNormalization()(block9)
    block9 = LeakyReLU()(block9)
    w = indexes_weights_temp.pop(0)
    conv10 = tf.nn.relu(tf.nn.conv2d(block9,vect_centers[w].reshape(w.shape), padding="SAME", strides=(1, 1))+last_bias)

    model = tf.keras.models.Model(inputs = inputs, outputs = conv10)

    if(res_weights):
        model.set_weights(res_weights)
    
    
    return model
    
    
def utils_for_Patho_iMap(original_pathonet_model):
    res_weights = []
    for i in (original_pathonet_model.get_weights()):
        if len(i.shape) != 4:
            res_weights.append(i)
    last_bias = res_weights[-1]
    res_weights = res_weights[:-1]
    
    _, _, indexes_weights, vect_centers, _ = extract_compressed_weights(original_pathonet_model)
    
    return res_weights, last_bias, indexes_weights, vect_centers
    
    

# Main function to save the compressed network in "iMap format" and compute space time ratios, space on disk in MB
@click.command()
@click.option('--compression', help='Considered quantization method (used only in the output file)')
@click.option('--net', help='original pre-trained PathoNet network path (i.e. uncompressed network)')
@click.option('--testset', help='name of folder containing the test set images+json files (NOTE: do no put the final /)')
@click.option('--compr_weights', default=".", help='quantized weights saved using pickle (i.e. the compressed network)')

def main(compression, net, testset, compr_weights):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    #Activate GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
        # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
    # Load original model
    model = tf.keras.models.load_model(net)
    
    with open(compr_weights, "rb") as w:
        lw = pickle.load(w)
    model.set_weights(lw)
    
    space_original, space_compressed = space_calc(model)
    space_ratio = round((space_compressed/space_original)**(-1), 3)
    
    data = [testset+"/"+f for f in os.listdir(testset) if '.jpg' in f]
    original, quant = product_time(model, data)
    time_ratio = round((quant/original)**(-1), 3)
    
    #create PathoNet with iMap
    res_weights, last_bias, indexes_weights, vect_centers = utils_for_Patho_iMap(model)
    #patho_imap = PathoNet_iMaps(last_bias, indexes_weights, vect_centers, res_weights = res_weights)
    
    
    
    with lzma.open("res_weights.xz", "wb") as f:
        pickle.dump(res_weights, f)

    with lzma.open("last_bias.xz", "wb") as f:
        pickle.dump(last_bias, f)

    with lzma.open("indexes_weights.xz", "wb") as f:
        pickle.dump(indexes_weights, f)

    with lzma.open("vect_centers.xz", "wb") as f:
        pickle.dump(vect_centers, f)
        
    space_on_disk_MB = round((os.path.getsize("last_bias.xz")+os.path.getsize("res_weights.xz")+os.path.getsize("indexes_weights.xz")+os.path.getsize("vect_centers.xz"))/1024**2, 3)
    
    
    file_res = "time_space_patho.txt"

    with open(file_res, "a+") as tex:
        tex.write(f"{compression} k={compr_weights.split('-')[3]} --> space ratio RAM = {space_ratio}, space on disk (MB) = {space_on_disk_MB}, original space on disk (MB) = {round(os.path.getsize(net)/1024**2, 3)}, time ratio = {time_ratio}\n")


if __name__ == '__main__':
    main()



