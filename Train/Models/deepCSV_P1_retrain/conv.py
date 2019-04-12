

#hardcoded converter

#/dense_1/dense_1/bias:0

# softmax or rectified

import h5py
import numpy as np

def _get_dense_layer_parameters(h5, id, activation):
    """Get weights, bias, and n-outputs for a dense layer"""
   
    weights = np.asarray(h5['/'+id+'/'+id+'/kernel:0'])
    bias = np.asarray(h5['/'+id+'/'+id+'/bias:0'])
    
    #print (weights)
    
    assert weights.shape[1] == bias.shape[0]
    # TODO: confirm that we should be transposing the weight
    # matrix the Keras case
    return_dict = {
        'weights': weights.T.flatten('C').tolist(),
        'bias': bias.flatten('C').tolist(),
        'architecture': 'dense',
        'activation': activation,
    }
    return return_dict, weights.shape[1]


f=h5py.File("mweights.h5")

layers=[]

layers.append(_get_dense_layer_parameters(f,'dense_1','rectified'))
layers.append(_get_dense_layer_parameters(f,'dense_2','rectified'))
layers.append(_get_dense_layer_parameters(f,'dense_3','rectified'))
layers.append(_get_dense_layer_parameters(f,'dense_4','rectified'))
layers.append(_get_dense_layer_parameters(f,'dense_5','rectified'))
layers.append(_get_dense_layer_parameters(f,'dense_6','softmax'))

layers=tuple(layers)

alldict={'layers':layers}

import json
with open('data.json', 'w') as fp:
    json.dump(alldict, fp)


