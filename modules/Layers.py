import tensorflow as tf
#from keras.engine import Layer
from keras.layers import Layer
import keras.backend as K
from keras import initializers
from keras import utils
from caloGraphNN_keras import *

global_layers_list = {} #same as for losses

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1
    
    grad_name = "GradientReversal%d" % reverse_gradient.num_calls
    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]
    
    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda=1., **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda
    
    def build(self, input_shape):
        self.trainable_weights = []
    
    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)
    
    def get_output_shape_for(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class SortLayer(Layer):

    def __init__(self, kernel_initializer='glorot_uniform', **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_size = utils.normalize(1, 1, 'kernel_size')
        #self.kernel_size = conv_utils.normalize_tuple(1, 1, 'kernel_size')
        super(SortLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        channel_axis = 2
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, 1)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',trainable=True)
        super(SortLayer, self).build(input_shape)  # Be sure to call this at the end


    def compute_output_shape(self, input_shape):
        outshape=list(input_shape)
        print('compute', tuple(outshape))
        return tuple(outshape)
        
    def call(self, x):
        n_batch = tf.shape(x)[0]
        values = K.conv1d(x, self.kernel, strides = 1, padding = "valid", data_format ='channels_last', dilation_rate = 1)
        values = tf.squeeze(values, axis=2)
        values = tf.nn.softsign(values)+1
        index = tf.nn.top_k(values, x.get_shape()[1]).indices
        values = tf.expand_dims(values,axis=2)
        x = x*values
        index = tf.expand_dims(index, axis=2)
        batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
        batch_range = tf.tile(batch_range, [1, x.get_shape()[1], 1])
        index_tensor = tf.concat([batch_range,index],axis=2)
        x = tf.gather_nd(x,index_tensor)
        return x


class SortLayer_noweights(Layer):

    def __init__(self, **kwargs):
        super(SortLayer_noweights, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        outshape=list(input_shape)
        print('compute', tuple(outshape))
        return tuple(outshape)
        
    def call(self, inputs):
        n_batch = tf.shape(inputs)[0]
        sorting_feature = tf.nn.softsign(inputs[:,:,0])+1.
        index = tf.nn.top_k(sorting_feature, inputs.get_shape()[1]).indices
        sorting_feature = tf.expand_dims(sorting_feature, axis=2)
        output = inputs * sorting_feature

        index = tf.expand_dims(index, axis=2)
        batch_range = tf.expand_dims(tf.expand_dims(tf.range(0, n_batch), axis=1), axis=1)
        batch_range = tf.tile(batch_range, [1, inputs.get_shape()[1], 1])
        index_tensor = tf.concat([batch_range,index],axis=2)

        out = tf.gather_nd(output,index_tensor)
        return out

class SmoothSort(Layer):

    def __init__(self, **kwargs):
        super(SmoothSort, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        outshape=list(input_shape)
        print('compute', tuple(outshape))
        return tuple(outshape)
        
    def call(self, inputs):
        n_batch = tf.shape(inputs)[0]
        sorting_feature = inputs[:,:,0]
        #sorting_feature = inputs
        #inputs.get_shape()[1]
        sorting_feature = tf.reshape(sorting_feature,[n_batch,tf.shape(inputs)[1],1])
        
        A_s = sorting_feature- tf.transpose(sorting_feature,perm=[0,2,1])
        A_s = tf.abs(A_s)

        n = tf.shape(sorting_feature)[1]
        one = tf.ones((n,1),dtype = tf.float32)
        #@ tf.transpose(one)
        B = tf.einsum('mij,jk->mik', A_s, one)
        K = tf.range(n) + 1
        C = tf.einsum('mij,jk->mik', sorting_feature, tf.expand_dims(tf.cast(n + 1 - 2 * K, dtype = tf.float32), 0) )
        P = tf.transpose(C - B, perm=[0, 2, 1])
        P = tf.nn.softmax(P / 0.1, -1)

        sf = tf.nn.softsign(inputs[:,:,0])+1
        sf = tf.expand_dims(sf,axis=2)
        scaled_inputs = sf*inputs
        
        output = tf.matmul(P,scaled_inputs)
        return output


class ParticleSort(Layer):

    def __init__(self, **kwargs):
        super(ParticleSort, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        outshape=list(input_shape)
        print('compute', tuple(outshape))
        return tuple(outshape)
        
    def call(self, inputs):
        n_batch = tf.shape(inputs)[0]
        sorting_feature = inputs[:,:,0]
        #sorting_feature = inputs
        #inputs.get_shape()[1]
        sorting_feature = tf.reshape(sorting_feature,[n_batch,tf.shape(inputs)[1],1])
        
        A_s = sorting_feature- tf.transpose(sorting_feature,perm=[0,2,1])
        A_s = tf.abs(A_s)

        n = tf.shape(sorting_feature)[1]
        one = tf.ones((n,1),dtype = tf.float32)
        #@ tf.transpose(one)
        B = tf.einsum('mij,jk->mik', A_s, one)
        K = tf.range(n) + 1
        C = tf.einsum('mij,jk->mik', sorting_feature, tf.expand_dims(tf.cast(n + 1 - 2 * K, dtype = tf.float32), 0) )
        P = tf.transpose(C - B, perm=[0, 2, 1])
        P = tf.nn.softmax(P / 0.1, -1)

        sf = tf.nn.softsign(inputs[:,:,0])+1
        sf = tf.expand_dims(sf,axis=2)
        scaled_inputs = sf*inputs
        
        output = tf.matmul(P,scaled_inputs)
        return output


global_layers_list['GradientReversal'] = GradientReversal
global_layers_list['SortLayer'] = SortLayer
global_layers_list['SortLayer_noweights'] = SortLayer_noweights
global_layers_list['SmoothSort'] = SmoothSort
global_layers_list['ParticleSort'] = ParticleSort


global_layers_list['GravNet'] = GravNet

print global_layers_list
