from keras.layers import Dense, Dropout, Flatten,Concatenate, Lambda, Convolution2D, LSTM, Convolution1D, Conv2D,GlobalAveragePooling1D, GlobalMaxPooling1D,TimeDistributed
from keras.models import Model
import tensorflow as tf
from keras.layers import BatchNormalization
#from keras.layers.normalization import BatchNormalization
from particle_net import *
from keras import backend as K
from Layers import *
from buildingBlocks import block_deepFlavourConvolutions, block_deepFlavourConvolutions_enlarged, block_deepFlavourDense,block_deepGraphDense,block_deepGraphConvolutions


def model_deepFlavourReference(Inputs,dropoutRate=0.1,momentum=0.6):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(6, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_particle_net(Inputs,dropoutRate=0.1,momentum=0.6):

    flat = Inputs[0]
    points = Inputs[1]
    neutrals = Inputs[2]
    vertex = Inputs[3]
    num_class = 7
    conv_params = [(16, (64,64,64)),(16, (128,128,128)),(16, (256,256,256))]
    conv_pooling = 'average'
    fc_params = [(256, 0.1)]
    num_points = 25
    name = 'particle_net'
    with tf.name_scope(name):
        features = points
        glob = keras.layers.BatchNormalization()(flat)
        fts_og = tf.squeeze(keras.layers.BatchNormalization(name='%s_fts_bn' % name)(tf.expand_dims(features, axis=2)), axis=2)
        vtx = tf.squeeze(keras.layers.BatchNormalization(name='%s_vtx_bn' % name)(tf.expand_dims(vertex, axis=2)), axis=2)
        npf = tf.squeeze(keras.layers.BatchNormalization(name='%s_npf_bn' % name)(tf.expand_dims(neutrals, axis=2)), axis=2)
        points = keras.layers.Lambda( lambda x: tf.slice(x, [0,0,0], [-1,-1,2]))(fts_og)
        fts = keras.layers.Lambda( lambda x: tf.slice(x, [0,0,2], [-1,-1,-1]))(fts_og)


        npf = keras.layers.Conv1D(64,1,activation='relu')(npf)
        npf = keras.layers.Conv1D(64,1,activation='relu')(npf)
        npf = keras.layers.Conv1D(32,1,activation='relu')(npf)
        npf = keras.layers.GlobalAveragePooling1D()(npf)

        vtx = keras.layers.Conv1D(64,1,activation='relu')(vtx)
        vtx = keras.layers.Conv1D(64,1,activation='relu')(vtx)
        vtx = keras.layers.Conv1D(64,1,activation='relu')(vtx)
        pts = points
        val = num_points
        for layer_idx, layer_param in enumerate(conv_params):
            K, channels = layer_param
            if layer_idx == 1:
                concer = keras.layers.Lambda(lambda x: tf.concat([x[0],x[1]], axis=1))
                fts = concer([fts,vtx])
                val = val + 4
            pts = points if layer_idx == 0 else fts
            fts = edge_conv(pts, fts, val, K, channels, with_bn=True, activation='relu',
                            pooling=conv_pooling, name='%s_%s%d' % (name, 'EdgeConv', layer_idx))

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)
        x = keras.layers.concatenate([pool,npf,glob])
        for layer_idx, layer_param in enumerate(fc_params):
            units, drop_rate = layer_param
            x = keras.layers.Dense(units, activation='relu')(x)
            if drop_rate is not None and drop_rate > 0:
                x = keras.layers.Dropout(drop_rate)(x)
                
        out = keras.layers.Dense(num_class, activation='softmax')(x)
        reg_pred=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred')(x)
        reg_pred_down=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred_down')(x)
        reg_pred_up=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred_up')(x)
        predictions = [out, reg_pred,reg_pred_down,reg_pred_up]
        return keras.Model(inputs=[flat, features, neutrals, vertex], outputs=predictions, name='ParticleNet')


def model_deepFlavourReference_reg(Inputs,dropoutRate=0.1,momentum=0.6):
    """           
    With Regression  
    """
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    cpf,npf,vtx = block_deepFlavourConvolutions_enlarged(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    flavour_pred=Dense(7, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    reg_pred=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred')(x)
    reg_pred_down=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred_down')(x)
    reg_pred_up=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred_up')(x)
    predictions = [flavour_pred, reg_pred,reg_pred_down,reg_pred_up]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepGraphReference_reg(Inputs,dropoutRate=0.1,momentum=0.6):
    """           
    With Regression  
    """
    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    comb = block_deepGraphConvolutions(charged=cpf,
                                    neutrals=npf,
                                    vertices=vtx,
                                    dropoutRate=dropoutRate,
                                    batchnorm=True, batchmomentum=momentum)

    comb = GlobalAveragePooling1D()(comb) 
    x = Concatenate()( [globalvars,comb ])
    x = block_deepGraphDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    flavour_pred=Dense(7, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    reg_pred=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred')(x)
    reg_pred_down=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred_down')(x)
    reg_pred_up=Dense(1, activation='linear',kernel_initializer='lecun_uniform',name='reg_pred_up')(x)
    predictions = [flavour_pred, reg_pred,reg_pred_down,reg_pred_up]
    model = Model(inputs=Inputs, outputs=predictions)
    return model




def model_deepCSV(Inputs,dropoutRate=0.1):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    globalvars = Inputs[0]

    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(Inputs[0])
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)

    predictions = Dense(4, activation='softmax',kernel_initializer='lecun_uniform')(x)
    
    model = Model(inputs=Inputs, outputs=predictions)
    return model



