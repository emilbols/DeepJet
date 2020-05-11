from keras.layers import Dense, Dropout, Flatten,Concatenate, Lambda, Convolution2D, LSTM, Convolution1D, Conv2D,GlobalAveragePooling1D, GlobalMaxPooling1D,TimeDistributed
from keras.models import Model
import tensorflow as tf
from keras.layers import BatchNormalization
#from keras.layers.normalization import BatchNormalization

from keras import backend as K
from Layers import *
from buildingBlocks import block_deepFlavourConvolutions, block_deepFlavourDense, block_deepFlavourDenseSmall, block_deepFlavourConvolutionsFat,block_deepFlavourConvolutionsFat_v3,block_deepFlavourConvolutionsFat_graph, block_deepFlavourDenseFat, block_deepFlavourDenseFat_v2, block_deepFlavourDenseFattest, block_SchwartzImage, block_deepFlavourBTVConvolutions, block_deepFlavourConvolutions_PermInvariant, block_deepFlavourDense_PermInvariant, block_deepFlavourConvolutions_PermInvariant_v3, block_deepFlavourConvolutions_PermInvariant_v2, block_deepFlavourDense_PermInvariant_v2, block_deepFlavourGraph,block_deepFlavourGraphHybrid,block_deepFlavourGraphHybrid_bigger,block_deepFlavourGraphShallow, graph_block, block_deepFlavourDenseFat_v3, block_deepFlavourDenseFat_v4, block_deepFlavourConvolutionsFat_noVertex,block_graph_vertex, block_deepFlavourConvolutions_multiBranch,block_deepFlavourConvolutions_multiBranch_smooth,block_deepFlavourConvolutions_multiBranch_comb_smooth,block_deepFlavourVertexConvolutions,block_deepFlavourVertexConvolutions_noseeds

def model_crosscheck(Inputs,nclasses,nregclasses):

    globalvars = (Inputs[0])
    
    x = Dense(50,activation='relu',kernel_initializer='lecun_uniform')(globalvars)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model, Inputs


def model_deepFlavourNoNeutralReference(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[2])
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[3])
    
    cpf, vtx = block_deepFlavourBTVConvolutions(
        charged=cpf,
        vertices=vtx,
        dropoutRate=dropoutRate,
        active=True,
        batchnorm=True,
        batchmomentum=momentum
        )
    
    
    #
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,vtx ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
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
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepFlavourReference_newreg(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    seeds  =  BatchNormalization(momentum=momentum,name='seeds_input_batchnorm')     (Inputs[4])
    near  =  BatchNormalization(momentum=momentum,name='near_input_batchnorm')     (Inputs[5])

    seeds    =    Flatten()     (seeds)
    near    =     Flatten()     (near)
    seeds = Dense(1,activation='relu',kernel_initializer='lecun_uniform')(seeds)
    near = Dense(1,activation='relu',kernel_initializer='lecun_uniform')(near)

    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[6])
    
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
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput,seeds,near ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourVertexReference_noseeds(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    seeds  =  BatchNormalization(momentum=momentum,name='seeds_input_batchnorm')     (Inputs[4])
    seeds = Flatten()(seeds)
    seeds = Dense(1,kernel_initializer='lecun_uniform',  activation='relu')(seeds)
    
    near  =  BatchNormalization(momentum=momentum,name='near_input_batchnorm')     (Inputs[5])
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[6])
    
    cpf,npf,vtx,near = block_deepFlavourVertexConvolutions_noseeds(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                neighbors = near,
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
    
    near = TimeDistributed(LSTM(64,go_backwards=True,implementation=2, name='near_lstm'))(near)
    near=BatchNormalization(momentum=momentum,name='nearlstm_batchnorm')(near)
    near = Dropout(dropoutRate)(near)
    near = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='near_conv1')(near)
    near=BatchNormalization(momentum=momentum,name='nearconv1_batchnorm')(near)
    near = Dropout(dropoutRate)(near)
    near = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='near_conv2')(near)
    near = Flatten()(near)
    
    x = Concatenate()( [globalvars,cpf,npf,vtx,seeds,near ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput,seeds ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourVertexReference(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    seeds  =  BatchNormalization(momentum=momentum,name='seeds_input_batchnorm')     (Inputs[4])
    near  =  BatchNormalization(momentum=momentum,name='near_input_batchnorm')     (Inputs[5])
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[6])
    
    cpf,npf,vtx,seeds,near = block_deepFlavourVertexConvolutions(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                seeders = seeds,
                                                neighbors = near,
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
    
    seeds = LSTM(50,go_backwards=True,implementation=2, name='seeds_lstm')(seeds)
    seeds=BatchNormalization(momentum=momentum,name='seedslstm_batchnorm')(seeds)
    seeds = Dropout(dropoutRate)(seeds)
    
    near = TimeDistributed(LSTM(64,go_backwards=True,implementation=2, name='near_lstm'))(near)
    near=BatchNormalization(momentum=momentum,name='nearlstm_batchnorm')(near)
    near = Dropout(dropoutRate)(near)
    near = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='near_conv1')(near)
    near=BatchNormalization(momentum=momentum,name='nearconv1_batchnorm')(near)
    near = Dropout(dropoutRate)(near)
    near = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='near_conv2')(near)
    near = Flatten()(near)
    
    x = Concatenate()( [globalvars,cpf,npf,vtx,seeds,near ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model



def model_deepFlavour_noConv(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    #cpf,npf,vtx = block_deepFlavourConvolutions(charged=cpf,
    #                                            neutrals=npf,
    #                                            vertices=vtx,
    #                                            dropoutRate=dropoutRate,
    #                                            active=True,
    #                                            batchnorm=True, batchmomentum=momentum)
    
    
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
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_test(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    with recurrent layers and batch normalisation
    standard dropout rate it 0.1
    should be trained for flavour prediction first. afterwards, all layers can be fixed
    that do not include 'regression' and the training can be repeated focusing on the regression part
    (check function fixLayersContaining with invert=True)
    """  
    #globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    #cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    #npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    #vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    #ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])

    globalvars = (Inputs[0])
    cpf    =     (Inputs[1])
    npf    =     (Inputs[2])
    vtx    =     (Inputs[3])
    ptreginput = (Inputs[4])

  
    cpf  = Flatten()(cpf)    
    npf  = Flatten()(npf)
    vtx  = Flatten()(vtx)
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = Dense(200,activation='relu',kernel_initializer='lecun_uniform')(x)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_Graph(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourGraph(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    cpf  = GlobalAveragePooling1D()(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpfsum_batchnorm')(cpf)
    
    npf = GlobalAveragePooling1D()(npf)
    npf=BatchNormalization(momentum=momentum,name='npfsum_batchnorm')(npf)
    
    vtx = GlobalAveragePooling1D()(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxsum_batchnorm')(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDenseFattest(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepGraph_v1(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):


    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf = graph_block(cpf, 64, True, batchmomentum=momentum)
    cpf = graph_block(cpf, 128, True, batchmomentum=momentum)
    cpf_skip = cpf
    cpf = graph_block(cpf, 256,True,batchmomentum=momentum)

    npf = graph_block(npf, 64, False, batchmomentum=momentum)
    npf = graph_block(npf, 64, False, batchmomentum=momentum)
    npf = graph_block(npf, 64, False, batchmomentum=momentum)

    vtx = graph_block(vtx,64, False, batchmomentum=momentum)
    vtx = graph_block(vtx,128, False, batchmomentum=momentum)
    vtx_skip = vtx
    vtx = graph_block(vtx,64, False, batchmomentum=momentum)
  
    concer = Lambda(lambda x: K.concatenate([x[0],x[1]], axis=1))
    comb =   concer([vtx_skip,cpf_skip])
    comb = graph_block(comb, 256, True, batchmomentum=momentum)


    cpf  = GlobalAveragePooling1D()(cpf)
    
    npf = GlobalAveragePooling1D()(npf)
    
    vtx = GlobalAveragePooling1D()(vtx)

    comb = GlobalAveragePooling1D()(comb)
        

    z = Concatenate()( [globalvars,cpf,npf,vtx, comb])
    z = Dense(256,activation='relu',kernel_initializer='lecun_uniform')(z)
    z = Dropout(dropoutRate)(z)
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(z)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepGraph_v2(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):


    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    npf    =     BatchNormalization(momentum=momentum,name='npf_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf = graph_block(cpf, 128, True, batchmomentum=momentum)
    cpf = graph_block(cpf, 256, True, batchmomentum=momentum)
    cpf = graph_block(cpf, 256, True, batchmomentum=momentum)
    cpf_skip = cpf
    cpf = graph_block(cpf, 256,True,batchmomentum=momentum)

    npf = graph_block(npf, 64, False, batchmomentum=momentum)
    npf = graph_block(npf, 64, False, batchmomentum=momentum)
    npf = graph_block(npf, 64, False, batchmomentum=momentum)

    vtx = graph_block(vtx,128, False, batchmomentum=momentum)
    vtx = graph_block(vtx,256, False, batchmomentum=momentum)
    vtx_skip = vtx
    vtx = graph_block(vtx,64, False, batchmomentum=momentum)
  
    concer = Lambda(lambda x: K.concatenate([x[0],x[1]], axis=1))
    comb =   concer([vtx_skip,cpf_skip])
    comb = graph_block(comb, 256, True, batchmomentum=momentum)
    comb = graph_block(comb, 256, True, batchmomentum=momentum)


    cpf_avg  = GlobalAveragePooling1D()(cpf)
    cpf_max  = GlobalMaxPooling1D()(cpf)
    
    npf_avg = GlobalAveragePooling1D()(npf)
    npf_max = GlobalMaxPooling1D()(npf)
    
    vtx_avg = GlobalAveragePooling1D()(vtx)
    vtx_max = GlobalMaxPooling1D()(vtx)

    comb_avg = GlobalAveragePooling1D()(comb)
    comb_max = GlobalAveragePooling1D()(comb)
        

    z = Concatenate()( [globalvars,cpf_avg,cpf_max,npf_avg,npf_max,vtx_avg,vtx_max, comb_avg,comb_max])
    z = Dense(256,activation='relu',kernel_initializer='lecun_uniform')(z)
    z = Dense(512,activation='relu',kernel_initializer='lecun_uniform')(z)
    z = Dropout(dropoutRate)(z)
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(z)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model



def model_deepFlavourReference_GraphHybrid(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    comb1,comb2 = block_deepFlavourGraphHybrid(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    comb1 = SortLayer_noweights()(comb1)
    comb1  = LSTM(250,go_backwards=True,implementation=2, name='comb1_lstm')(comb1)
    comb1 = BatchNormalization(momentum=momentum,name='comb1_lstm_batchnorm')(comb1)
    comb1 = Dropout(dropoutRate)(comb1)
    
    comb2 = SortLayer_noweights()(comb2)
    comb2  = LSTM(100,go_backwards=True,implementation=2, name='comb2_lstm')(comb2)
    comb2 = BatchNormalization(momentum=momentum,name='comb2_lstm_batchnorm')(comb2)
    comb2 = Dropout(dropoutRate)(comb2)
    
    
    x = Concatenate()( [globalvars,comb1,comb2 ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_GraphHybrid_bigger(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    comb, npf = block_deepFlavourGraphHybrid_bigger(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    
    comb  = GlobalAveragePooling1D()(comb)
    npf  = GlobalAveragePooling1D()(npf)
   

    x = Concatenate()( [globalvars,comb,npf ])
    
    x = block_deepFlavourDenseFat(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepFlavourReference_GraphShallow(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourGraphShallow(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    cpf  = GlobalAveragePooling1D()(cpf)
    #cpf  = Flatten()(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpfsum_batchnorm')(cpf)
    
    npf = GlobalAveragePooling1D()(npf)
    #npf = Flatten()(npf)
    npf=BatchNormalization(momentum=momentum,name='npfsum_batchnorm')(npf)
    
    vtx = GlobalAveragePooling1D()(vtx)
    #vtx = Flatten()(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxsum_batchnorm')(vtx)
        
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDenseFat(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepFlavourReference_GraphShallow_noreg(Inputs,nclasses,dropoutRate=0.1,momentum=0.6):
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
    
    cpf,npf,vtx = block_deepFlavourGraphShallow(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    cpf  = GlobalAveragePooling1D()(cpf)
    #cpf  = Flatten()(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpfsum_batchnorm')(cpf)
    
    npf = GlobalAveragePooling1D()(npf)
    #npf = Flatten()(npf)
    npf=BatchNormalization(momentum=momentum,name='npfsum_batchnorm')(npf)
    
    vtx = GlobalAveragePooling1D()(vtx)
    #vtx = Flatten()(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxsum_batchnorm')(vtx)
        
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDenseFat(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_PermInvariant(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourConvolutions_PermInvariant(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    cpf  = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpfsum_batchnorm')(cpf)
    
    npf = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(npf)
    npf=BatchNormalization(momentum=momentum,name='npfsum_batchnorm')(npf)
    
    vtx = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxsum_batchnorm')(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDense_PermInvariant(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepFlavourReference_PermInvariant_v2(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourConvolutions_PermInvariant_v2(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    cpf  = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpfsum_batchnorm')(cpf)
    
    npf = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(npf)
    npf=BatchNormalization(momentum=momentum,name='npfsum_batchnorm')(npf)
    
    vtx = Lambda(lambda x: K.sum(x, axis=1), output_shape=lambda s: (s[0], s[2]))(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxsum_batchnorm')(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDense_PermInvariant_v2(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model



def model_deepFlavourReference_PermInvariant_v3(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    comb = block_deepFlavourConvolutions_PermInvariant_v3(charged=cpf,
                                                          neutrals=npf,
                                                          vertices=vtx,
                                                          dropoutRate=dropoutRate,
                                                          active=True,
                                                          batchnorm=True, batchmomentum=momentum)
    

    #concer = Lambda(lambda x: K.concatenate([K.expand_dims(x[0],axis=2),K.expand_dims(x[1],axis=2),K.expand_dims(x[2],axis=2)]))
    concer = Lambda(lambda x: K.concatenate([K.expand_dims(x[0],axis=2),K.expand_dims(x[1],axis=2)]))
    pool = GlobalAveragePooling1D()(comb)
    max_pool = GlobalMaxPooling1D()(comb)
    comb = concer([pool,max_pool])
    comb=BatchNormalization(momentum=momentum,name='combsum_batchnorm')(comb)
    comb = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='conv_again')(comb)
    comb=BatchNormalization(momentum=momentum,name='combsum_batchnorm1')(comb)
    comb = Convolution1D(1, 1, kernel_initializer='lecun_uniform',  activation='relu', name='conv_again1')(comb)
    comb = Flatten()(comb)

    
    
    x = Concatenate()( [globalvars,comb ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_NoRNN(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    

    cpf=Flatten()(cpf)
    npf=Flatten()(npf)
    vtx=Flatten()(vtx)
      
    x = Concatenate()( [globalvars,cpf,npf,vtx ])

    x = Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense')(x)
    x = BatchNormalization(momentum=momentum,name='df_batchnorm')(x)
    x = Dropout(dropoutRate,name='df_dense_dropout')(x)
    

    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_sort(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf = SortLayer()(cpf)
    npf = SortLayer()(npf)
    vtx = SortLayer()(vtx)

    cpf    =     BatchNormalization(momentum=momentum,name='cpf_batchnorm_sort')     (cpf)
    npf    =     BatchNormalization(momentum=momentum,name='npf_batchnorm_sort')     (npf)
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_batchnorm_sort')     (vtx)
    

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
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_fat(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourConvolutionsFat(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    cpf  = LSTM(300,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(100,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(150,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDenseFat_v2(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepFlavourReference_fat_v2(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourConvolutionsFat_graph(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    #
    cpf  = LSTM(350,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(100,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(200,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDenseFat_v3(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def model_deepFlavourReference_fat_v3(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourConvolutionsFat_v3(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    cpf = SortLayer_noweights()(cpf)
    cpf  = LSTM(350,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = SortLayer_noweights()(npf)
    npf = LSTM(100,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = SortLayer_noweights()(vtx)
    vtx = LSTM(200,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDenseFat_v3(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_fat_smooth_v4(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf,npf,vtx = block_deepFlavourConvolutions_multiBranch_smooth(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    
    cpf = SmoothSort()(cpf)
    cpf  = LSTM(200,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    

    npf = SmoothSort()(npf)
    npf = LSTM(100,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = SmoothSort()(vtx)
    vtx = LSTM(100,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDenseFat_v4(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_fat_comb_smooth_v4(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf1,cpf2,cpf3,npf,vtx = block_deepFlavourConvolutions_multiBranch_comb_smooth(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    cpf1 = ParticleSort()(cpf1)
    cpf1  = LSTM(100,go_backwards=True,implementation=2, name='cpf1_lstm')(cpf1)
    cpf1=BatchNormalization(momentum=momentum,name='cpf1lstm_batchnorm')(cpf1)
    cpf1 = Dropout(dropoutRate)(cpf1)
    
    cpf2 = ParticleSort()(cpf2)
    cpf2  = LSTM(150,go_backwards=True,implementation=2, name='cpf2_lstm')(cpf2)
    cpf2=BatchNormalization(momentum=momentum,name='cpf2lstm_batchnorm')(cpf2)
    cpf2 = Dropout(dropoutRate)(cpf2)
    
    cpf3 = ParticleSort()(cpf3)
    cpf3  = LSTM(200,go_backwards=True,implementation=2, name='cpf3_lstm')(cpf3)
    cpf3=BatchNormalization(momentum=momentum,name='cpf3lstm_batchnorm')(cpf3)
    cpf3 = Dropout(dropoutRate)(cpf3)
    

    npf = ParticleSort()(npf)
    npf = LSTM(100,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = ParticleSort()(vtx)
    vtx = LSTM(100,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf1,cpf2,cpf3,npf,vtx ])
    
    x = block_deepFlavourDenseFat_v4(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourReference_fat_v4(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    ptreginput = BatchNormalization(momentum=momentum,name='reg_input_batchnorm')     (Inputs[4])
    
    cpf1,cpf2,cpf3,npf,vtx = block_deepFlavourConvolutions_multiBranch(charged=cpf,
                                                neutrals=npf,
                                                vertices=vtx,
                                                dropoutRate=dropoutRate,
                                                active=True,
                                                batchnorm=True, batchmomentum=momentum)
    
    
    cpf1 = SortLayer_noweights()(cpf1)
    cpf1  = LSTM(100,go_backwards=True,implementation=2, name='cpf1_lstm')(cpf1)
    cpf1=BatchNormalization(momentum=momentum,name='cpf1lstm_batchnorm')(cpf1)
    cpf1 = Dropout(dropoutRate)(cpf1)
    
    cpf2 = SortLayer_noweights()(cpf2)
    cpf2  = LSTM(150,go_backwards=True,implementation=2, name='cpf2_lstm')(cpf2)
    cpf2=BatchNormalization(momentum=momentum,name='cpf2lstm_batchnorm')(cpf2)
    cpf2 = Dropout(dropoutRate)(cpf2)
    
    cpf3 = SortLayer_noweights()(cpf3)
    cpf3  = LSTM(200,go_backwards=True,implementation=2, name='cpf3_lstm')(cpf3)
    cpf3=BatchNormalization(momentum=momentum,name='cpf3lstm_batchnorm')(cpf3)
    cpf3 = Dropout(dropoutRate)(cpf3)
    

    npf = SortLayer_noweights()(npf)
    npf = LSTM(100,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = SortLayer_noweights()(vtx)
    vtx = LSTM(100,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    
    x = Concatenate()( [globalvars,cpf1,cpf2,cpf3,npf,vtx ])
    
    x = block_deepFlavourDenseFat_v4(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    reg = Concatenate()( [flavour_pred, ptreginput ] ) 
    
    reg_pred=Dense(nregclasses, activation='linear',kernel_initializer='ones',name='regression_pred',trainable=True)(reg)
    
    predictions = [flavour_pred,reg_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_deepFlavourSmart(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
        
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model



def model_deepFlavourSmart_noRNN(Inputs,nclasses,nregclasses,dropoutRate=0.1,momentum=0.6):
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
    cpf=BatchNormalization(momentum=momentum,name='cpflstm_batchnorm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf=Flatten()(cpf)
    
    npf=BatchNormalization(momentum=momentum,name='npflstm_batchnorm')(npf)
    npf = Dropout(dropoutRate)(npf)
    npf=Flatten()(npf)
      
 
    vtx=BatchNormalization(momentum=momentum,name='vtxlstm_batchnorm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx=Flatten()(vtx)
    
    
    x = Concatenate()( [globalvars,cpf,npf,vtx ])
    
    x = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
        
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_DeepCSV_RNN(Inputs,nclasses,nregclasses,dropoutRate=-1,momentum = 0.6):
    


    globalvars = BatchNormalization(momentum=0.6,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=0.6,name='cpf_input_batchnorm')     (Inputs[1])
    eta_rel    =     BatchNormalization(momentum=0.6,name='eta_rel_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=0.6,name='vtx_input_batchnorm')     (Inputs[3])
    cpf = Lambda( lambda x: tf.slice(x, [0,0,0], [-1,6,-1]))(cpf)

    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    


    eta_rel  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='eta_rel_conv0')(eta_rel)
    eta_rel = Dropout(dropoutRate)(eta_rel)                                                   
    eta_rel  = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='eta_rel_conv1')(eta_rel)
    eta_rel = Dropout(dropoutRate)(eta_rel)                                                   
    eta_rel  = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu', name='eta_rel_conv2')(eta_rel)
    test = True
    vtx = Lambda( lambda x: tf.slice(x, [0,0,0], [-1,1,-1]))(vtx)
    if test:
        vtx = Flatten()(vtx)
        vtx = Dense(64, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Dense(32, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Dense(32, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Dense(8, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        vtx = Dropout(dropoutRate)(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    

    eta_rel  = Flatten()(eta_rel)
    
    if not test:
        vtx  = Flatten()(vtx)
    
        
    x = Concatenate()( [globalvars,cpf,eta_rel,vtx ])
        
    x  = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)

    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    

    model = Model(inputs=Inputs, outputs=predictions)
    return model


def model_DeepCSV_more_RNN(Inputs,nclasses,nregclasses,dropoutRate=-1,momentum = 0.6):
    


    globalvars = BatchNormalization(momentum=momentum,name='globals_input_batchnorm') (Inputs[0])
    cpf    =     BatchNormalization(momentum=momentum,name='cpf_input_batchnorm')     (Inputs[1])
    eta_rel    =     BatchNormalization(momentum=momentum,name='eta_rel_input_batchnorm')     (Inputs[2])
    vtx    =     BatchNormalization(momentum=momentum,name='vtx_input_batchnorm')     (Inputs[3])

    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    


    eta_rel  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='eta_rel_conv0')(eta_rel)
    eta_rel = Dropout(dropoutRate)(eta_rel)                                                   
    eta_rel  = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='eta_rel_conv1')(eta_rel)
    eta_rel = Dropout(dropoutRate)(eta_rel)                                                   
    eta_rel  = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu', name='eta_rel_conv2')(eta_rel)
    
    
    vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    

    eta_rel  = Flatten()(eta_rel)
    
    vtx  = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    x = Concatenate()( [globalvars,cpf,eta_rel,vtx ])
        
    x  = block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=True,batchmomentum=momentum)

    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    

    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_deepcsv(Inputs,nclasses,nregclasses,dropoutRate=-1):
    
    cpf=Inputs[1]
    vtx=Inputs[2]
    
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
    cpf = Dropout(dropoutRate)(cpf)                                                   
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    
    
    vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    
    cpf=Flatten()(cpf)
    vtx=Flatten()(vtx)
        
    x = Concatenate()( [Inputs[0],cpf,vtx ])
        
    x  = block_deepFlavourDense(x,dropoutRate)

    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_broad(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour', as for DPS note
    """  
   
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    image = block_SchwartzImage(image=Inputs[4],dropoutRate=dropoutRate,active=False)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,image ])
    
    x  = block_deepFlavourDense(x,dropoutRate)

    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_broad_map(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    """  
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    image = block_SchwartzImage(image=Inputs[4],dropoutRate=dropoutRate)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,image ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def convolutional_model_broad_map_reg(Inputs,nclasses,nregclasses,dropoutRate):
    """
    reference 1x1 convolutional model for 'deepFlavour'
    """  
    
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    image = block_SchwartzImage(image=Inputs[4],dropoutRate=dropoutRate)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,image,Inputs[5] ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    predictions = [Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x),
                   Dense(nregclasses, activation='linear',kernel_initializer='ones',name='E_pred')(x)]
    model = Model(inputs=Inputs, outputs=predictions)
    return model




def convolutional_model_broad_reg(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    the inputs are really not working as they are. need a reshaping well before
    """  
    cpf,npf,vtx = block_deepFlavourConvolutions(charged=Inputs[1],
                                                neutrals=Inputs[2],
                                                vertices=Inputs[3],
                                                dropoutRate=dropoutRate)
    
    
    cpf  = LSTM(150,go_backwards=True,implementation=2, name='cpf_lstm')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    
    npf = LSTM(50,go_backwards=True,implementation=2, name='npf_lstm')(npf)
    npf = Dropout(dropoutRate)(npf)
    
    vtx = LSTM(50,go_backwards=True,implementation=2, name='vtx_lstm')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx,Inputs[4] ])
    
    x  = block_deepFlavourDense(x,dropoutRate)
    
    
    predictions = [Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x),
                   Dense(nregclasses, activation='linear',kernel_initializer='ones',name='E_pred')(x)]
                   
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_broad_reg2(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    Flavour tagging and regression in one model. Fully working
    """
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Flatten()(cpf)
    
    
    npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Flatten()(npf)
    
    vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[3])
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Flatten()(vtx)
    
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx ] )
    

    x=  Dense(350, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    flav=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    
    
    
    
    ptcpf  = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    ptcpf = Dropout(dropoutRate)(ptcpf)
    ptcpf = Flatten()(ptcpf)
    ptnpf  = Convolution1D(2, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    ptnpf = Dropout(dropoutRate)(ptnpf)
    ptnpf = Flatten()(ptnpf)
   
    xx=Concatenate()( [Inputs[4],flav,ptcpf,ptnpf] )
    xx=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(xx)
    
    ptandsigma=Dense(2, activation='linear',kernel_initializer='lecun_uniform')(xx)
    
    predictions = [flav,ptandsigma]
    model = Model(inputs=Inputs, outputs=predictions)
    return model


def convolutional_model_lessbroad(Inputs,nclasses,nregclasses,dropoutRate=-1):
    """
    the inputs are really not working as they are. need a reshaping well before
    """
   
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform',input_shape=Inputshapes[0])(Inputs[0])
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform')(gl)
    #gl = Dense(8, activation='relu',kernel_initializer='lecun_uniform')(gl)
    
    
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf  = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(cpf)
    cpf = Dropout(dropoutRate)(cpf)
    cpf = Flatten()(cpf)
    
    
    npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    npf = Dropout(dropoutRate)(npf)
    npf = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(npf)
    npf = Dropout(dropoutRate)(npf)
    npf = Flatten()(npf)
    
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[3])
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu')(vtx)
    vtx = Dropout(dropoutRate)(vtx)
    vtx = Flatten()(vtx)
    
    x = Concatenate()( [Inputs[0],cpf,npf,vtx ] )
    x = Dropout(dropoutRate)(x)

    x=  Dense(600, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model

def convolutional_model_ConvCSV(Inputs,nclasses,nregclasses,dropoutRate=0.25):
    """
    Inputs similar to 2016 training, but with covolutional layers on each track and sv
    """
    
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[1])
    a = Dropout(dropoutRate)(a)
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(a)
    a = Dropout(dropoutRate)(a)
    a  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(a)
    a = Dropout(dropoutRate)(a)
    a=Flatten()(a)
    
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(Inputs[2])
    c = Dropout(dropoutRate)(c)
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(c)
    c = Dropout(dropoutRate)(c)
    c  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu')(c)
    c = Dropout(dropoutRate)(c)
    c=Flatten()(c)
    
    x = Concatenate()( [Inputs[0],a,c] )
    
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform')(x)
    predictions = Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform')(x)
    model = Model(inputs=Inputs, outputs=predictions)
    return model
