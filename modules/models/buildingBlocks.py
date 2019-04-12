'''
standardised building blocks for the models
'''

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

def block_deepFlavourBTVConvolutions(charged,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    vtx = vertices
    if active:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,vtx


def block_deepFlavourConvolutions(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx


def block_deepFlavourConvolutions_PermInvariant(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv_prot')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm_prot')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout_prot')(cpf)                                                   
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
        npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv_prot')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm_prot')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout_prot')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx

def block_deepFlavourConvolutions_PermInvariant_v2(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm3')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout3')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv4')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm4')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout4')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv5')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm5')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout5')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv6')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm6')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout6')(cpf)                                                   
        cpf  = Convolution1D(512, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv7')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
        npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm2')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout2')(npf)
        npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv3')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm3')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout3')(npf)
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv4')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm3')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout3')(vtx)
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv4')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm4')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout4')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv5')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx


def block_deepFlavourConvolutionsFat(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(512, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_convStart')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnormStart')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropoutStart')(cpf)                                                   
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_convStart1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnormStart1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropoutStart1')(cpf)                                                   
        cpf  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_convStart2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnormStart2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropoutStart2')(cpf)                                                   
        cpf  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_convStart3')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnormStart3')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropoutStart3')(cpf)                                                   
        cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_convmore1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnormmore1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropoutmore1')(npf) 
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_convmore2')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnormmore2')(npf)
        npf = Dropout(dropoutRate,name='npf_dropoutmore2')(npf) 
        npf = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_convFirst')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnormFirst')(npf)
        npf = Dropout(dropoutRate,name='npf_dropoutFirst')(npf) 
        npf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_convFirst1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnormFirst1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropoutFirst1')(npf) 
        npf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_convFirst2')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnormFirst2')(npf)
        npf = Dropout(dropoutRate,name='npf_dropoutFirst2')(npf) 
        npf = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
        npf = Convolution1D(16, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(4, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(512, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_convmore')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnormmore1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropoutmore1')(vtx) 
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_convFirst')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnormFirst')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropoutFirst')(vtx) 
        vtx = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_convFirst0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnormFirst0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropoutFirst0')(vtx) 
        vtx = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_convFirst1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnormFirst1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropoutFirst1')(vtx) 
        vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(32, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(8, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx


def block_deepFlavourDenseFat(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(500, activation='relu',kernel_initializer='lecun_uniform', name='df_dense')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense8')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm8')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout8')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense9')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm9')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout9')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

def block_deepFlavourDense(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)

    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

def block_deepFlavourDense_PermInvariant(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout5')(x) 
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout6')(x) 
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout7')(x) 
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense8')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm8')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout8')(x) 
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense9')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm9')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout9')(x) 
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense10')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm10')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout10')(x) 
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense11')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm11')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout11')(x) 
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense12')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm12')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout12')(x) 
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

def block_deepFlavourDense_PermInvariant_v2(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(300, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout2')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout3')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout4')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout5')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout6')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense7')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm7')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout7')(x)

    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x


def block_SchwartzImage(image,dropoutRate,active=True):
    '''
    returns flattened output
    '''
    
    if active:
        image =   Convolution2D(64, (8,8)  , border_mode='same', activation='relu',
                                kernel_initializer='lecun_uniform', name='swz_conv0')(image)
        image = MaxPooling2D(pool_size=(2, 2), name='swz_maxpool0')(image)
        image = Dropout(dropoutRate)(image)
        image =   Convolution2D(64, (4,4) , border_mode='same', activation='relu',
                                kernel_initializer='lecun_uniform', name='swz_conv1')(image)
        image = MaxPooling2D(pool_size=(2, 2), name='swz_maxpool1')(image)
        image = Dropout(dropoutRate)(image)
        image =   Convolution2D(64, (4,4)  , border_mode='same', activation='relu',
                                kernel_initializer='lecun_uniform', name='swz_conv2')(image)
        image = MaxPooling2D(pool_size=(2, 2), name='swz_maxpool2')(image)
        image = Dropout(dropoutRate)(image)
        image = Flatten()(image)

    else:
        #image=Cropping2D(crop)(image)#cut almost all of the 20x20 pixels
        image = Flatten()(image)
        image = Dense(1,kernel_initializer='zeros',trainable=False, name='swz_conv_off')(image)#effectively multipy by 0
        
    return image
