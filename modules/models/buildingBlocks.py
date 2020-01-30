'''
standardised building blocks for the models
'''

from keras.layers import Dense, Dropout, Flatten,Convolution2D, Convolution1D, Lambda, LeakyReLU,Reshape
#from keras.layers.pooling import MaxPooling2D
from keras.layers import MaxPool2D
#from keras.layers.normalization import BatchNormalization
from keras.layers import BatchNormalization
from Layers import *

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


def block_deepFlavourVertexConvolutions(charged,neutrals,vertices,seeders,neighbors,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
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
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    seeds=seeders
    if active:
        seeds  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='seeds_conv0')(seeds)
        if batchnorm:
            seeds = BatchNormalization(momentum=batchmomentum ,name='seeds_batchnorm0')(seeds)
        seeds = Dropout(dropoutRate,name='seeds_dropout0')(seeds)                                                   
        seeds  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='seeds_conv1')(seeds)
        if batchnorm:
            seeds = BatchNormalization(momentum=batchmomentum,name='seeds_batchnorm1')(seeds)
        seeds = Dropout(dropoutRate,name='seeds_dropout1')(seeds)                                                   
        seeds  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='seeds_conv2')(seeds)
        if batchnorm:
            seeds = BatchNormalization(momentum=batchmomentum,name='seeds_batchnorm2')(seeds)
        seeds = Dropout(dropoutRate,name='seeds_dropout2')(seeds)                                                   
        seeds  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='seeds_conv3')(seeds)
    else:
        seeds = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(seeds)
        
    near=neighbors
    near = Reshape((5,20,-1))(near)
    if active:
        near  = Convolution2D(100, (1,1), kernel_initializer='lecun_uniform',  activation='relu', name='near_conv0')(near)
        if batchnorm:
            near = BatchNormalization(momentum=batchmomentum ,name='near_batchnorm0')(near)
    else:
        near = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(near)
        


    return cpf,npf,vtx,seeds,near


def block_deepFlavourVertexConvolutions_noseeds(charged,neutrals,vertices,neighbors,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
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
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

  
    near=neighbors
    near = Reshape((5,20,-1))(near)
    if active:
        near  = Convolution2D(100, (1,1), kernel_initializer='lecun_uniform',  activation='relu', name='near_conv0')(near)
        if batchnorm:
            near = BatchNormalization(momentum=batchmomentum ,name='near_batchnorm0')(near)
    else:
        near = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(near)
        


    return cpf,npf,vtx,near




def graph_block(features, dim_size,with_graph,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    x = features
    x  = Convolution1D(dim_size, 1, kernel_initializer='lecun_uniform',  activation='relu')(x)
    x = BatchNormalization(momentum=batchmomentum)(x)
    x  = Convolution1D(dim_size, 1, kernel_initializer='lecun_uniform',  activation='relu')(x)
    x = BatchNormalization(momentum=batchmomentum)(x)
    x  = Convolution1D(dim_size, 1, kernel_initializer='lecun_uniform',  activation='relu')(x)
    x = BatchNormalization(momentum=batchmomentum)(x)
    if with_graph:
        x  = GravNet(10,256,dim_size, 256)(x)
        x = BatchNormalization(momentum=batchmomentum)(x)                                               
    return x


def block_deepFlavourGraph(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = GravNet(5,2,100, 100, name='cpf_grav0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = GravNet(5,2,100, 100, name='cpf_grav1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm3')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout3')(cpf)                                                   
        cpf  = GravNet(5,2,128, 100, name='cpf_grav2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm4')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout4')(cpf)                                                   
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='cpf_conv2')(cpf)
    else:
        cpf = GravNet(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf  = GravNet(5,2,100, 40, name='npf_grav0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf)
        npf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf  = GravNet(5,2,100, 40, name='npf_grav1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm2')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout2')(npf)
        npf  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='npf_conv3')(npf)
    else:
        npf = GravNet(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx)
        vtx  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = GravNet(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx



def block_deepFlavourGraphHybrid(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
    if batchnorm:
        cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
    cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
    if batchnorm:
        cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm1')(cpf)
    cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
    if batchnorm:
        cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm2')(cpf)
    cpf  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv3')(cpf)
        
    npf=neutrals
    npf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
    if batchnorm:
        npf = BatchNormalization(momentum=batchmomentum ,name='npf_batchnorm0')(npf)
    npf = Dropout(dropoutRate,name='npf_dropout0')(npf)
    npf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
    if batchnorm:
        npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
    npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
    npf  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv2')(npf)
    
    vtx = vertices

    vtx  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
    if batchnorm:
        vtx = BatchNormalization(momentum=batchmomentum ,name='vtx_batchnorm0')(vtx)
    vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx)
    vtx  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
    if batchnorm:
        vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
    vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
    vtx  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
    if batchnorm:
        vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
    vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
    vtx  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)

    concer = Lambda(lambda x: K.concatenate([x[0],x[1]], axis=1))

    comb =   concer([cpf,vtx])
    comb  = GravNet(8, 64, 128, 100, name='comb_grav')(comb)
    comb = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv')(comb)

    comb2 = concer([cpf,npf])
    comb2  = GravNet(8, 64, 64, 100, name='comb2_grav')(comb2)
    comb2 = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb2_conv')(comb2)


    return comb,comb2


def block_deepFlavourGraphHybrid_bigger(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
    if batchnorm:
        cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
    cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
    if batchnorm:
        cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm1')(cpf)
    cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        
    npf=neutrals
    npf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
    if batchnorm:
        npf = BatchNormalization(momentum=batchmomentum ,name='npf_batchnorm0')(npf)
    npf = Dropout(dropoutRate,name='npf_dropout0')(npf)
    npf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
    if batchnorm:
        npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
    npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
    npf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv2')(npf)
    
    vtx = vertices

    vtx  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
    if batchnorm:
        vtx = BatchNormalization(momentum=batchmomentum ,name='vtx_batchnorm0')(vtx)
    vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx)
    vtx  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
    if batchnorm:
        vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
    vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
    vtx  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
 
    concer = Lambda(lambda x: K.concatenate([x[0],x[1]], axis=1))
    permute = Lambda(lambda x: K.permute_dimensions(x,(0,2,1)))

    comb =   concer([cpf,vtx])
    comb  = GravNet(7, 3, 64, 512, name='comb_grav0')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum ,name='comb_batchnorm0')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout0')(comb)
    comb = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv0')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum ,name='comb_batchnorm1')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout1')(comb)
    comb = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv1')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum ,name='comb_batchnorm2')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout2')(comb)
    comb  = GravNet(7, 3, 128, 512, name='comb_grav1')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum ,name='comb_batchnorm3')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout3')(comb)
    comb = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv2')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum ,name='comb_batchnorm4')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout4')(comb)
    comb = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv3')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum ,name='comb_batchnorm5')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout5')(comb)
    comb  = GravNet(20, 3, 256, 512, name='comb_grav2')(comb)
    return comb, npf



def block_deepFlavourGraphShallow(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm0')(cpf)
        cpf  = Dropout(dropoutRate,name='cpf_dropout0')(cpf)
        cpf  = GravNet(5, 3, 100, 100, name='cpf_grav0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf  = Dropout(dropoutRate,name='cpf_dropout1')(cpf)
        cpf  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='cpf_conv1')(cpf)
    else:
        cpf = GravNet(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf  = Dropout(dropoutRate,name='npf_dropout0')(npf)
        npf  = GravNet(3, 3, 100, 100, name='npf_grav0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf  = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='tanh', name='npf_conv1')(npf)
    else:
        npf = GravNet(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx) 
        vtx  = Convolution1D(128, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
    else:
        vtx = GravNet(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx

def block_deepFlavourConvolutions_multiBranch(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
   
    cpf=charged
    if active:
        cpf  = Convolution1D(150, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf  = Convolution1D(150, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm1')(cpf)
        cpf  = Convolution1D(150, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        cpf = SortLayer_noweights()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(256, 3, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv3')(cpf)
        cpf1 = cpf
        cpf = SortLayer_noweights()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm3')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(256, 3, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv4')(cpf)
        cpf2 = cpf
        cpf = SortLayer_noweights()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm4')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout3')(cpf)                                                   
        cpf3  = Convolution1D(256, 3, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv5')(cpf)
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
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf1,cpf2,cpf3,npf,vtx

def block_deepFlavourConvolutions_multiBranch_smooth(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
   
    cpf=charged
    if active:
        cpf  = Convolution1D(150, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf  = Convolution1D(150, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm1')(cpf)
        cpf  = GravNet(7, 3, 150, 150, name='grav0')(cpf)
        cpf = SmoothSort()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)                       
        cpf  = Convolution1D(256, 5, kernel_initializer='lecun_uniform', activation='relu', name='cpf_conv3')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm3')(cpf)                       
        cpf  = GravNet(7, 3, 150, 150, name='grav1')(cpf)
        cpf = SmoothSort()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm4')(cpf)                       
        cpf  = Convolution1D(256, 5, kernel_initializer='lecun_uniform', activation='relu', name='cpf_conv5')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm5')(cpf) 
        cpf  = GravNet(7, 3, 150, 150, name='grav2')(cpf)        
        cpf = SmoothSort()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm6')(cpf)
        cpf  = Convolution1D(256, 5, kernel_initializer='lecun_uniform', name='cpf_conv7')(cpf)
        cpf = LeakyReLU(alpha=0.1)(cpf)
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
        npf = Convolution1D(128, 1, kernel_initializer='lecun_uniform', name='npf_conv2')(npf)
        npf = LeakyReLU(alpha=0.1)(npf)
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
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform', name='vtx_conv3')(vtx)
        vtx = LeakyReLU(alpha=0.1)(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx

def block_deepFlavourConvolutions_multiBranch_comb_smooth(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
   
    cpf=charged
    if active:
        cpf  = Convolution1D(150, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf  = Convolution1D(150, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm1')(cpf)
        cpf  = Convolution1D(150, 1, kernel_initializer='lecun_uniform', name='cpf_conv2')(cpf)
        cpf = LeakyReLU(alpha=0.1)(cpf)
        cpf = ParticleSort()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)                       
        cpf  = Convolution1D(256, 3, kernel_initializer='lecun_uniform', name='cpf_conv3')(cpf)
        cpf = LeakyReLU(alpha=0.1)(cpf)
        cpf1 = cpf
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm3')(cpf)                       
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform', name='cpf_conv4')(cpf)
        cpf = LeakyReLU(alpha=0.1)(cpf)
        cpf = ParticleSort()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm4')(cpf)                       
        cpf  = Convolution1D(256, 3, kernel_initializer='lecun_uniform', name='cpf_conv5')(cpf)
        cpf = LeakyReLU(alpha=0.1)(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm5')(cpf) 
        cpf2 = cpf
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform', name='cpf_conv6')(cpf)
        cpf = LeakyReLU(alpha=0.1)(cpf)
        cpf = ParticleSort()(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm6')(cpf)
        cpf  = Convolution1D(256, 3, kernel_initializer='lecun_uniform', name='cpf_conv7')(cpf)
        cpf3 = LeakyReLU(alpha=0.1)(cpf)
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
        npf = Convolution1D(128, 1, kernel_initializer='lecun_uniform', name='npf_conv2')(npf)
        npf = LeakyReLU(alpha=0.1)(npf)
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
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform', name='vtx_conv3')(vtx)
        vtx = LeakyReLU(alpha=0.1)(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf1,cpf2,cpf3,npf,vtx


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
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv3')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm3')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout3')(cpf)                                                   
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv4')(cpf)
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
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
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


def block_deepFlavourConvolutions_PermInvariant_v3(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
    if batchnorm:
        cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
    cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
    cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
    if batchnorm:
        cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm1')(cpf)
    cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
    cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
    if batchnorm:
        cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm2')(cpf)
    cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
    cpf  = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv3')(cpf)
        
    npf = neutrals
    npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
    if batchnorm:
        npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
    npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
    npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
    if batchnorm:
        npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
    npf = Dropout(dropoutRate,name='npf_dropout1')(npf) 
    npf = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv2')(npf)
        
    vtx = vertices
    vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
    if batchnorm:
        vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
    vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
    vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
    if batchnorm:
        vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
    vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx) 
    vtx = Convolution1D(64, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        

    concer = Lambda(lambda x: K.concatenate([x[0],x[1],x[2]], axis=1))
    permute = Lambda(lambda x: K.permute_dimensions(x,(0,2,1)))
    comb = concer([cpf,npf,vtx])
    comb = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv0')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum,name='comb_batchnorm0')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout0')(comb)     
    comb = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv1')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum,name='comb_batchnorm1')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout1')(comb)     
    comb = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv2')(comb)
    if batchnorm:
        comb = BatchNormalization(momentum=batchmomentum,name='comb_batchnorm2')(comb)
    comb = Dropout(dropoutRate,name='comb_dropout2')(comb)     
    comb = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='comb_conv3')(comb)

    return comb


def block_deepFlavourConvolutionsFat(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout0')(cpf)                                                   
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout1')(cpf)                                                   
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf = Dropout(dropoutRate,name='cpf_dropout2')(cpf)                                                   
        cpf  = Convolution1D(50, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout0')(npf) 
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Dropout(dropoutRate,name='npf_dropout1')(npf)
        npf = Convolution1D(50, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout0')(vtx) 
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout1')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Dropout(dropoutRate,name='vtx_dropout2')(vtx)
        vtx = Convolution1D(50, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx


def block_deepFlavourConvolutionsFat_v3(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Convolution1D(75, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Convolution1D(75, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx


def block_deepFlavourConvolutionsFat_noVertex(charged,neutrals,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6 ):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf  = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='cpf_conv3')(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Convolution1D(75, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    return cpf,npf

def block_graph_vertex(seeds,neighbors,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    sds = seeds
    near = neighbors
 
    sds  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='sds_conv0')(sds)
    if batchnorm:
        sds = BatchNormalization(momentum=batchmomentum ,name='sds_batchnorm0')(sds)
    sds  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='sds_conv1')(sds)
    if batchnorm:
        sds = BatchNormalization(momentum=batchmomentum ,name='sds_batchnorm1')(sds)
  
    sds1 = Lambda( lambda x: tf.slice(x, [0,0,0], [-1,1,-1]))(sds)
    sds2 = Lambda( lambda x: tf.slice(x, [0,1,0], [-1,1,-1]))(sds)
    sds3 = Lambda( lambda x: tf.slice(x, [0,2,0], [-1,1,-1]))(sds)
    sds4 = Lambda( lambda x: tf.slice(x, [0,3,0], [-1,1,-1]))(sds)

        
    near  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='near_conv0')(near)
    if batchnorm:
        near = BatchNormalization(momentum=batchmomentum ,name='near_batchnorm0')(near)
    near  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='near_conv1')(near)
    if batchnorm:
        near = BatchNormalization(momentum=batchmomentum ,name='near_batchnorm1')(near)
    sds = keras.layers.Reshape((1,4,256))(sds)
    near = keras.layers.Reshape((20,4,256))(near)
   # near1 = Lambda( lambda x: tf.slice(x, [0,0,0], [-1,20,-1]))(near)
   # near2 = Lambda( lambda x: tf.slice(x, [0,20,0], [-1,20,-1]))(near)
   # near3 = Lambda( lambda x: tf.slice(x, [0,40,0], [-1,20,-1]))(near)
   # near4 = Lambda( lambda x: tf.slice(x, [0,60,0], [-1,20,-1]))(near)

    concer = Lambda(lambda x: K.concatenate([x[0],x[1]], axis=1))
    #large_concer = Lambda(lambda x: K.concatenate([x[0],x[1],x[2],x[3]], axis=1))
    #vtx1 = concer([sds1,near1])
    #vtx2 = concer([sds2,near2])
    #vtx3 = concer([sds3,near3])
    #vtx4 = concer([sds4,near4])
    #vtx = large_concer([vtx1,vtx2,vtx3,vtx4])
    vtx = concer([sds,near])    
    vtx = GravNet(10,256,75, 256)(vtx)
    vtx = GlobalAveragePooling1D()(vtx)

    #vtx  = Convolution1D(256, 21,strides=21, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx) 
    #vtx  = Convolution1D(75, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx) 
    """
    vtx1 = concer([sds1,near1])
    vtx1 = GravNet(10,256,75, 256)(vtx1)
    vtx1 = GlobalAveragePooling1D()(vtx1)

    vtx2 = concer([sds2,near2])
    vtx2  = GravNet(10,256,75, 256)(vtx2)
    vtx2 = GlobalAveragePooling1D()(vtx2)

    vtx3 = concer([sds3,near3])
    vtx3  = GravNet(10,256,75, 256)(vtx3)
    vtx3 = GlobalAveragePooling1D()(vtx3)

    vtx4 = concer([sds4,near4])
    vtx4  = GravNet(10,256,75, 256)(vtx4)
    vtx4 = GlobalAveragePooling1D()(vtx4)    

    vtx = large_concer([vtx1,vtx2,vtx3,vtx4])
    """
    return vtx


def block_deepFlavourConvolutionsFat_graph(charged,neutrals,vertices,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    '''
    deep Flavour convolution part. 
    '''
    cpf=charged
    if active:
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv0')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum ,name='cpf_batchnorm0')(cpf)
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv1')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm1')(cpf)
        cpf  = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='cpf_conv2')(cpf)
        if batchnorm:
            cpf = BatchNormalization(momentum=batchmomentum,name='cpf_batchnorm2')(cpf)
        cpf  = GravNet(6,256,100, 256)(cpf)
    else:
        cpf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(cpf)
        
    npf=neutrals
    if active:
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv0')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm0')(npf)
        npf = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='npf_conv1')(npf)
        if batchnorm:
            npf = BatchNormalization(momentum=batchmomentum,name='npf_batchnorm1')(npf)
        npf = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu' , name='npf_conv2')(npf)
    else:
        npf = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(npf)

    vtx = vertices
    if active:
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv0')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm0')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv1')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm1')(vtx)
        vtx = Convolution1D(256, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv2')(vtx)
        if batchnorm:
            vtx = BatchNormalization(momentum=batchmomentum,name='vtx_batchnorm2')(vtx)
        vtx = Convolution1D(100, 1, kernel_initializer='lecun_uniform',  activation='relu', name='vtx_conv3')(vtx)
    else:
        vtx = Convolution1D(1,1, kernel_initializer='zeros',trainable=False)(vtx)

    return cpf,npf,vtx


def block_deepFlavourDenseFat(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout')(x)
        x=  Dense(512, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

def block_deepFlavourDenseFat_v2(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
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
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

def block_deepFlavourDenseFat_v3(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(512, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x=  Dense(256, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x=  Dense(256, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate)(x)
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

def block_deepFlavourDenseFat_v4(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(256, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(512, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(256, activation='relu',kernel_initializer='lecun_uniform', name='df_dense2')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm2')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(256, activation='relu',kernel_initializer='lecun_uniform', name='df_dense3')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm3')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(256, activation='relu',kernel_initializer='lecun_uniform', name='df_dense4')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm4')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(256, activation='relu',kernel_initializer='lecun_uniform', name='df_dense5')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm5')(x)
        x = Dropout(dropoutRate)(x)
        x=  Dense(256, activation='relu',kernel_initializer='lecun_uniform', name='df_dense6')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm6')(x)
        x = Dropout(dropoutRate)(x)
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

def block_deepFlavourDenseFattest(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(512, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
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


def block_deepFlavourDenseSmall(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(200, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm0')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout0')(x)
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense1')(x)
        if batchnorm:
            x = BatchNormalization(momentum=batchmomentum,name='df_dense_batchnorm1')(x)
        x = Dropout(dropoutRate,name='df_dense_dropout1')(x)

    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

def block_deepFlavourDense_PermInvariant(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(100, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
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
        x = Dropout(dropoutRate,name='df_dense_dropout33')(x) 
    else:
        x= Dense(1,kernel_initializer='zeros',trainable=False,name='df_dense_off')(x)
    
    return x

def block_deepFlavourDense_PermInvariant_v2(x,dropoutRate,active=True,batchnorm=False,batchmomentum=0.6):
    if active:
        x=  Dense(400, activation='relu',kernel_initializer='lecun_uniform', name='df_dense0')(x)
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
