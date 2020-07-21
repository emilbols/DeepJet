
#import sys
#import tensorflow as tf
#sys.modules["keras"] = tf.keras

from DeepJetCore.training.training_base import training_base
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights
from Losses import *

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_deepFlavourReference_reg
    
    train.setModel(model_deepFlavourReference_reg,dropoutRate=0.1,momentum=0.3)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)

    train.keras_model.load_weights('/data/ml/ebols/DJ_puppi_2018_RestartedModel_v1/KERAS_check_model_block_1_epoch_05.h5')
    
    train.compileModel(learningrate=0.0003,
                       loss=['categorical_crossentropy',huber_loss,asym_loss_down,asym_loss_up],
                       loss_weights=[1.0,0.1,1.0,1.0],
                       metrics=['accuracy'])


    train.train_data.maxFilesOpen=1
    
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=1, 
                                     batchsize=10000, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=--1, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.0001)
    
    
    #print('fixing input norms...')
    #train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')

    
train.compileModel(learningrate=0.0003,
                   loss=['categorical_crossentropy',huber_loss,asym_loss_down,asym_loss_up],
                   loss_weights=[1.0,0.1,1.0,1.0],
                   metrics=['accuracy'])
    
print(train.keras_model.summary())
#printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=65, #sweet spot from looking at the testing plots 
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=-1, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=10, 
                                 lr_minimum=0.00001,
                                 verbose=1,checkperiod=1)
