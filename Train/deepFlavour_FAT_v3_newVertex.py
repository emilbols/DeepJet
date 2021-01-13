

from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_deepFlavourReference_fat_v3_vertex
    
    train.setModel(model_deepFlavourReference_fat_v3_vertex,dropoutRate=0.1,momentum=0.3)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy',loss_meansquared],
                       metrics=['accuracy'],
                       loss_weights=[1., 0.000000000001])


    train.train_data.maxFilesOpen=5
    
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=1, 
                                     batchsize=2000, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=3, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.0001, 
                                     maxqsize=1)
    
    
    print('fixing input norms...')
    train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
train.compileModel(learningrate=0.0001,
                   loss=['categorical_crossentropy',loss_meansquared],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.000000000001])
    
print(train.keras_model.summary())
#printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=63, #sweet spot from looking at the testing plots 
                                 batchsize=2000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=4, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.00001, 
                                 maxqsize=1,verbose=1,checkperiod=3)
