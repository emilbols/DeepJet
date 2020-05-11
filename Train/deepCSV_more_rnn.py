

from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared, mod_crossentropy_nest_v2
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_DeepCSV_more_RNN
    
    train.setModel(model_DeepCSV_more_RNN,dropoutRate=0.1,momentum=0.3)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.003,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])


    train.train_data.maxFilesOpen=4
    
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=1, 
                                     batchsize=10000, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=3, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.00000000000001, 
                                     maxqsize=1)
    
    
    print('fixing input norms...')
    #train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
    train.compileModel(learningrate=0.003,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    
print(train.keras_model.summary())
printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=100, #sweet spot from looking at the testing plots 
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience= 2, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=3, 
                                 lr_minimum=0.0000000000000001, 
                                 maxqsize=1,verbose=1,checkperiod=3)
