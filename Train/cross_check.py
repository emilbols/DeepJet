

from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared, mod_crossentropy
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_crosscheck
    
    Inputs = train.setModel_returnInput(model_crosscheck)
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=[mod_crossentropy(Inputs)],
                       metrics=['accuracy'])


    train.train_data.maxFilesOpen=30
    
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=1, 
                                     batchsize=10000, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=3, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.0001, 
                                     maxqsize=5)
    
    
    #print('fixing input norms...')
    #train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
    train.compileModel(learningrate=0.0003,
                           loss=[mod_crossentropy(Inputs)],
                           metrics=['accuracy'])
    
print(train.keras_model.summary())
printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=300, #sweet spot from looking at the testing plots 
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.8, 
                                 lr_patience=15, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=8, 
                                 lr_minimum=0.00001, 
                                 maxqsize=5,verbose=1,checkperiod=3)
