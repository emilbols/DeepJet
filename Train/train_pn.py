

from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared, mod_crossentropy_nest_v2
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    print('new training')
    from models import get_particle_net_model
    
    train.setModel(get_particle_net_model,dropoutRate=0.1,momentum=0.3)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy',loss_meansquared],
                       metrics=['accuracy'],
                       loss_weights=[1., 0.000000000001])


    train.train_data.maxFilesOpen=30
    
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=10, 
                                     batchsize=1024, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=3, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.0001, 
                                     maxqsize=1,checkperiod=3,save_weights=True)
    
    
train.compileModel(learningrate=0.0001,
                   loss=['categorical_crossentropy',loss_meansquared],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.000000000001])
    
print(train.keras_model.summary())
printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=10, #sweet spot from looking at the testing plots 
                                 batchsize=1024, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience= -3, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=5, 
                                 lr_minimum=0.000001, 
                                 maxqsize=1,verbose=1,checkperiod=3,save_weights=True)
