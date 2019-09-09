


from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL, loss_meansquared
from DeepJetCore.modeltools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_deepFlavourReference_GraphShallow
    
    train.setModel(model_deepFlavourReference_GraphShallow,dropoutRate=0.1,momentum=0.3)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001, clipnorm = 1.0,
                       loss=['categorical_crossentropy',loss_meansquared],
                       metrics=['accuracy'],
                       loss_weights=[1., 0.000000000001])


    train.train_data.maxFilesOpen=5
    
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=1, 
                                     batchsize=5000, 
                                     stop_patience=300, 
                                     lr_factor=0.2, 
                                     lr_patience=3, 
                                     lr_epsilon=0.00001, 
                                     lr_cooldown=1, 
                                     lr_minimum=0.000001, 
                                     maxqsize=5,verbose=1,checkperiod=3)
    
    
    print('fixing input norms...')
    train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
train.compileModel(learningrate=0.0003, clipnorm = 1.0,
                   loss=['categorical_crossentropy',loss_meansquared],
                   metrics=['accuracy'],
                   loss_weights=[1., 0.000000000001])
    
print(train.keras_model.summary())
#printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=250, #sweet spot from looking at the testing plots 
                                 batchsize=5000, 
                                 stop_patience=300, 
                                 lr_factor=0.2, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=1, 
                                 lr_minimum=0.0000001, 
                                 maxqsize=5,verbose=1,checkperiod=1)
