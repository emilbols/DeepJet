

from DeepJetCore.training.training_base import training_base
from Losses import loss_NLL

import time
print time.strftime("%Y-%m-%d %H:%M")

#also does all the parsing
train=training_base(testrun=False)
print 'Inited'

if not train.modelSet():
    from models import dense_model
    print 'Setting model'
    train.setModel(dense_model,dropoutRate=0.1)
    
    train.compileModel(learningrate=0.0035,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])


model,history = train.trainModel(nepochs=400, 
                                 batchsize=12500, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=15, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.000001, 
                                 maxqsize=100)

print time.strftime("%Y-%m-%d %H:%M")
