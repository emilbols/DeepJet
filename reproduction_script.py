import imp
imp.find_module('setGPU')
import setGPU
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint
from keras.models import Input, Model
from keras.layers import Dense
from sklearn.utils import shuffle

def mod_crossentropy(tensor_input):
    def mod_crossentropy_nest(y_true,y_pred):
        y_true = K.print_tensor(y_true, message="\n y_true is: ")
        x = K.print_tensor(tensor_input, message="\n x is: ")
        loss = K.categorical_crossentropy(y_true,y_pred) + K.sum(x) - K.sum(x)
        return loss
    return mod_crossentropy_nest


class newline_callbacks_begin(Callback): 
    def on_train_begin(self,logs={}):                                                                                             
        self.outputDir='/afs/cern.ch/work/e/ebols/private/DeepJet/DeepJet/' 
    
    def on_train_batch_end(self,batch,batch_logs={}):
        import os 
        blossfile=os.path.join( self.outputDir, 'batch_losses.log')  
        f = open(blossfile, 'a')  
        f.write(str(batch_logs.get('loss')))
        f.write(" ")
        f.write(str(batch_logs.get('val_loss')))   
        f.write("\n") 
        f.close()  





def generator(data):
    # dummy generator that only works for batch sizes which are a multiple of the dataset
    psamples=0 #for random shuffling
    nepoch=0
    shufflecounter=0
    shufflecounter2=0
    begin = 0
    batch_size = 10000
    totalbatches = (data['x'].shape[0])/batch_size
    end = begin+batch_size
    processedbatches=0
    x = 0
    y = 0
    while 1:
        if processedbatches == totalbatches:
            processedbatches=0
            begin=0
            end=begin+batch_size
            nepoch+=1
            data['x'] = shuffle(data['x'],random_state=psamples)
            data['y'] = shuffle(data['y'],random_state=psamples)
            psamples=psamples+1
            

        x = data['x'][begin:end]
        y = data['y'][begin:end]
        begin = begin + batch_size
        end = end + batch_size
        processedbatches=processedbatches+1
        print('yielded batch = ' + str(processedbatches) +'\n')
        print('x_yield: \n')
        print(x[0])
        yield (x,y)


x_train = np.random.rand(2000000,2)
y_train = np.around(x_train[:,0]*x_train[:,1]+0.3)
y_train = keras.utils.to_categorical(y_train)
data_train= {'x':x_train , 'y':y_train}

x_val = np.random.rand(2000000,2)
y_val = np.around(x_val[:,0]*x_val[:,1]+0.3)
y_val = keras.utils.to_categorical(y_val)
data_val ={'x':x_val , 'y':y_val}
print np.sum(y_val,axis=0)

inp = Input(shape=(2,))
x = Dense(100, activation='relu')(inp)
x = Dense(100, activation='relu')(x)
x = Dense(100, activation='relu')(x)
output = Dense(2, activation='softmax')(x)
model = Model(inputs=inp,outputs=output)
print(model.summary())

batch_callback = newline_callbacks_begin()
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(loss=mod_crossentropy(inp), optimizer=adam, metrics=['accuracy'])
without_gen = False

if without_gen:
    model.fit(x=data_train['x'],y=data_train['y'],validation_data=(data_val['x'],data_val['y']),batch_size=10000,epochs=10,callbacks=[batch_callback])
else:
    model.fit_generator(generator(data_train) ,
                            steps_per_epoch=2000000/10000, 
                            epochs=10,
                            callbacks=[batch_callback],
                            validation_data=generator(data_val),
                            validation_steps=2000000/10000,
                            max_q_size=100)


