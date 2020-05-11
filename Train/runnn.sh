predict.py /data/ml/ebols/TrainingSet0p15Step8/KERAS_model.h5 /data/ml/ebols/TestTTBar/dataCollection.dc /data/ml/ebols/Prediction0p15Step8 &> predict0p15Step8.log &
wait $!
predict.py /data/ml/ebols/TrainingSet0p15Step9/KERAS_model.h5 /data/ml/ebols/TestTTBar/dataCollection.dc /data/ml/ebols/Prediction0p15Step9 &> predict0p15Step9.log &
wait $!
predict.py /data/ml/ebols/TrainingSet0p15Step10/KERAS_model.h5 /data/ml/ebols/TestTTBar/dataCollection.dc /data/ml/ebols/Prediction0p15Step10 &> predict0p15Step10.log &
