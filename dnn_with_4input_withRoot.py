#Python/Data
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
import numpy as np
import h5py as h5
import uproot3 as ur
import awkward as ak
from data_functions import *
from matplotlib import style
import os
import shutil

#ML
import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split
import tensorflow as tf

import mplhep as hep
hep.set_style(hep.style.CMS)
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

shuffle_split = True
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
import sys
sys.path.insert(0, './functions')
from training_functions import * 

N_Events=200_000
log_hit=False
particle='pi+'
new_dir="dnn_flatten_end_dprate_0.05_lr7_EXYZ_64_64_64_64_pi+_2-40deg"
path="/home/bishnu/EIC/output_reg_dnn_straw/"
output_path=f"{path}{new_dir}"
ifExists=os.path.exists(output_path)
if ifExists==True:
    shutil.rmtree(output_path)
    os.mkdir(output_path)
if ifExists==False:
    os.mkdir(output_path)
learning_rate = 1e-7
dropout_rate = 0.05
batch_size = 1024
N_Epochs = 300
patience = 10
N_Latent = 128
shuffle_split = True #Turn FALSE for images!
train_shuffle = True #False for better root hf comparison
loss = 'mae'



#root_file = "/media/miguel/Elements/Data_hcali/Data1/log_uniform_pi+_20deg.root"
#root_file = "/media/miguel/Elements/Data_hcali/Data1/Jan_2023_log_space_Files/log_uniform_pi+_17deg_jan26_23.root"

### ELECTRON 17 DEG HCAL
#root_file = "/media/miguel/Elements/Data_hcali/Data1/Jan_2023_log_space_Files/log_uniform_e-_17deg_jan26_23.root"

## pion 2 - 40 deg
root_file = "/media/miguel/Elements/Data_hcali/Data1/Jan_2023_log_space_Files/log_uniform_pi+_2_40deg.root"
#root_file = "/media/miguel/Elements/Data_hcali/Data1/Jan_2023_log_space_Files/PFN_train_file"
detector_name = "HcalEndcapPHitsReco" #or "HcalEndcapPInsertHitsReco"

ur_file = ur.open( root_file )
ur_tree = ur_file['events']

#cut_primary = array["MCParticles.generatorStatus"]==1                                              
genPx = ur_tree.array('MCParticles.momentum.x',entrystop=N_Events)[:,2]
genPy = ur_tree.array('MCParticles.momentum.y',entrystop=N_Events)[:,2]
genPz = ur_tree.array('MCParticles.momentum.z',entrystop=N_Events)[:,2]
mass = ur_tree.array("MCParticles.mass", entrystop=N_Events)[:,2]
root_gen_P = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
gen_energy=np.sqrt(root_gen_P**2 + mass**2)

hit_e =ur_tree.array(f'{detector_name}.energy' ,entrystop=N_Events)
PosRecoX = ur_tree.array(f'{detector_name}.position.x',entrystop=N_Events)
PosRecoY = ur_tree.array(f'{detector_name}.position.y',entrystop=N_Events)
PosRecoZ= ur_tree.array(f'{detector_name}.position.z',entrystop=N_Events)
time= ur_tree.array(f'{detector_name}.time',entrystop=N_Events)

max_length = max(len(seq) for seq in hit_e)
if (particle=='pi+'):
    if log_hit==False:
        mean_hit= 0.0013525796176872475
        std_hit = 0.008820865515446804
        mean_target= 56.7935219292301  
        std_target=  80.59976326771323

    elif log_hit==True:
        mean_hit= -3.996334547054441
        std_hit =  0.8841258843383055
        mean_target= 1.2015416612519736
        std_target = 0.7683784305726994
    
    mean_X= -110.71318062140894
    std_X = 986.3043191820791
    mean_Y= -40.869369728254796
    std_Y = 1010.5846773983045
    mean_Z= 4407.6375918933545
    std_Z = 322.61474120326426

elif particle=='e-':
    mean_hit= 0.007688988742008094
    std_hit =  0.027424001946040586
    mean_X= -96.62919517378967
    std_X = 886.0139020995832
    mean_Y= 18.433972498143497
    std_Y =  876.033085912942
    mean_Z= 4043.5278760392043
    std_Z =  145.46183555397687
    mean_target= 32.17839962794849
    std_target = 41.85756951453055

print('mean _hit  ',mean_hit,'  mean_X ',mean_X,'    mean_target  ',mean_target)    
# Tensorflow CallBacks                                                                                        
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay,verbose=0)                                   
early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)                                          
history_logger=tf.keras.callbacks.CSVLogger(output_path+"/log.csv", separator=",", append=True)                      
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=output_path, save_best_only=True)                     
#callbacks=[lr_scheduler, early_stopping,history_logger,batch_history(),model_checkpoint]
callbacks=[history_logger]


#### Normalize the input

normalize_hite=(hit_e-mean_hit)/std_hit
processed_hit = pad_sequences(normalize_hite, maxlen=max_length, padding='post', dtype='float64')


normalize_X=(PosRecoX-mean_X)/std_X
processed_X = pad_sequences(normalize_X, maxlen=max_length,padding='post', dtype='float64')

normalize_Y=(PosRecoY-mean_Y)/std_Y
processed_Y = pad_sequences(normalize_Y, maxlen=max_length, padding='post', dtype='float64')

normalize_Z=(PosRecoZ-mean_Z)/std_Z
processed_Z = pad_sequences(normalize_Z, maxlen=max_length, padding='post', dtype='float64')


processed_target=(gen_energy - mean_target)/std_target

#processed_target=processed_target.reshape(-1,1)
# Concatenate the input data along the feature axis
input_data = np.concatenate([processed_hit[:, :, np.newaxis], 
                              processed_X[:, :, np.newaxis], 
                              processed_Y[:, :, np.newaxis], 
                              processed_Z[:, :, np.newaxis]], 
                             axis=2)
## NO NEED TO CHANGE ANYTHING BELOW IF YOU CHOOSE TO CHANGE THE NUMBER OF INPUT IN MODEL E, X, Y OR Z
(X_train, X_val, X_test,
Y_train, Y_val, Y_test) = data_split(input_data, processed_target, val=0.2, test=0.3,shuffle=shuffle_split)

num_pixels=input_data.shape[-2]
num_pixels_2=input_data.shape[-1]
# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_pixels, num_pixels_2)),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dropout(dropout_rate),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(units=128, activation='relu'),
    #tf.keras.layers.Dense(units=128, activation='relu'),
    #tf.keras.layers.Dense(units=64, activation='relu'),
    #tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

# Compile the model
model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate))


csv_logger = tf.keras.callbacks.CSVLogger(output_path +'/log.csv',separator=",")

# Train the model
history=model.fit(x=X_train, y=Y_train, epochs=N_Epochs, batch_size=batch_size,validation_data=(X_val, Y_val), callbacks=[csv_logger])

fig,axes=plt.subplots(figsize=(14,10))
axes.plot(history.history['loss'])
axes.plot(history.history['val_loss'])
axes.set_title('[HDF5] Model Loss vs. Epoch',fontsize=26)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
axes.set_ylim(0,0.5)

mypreds = model.predict(X_test,batch_size=batch_size)

mypreds_final=mypreds*std_target + mean_target
Y_test_final=Y_test*std_target + mean_target
mypreds_final=mypreds_final[mypreds_final>0]
Y_test_final=Y_test_final[Y_test_final>0]



#R_model.save("%s/R_energy_regression.h5"%(output_path))
#R_preds = R_model.predict(R_X_test,batch_size=400)
np.save("%s/R_predictions.npy"%(output_path),mypreds_final)
#np.save("%s/R_y_test.npy"%(output_path),R_Y_test)
np.save("%s/Y_test.npy"%(output_path),Y_test_final)




#mypreds_final[mypreds_final<0]=256
#Y_test_final[Y_test_final<0]=256

plt.hist(mypreds_final,bins=100, range=(0, 200))
plt.hist(Y_test_final, bins=100, range=(0,200))
plt.legend(['Prediction', 'True'])
