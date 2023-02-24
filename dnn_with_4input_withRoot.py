#Python/Data
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
import numpy as np
import h5py as h5
import sys
import uproot3 as ur
import awkward as ak
from data_functions import *
from matplotlib import style
sys.path.insert(0, './functions')
from training_functions import *

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


working_dir="/home/bishnu/EIC"
new_dir="DNN_output"
output_path=f"{working_dir}/{new_dir}"


# Define Hyper parameters
learning_rate = 1e-5
dropout_rate = 0.05
batch_size = 512
N_Epochs = 100
patience = 20
shuffle_split = False #Turn FALSE for images!
train_shuffle = False #False for better root hf comparison
Y_scalar = True
loss = 'mae'
#loss = tf.keras.losses.MeanAbsoluteError()

## READ THE FILES ROOT FILES IN THIS CASE
N_Events=200_000
sampling_fraction=0.02
#root_file = "/media/miguel/Elements/Data_hcali/Data1/log_uniform_pi+_20deg.root"
root_file = "/media/miguel/Elements/Data_hcali/Data1/Jan_2023_log_space_Files/log_uniform_pi+_17deg_jan26_23.root"
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

# Mean and std are calculated and plugged in her for now


log_hit=False
if log_hit:
    hit_e[hit_e==0]=0.000001
    hit_e=np.log10(hit_e)
    gen_energy=np.log10(gen_energy)
if log_hit==False:
    mean_hit= 0.0013525796176872475
    std_hit = 0.008820865515446804
    mean_target= 56.7935219292301  
    std_target=  80.59976326771323

else:
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

# Tensorflow CallBacks                                                                                        
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay,verbose=0)                                   
early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)                                          
history_logger=tf.keras.callbacks.CSVLogger(output_path+"/log.csv", separator=",", append=True)                      
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=output_path, save_best_only=True)                     
callbacks=[lr_scheduler, early_stopping,history_logger,batch_history(),model_checkpoint] 

normalize_hite=(hit_e-mean_hit)/std_hit
processed_hit = pad_sequences(normalize_hite, padding='post', dtype='float64')


normalize_X=(PosRecoX-mean_X)/std_X
processed_X = pad_sequences(normalize_X, padding='post', dtype='float64')



normalize_Y=(PosRecoX-mean_Y)/std_Y
processed_Y = pad_sequences(normalize_Y, padding='post', dtype='float64')

normalize_Z=(PosRecoZ-mean_Z)/std_Z
processed_Z = pad_sequences(normalize_Z, padding='post', dtype='float64')


processed_target=(gen_energy - mean_target)/std_target
processed_target=processed_target.reshape(-1,1)

## split DATA INTO TRAIN AND TEST
(X1_train, X1_val, X1_test,
 X2_train, X2_val, X2_test,
 X3_train, X3_val, X3_test,
 X4_train, X4_val, X4_test,
Y_train, Y_val, Y_test) = data_split(processed_hit, processed_X, processed_Y, processed_Z,\
                                     processed_target, val=0.2, test=0.3,shuffle=shuffle_split)




#dataset = dataset.shuffle(gen_P.size).batch(batch_size)
input_shape=processed_hit.shape[1]
print(input_shape)
# Define your neural network model
inputs1 = keras.layers.Input(shape=(input_shape,))  # your input shape for x1
inputs2 = keras.layers.Input(shape=(input_shape,)) # your input shape for x2
inputs3= keras.layers.Input(shape=(input_shape,))  # your input shape for x3
inputs4 = keras.layers.Input(shape=(input_shape,)) # your input shape for x4


# Define the layers for processing the first input array
x1 = layers.Dense(64, activation="relu")(inputs1)
x1 = layers.Dense(32, activation="relu")(x1)

# Define the layers for processing the second input array
x2 = layers.Dense(64, activation="relu")(inputs2)
x2 = layers.Dense(32, activation="relu")(x2)

# Define the layers for processing the first input array
x3 = layers.Dense(64, activation="relu")(inputs3)
x3 = layers.Dense(32, activation="relu")(x3)

# Define the layers for processing the second input array
x4 = layers.Dense(64, activation="relu")(inputs4)
x4 = layers.Dense(32, activation="relu")(x4)

concat = layers.concatenate([x1, x2, x3,x4])
output_layer = tf.keras.layers.Dense(1, activation='linear')(concat)




model = tf.keras.models.Model(inputs=[inputs1, inputs2,  inputs3, inputs4], outputs=output_layer)
learning_rate=1e-2
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mae')

history=model.fit(x=[X1_train, X2_train, X3_train, X4_train], y=Y_train, epochs=N_Epochs, batch_size=batch_size,\
callbacks=callbacks, validation_data=([X1_val, X2_val, X3_val, X4_val],Y_val))


mypreds = model.predict([X1_test, X2_test, X3_test,X4_test])

mypreds_final=mypreds*std_target + mean_target
Y_test_final=Y_test*std_target + mean_target
mypreds_final=mypreds_final[mypreds_final>0]
Y_test_final=Y_test_final[Y_test_final>0]
                                                                                                              
#pfn.layers                                                                                                    
#pfn.save("%s/energy_regression.h5"%(output_path))                                                                  
np.save("%s/predictions.npy"%(output_path),mypreds_final)
np.save("%s/test_data.npy"%(output_path),Y_test_final)
#FIXME: un-norm the predictions 

fig,axes=plt.subplots(figsize=(14,10))
axes.plot(history.history['loss'])
axes.plot(history.history['val_loss'])
axes.set_title('[HDF5] Model Loss vs. Epoch',fontsize=26)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'])
axes.set_ylim(0,0.5)
plt.savefig("%s/loss_epoch.png"%(output_path))
                            
mypreds_final=mypreds*std_target + mean_target
Y_test_final=Y_test*std_target + mean_target
mypreds_final=mypreds_final[mypreds_final>0]
Y_test_final=Y_test_final[Y_test_final>0]

#mypreds_final[mypreds_final<0]=256
#Y_test_final[Y_test_final<0]=256

plt.hist(mypreds_final,bins=100, range=(0, 200))
plt.hist(Y_test_final, bins=100, range=(0,200))
plt.legend(['Prediction', 'True'])

plt.savefig("%s/prediction_true.png"%(output_path))

def get_mean_std():
    
    mean_hit=np.nanmean(ak.flatten(hit_e))
    std_hit=np.nanstd(ak.flatten(hit_e))
    
    mean_X=np.nanmean(ak.flatten(PosRecoX))
    std_X=np.nanstd(ak.flatten(PosRecoX))
    
    mean_Y=np.nanmean(ak.flatten(PosRecoY))
    std_Y=np.nanstd(ak.flatten(PosRecoY))
    
    mean_Z=np.nanmean(ak.flatten(PosRecoZ))
    std_Z=np.nanstd(ak.flatten(PosRecoZ))
    
    mean_target=np.nanmean(gen_energy)
    std_target=np.nanstd(gen_energy)
    
    print ('mean_hit=', mean_hit, '   std_hit = ', std_hit)
    
    print ('mean_X=' , mean_X, '   std_X =', std_X)
    
    print ('mean_Y=', mean_Y, '   std_Y = ', std_Y)
    print ('mean_Z=', mean_Z, '   std_Z = ', std_Z)
    
    
    
    print ('mean_target=', mean_target, '   std_target =', std_target)
    
#get_mean_std()








