import tensorflow as tf
from tensorflow import keras
from energyflow.archs import PFN
from energyflow.utils import data_split
import h5py as h5
import numpy as np
import os
import shutil
import pickle
import subprocess
import sys
#print(tf.config.experimental.list_physical_devices('GPU'))
sys.path.insert(0, './functions')
from training_functions import *
shuffle_split = True

file_path="/media/miguel/Elements/Data_hcali/Data1/Jan_2023_log_space_Files"
PFN_train_file='log_uniform_pi+_17deg_jan26_23_full.hdf5' #log_uniform_pi+_17deg_jan26_23_train_split.hdf5'                                                         
h5_filename=f"{file_path}/{PFN_train_file}"
print(h5_filename)

h5_file = h5.File(h5_filename,'r')
print(list(h5_file.keys()))
N_Events=50_000
gen_P = h5_file['mc'][:N_Events,8,0]
#h5_gen_Theta = h5_file['mc'][:N_Events,9,0]        
hits_e = h5_file[detector][:N_Events,0]
posX = h5_file[detector][:N_Events,1]
posY = h5_file[detector][:N_Events,2]
posZ = h5_file[detector][:N_Events,3]

arrays = np.stack((hits_e, posX, posY, posZ), axis=-1)

# Calculate the mean along the last axis
mean = np.nanmean(arrays,axis=(0,1))
std=   np.nanstd(arrays,axis=(0,1))
mean_target=np.nanmean(gen_P)
std_target=np.nanstd(gen_P)


hits_e=np.nan_to_num(hits_e)
posX=np.nan_to_num(posX)
posY=np.nan_to_num(posY)
posZ=np.nan_to_num(posZ)

normalize_output=(gen_P-mean_target)/std_target
normalize_hit=(hits_e - mean[0]) / std[0]
normalize_posX= (posX - mean[1])/std[1]
normalize_posY= (posY - mean[2])/std[2]
normalize_posZ= (posZ - mean[3])/std[3]

print(mean)

(X1_train, X1_val, X1_test,
 X2_train, X2_val, X2_test,
 X3_train, X13val, X3_test,
 X4_train, X3_val, X4_test,
Y_train, Y_val, Y_test) = data_split(normalize_hit, normalize_posX, normalize_posY, normalize_posZ,normalize_output, val=0.2, test=0.3,shuffle=shuffle_split)



# Create a TensorFlow Dataset from your input arrays
#dataset = tf.data.Dataset.from_tensor_slices((hits_e, posX, posY, posZ,gen_P))

# Shuffle and batch the dataset
batch_size = 32
#dataset = dataset.shuffle(gen_P.size).batch(batch_size)
input_shape=hits_e.shape[1]
# Define your neural network model
inputs1 = keras.layers.Input(shape=(input_shape,))  # your input shape for x1
inputs2 = keras.layers.Input(shape=(input_shape,)) # your input shape for x2
inputs3= keras.layers.Input(shape=(input_shape,))  # your input shape for x1
inputs4 = keras.layers.Input(shape=(input_shape,)) # your input shape for x2

hidden1 = tf.keras.layers.Dense(64, activation='relu')(inputs1)
hidden2 = tf.keras.layers.Dense(64, activation='relu')(inputs2)
hidden3 = tf.keras.layers.Dense(64, activation='relu')(inputs3)
#hidden2 = tf.keras.layers.Dense(64, activation='relu')(input4)

x = tf.keras.layers.concatenate([hidden1, hidden2, hidden3, inputs4])

output_layer = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.models.Model(inputs=[inputs1, inputs2,  inputs3, inputs4], outputs=output_layer)


do_normalization=True
input_dim=4   #input_dimension                                                                                                                                      
learning_rate = 5e-3
dropout_rate = 0.1
batch_size = 128
N_Epochs = 1
patience = 10
N_Latent = 128
shuffle_split = True #Turn FALSE for images!                   
train_shuffle = True #Turn TRUE for images!     
loss = 'mae' #'mae'                                                                                                                    
num_global_features=1

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
print(input_dim)
label="test_dnn"
output_dir = "/home/bishnu/EIC/pfn_output/"
path=output_dir+label


path=output_dir+label
#path = label                                                                        
shutil.rmtree(path, ignore_errors=True)
os.makedirs(path)
Phi_sizes, F_sizes = (100, 100, N_Latent), (100, 100, 100)
output_act, output_dim = 'linear', 1 #Train to predict error 

# Tensorflow CallBacks                                                                                                                                              
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay,verbose=0)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)
history_logger=tf.keras.callbacks.CSVLogger(path+"/log.csv", separator=",", append=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path, save_best_only=True)
callbacks=[lr_scheduler, early_stopping]#,history_logger,batch_history(),model_checkpoint]



model.compile(optimizer=optimizer, loss='mse')
model_fit=model.fit(x=[X1_train, X2_train, X3_train, X4_train], y=Y_train,epochs=N_Epochs)

mypreds = model.predict([X1_test, X2_test, X3_test,X4_test])
np.save("%s/h5_y_test.npy"%(path),h5_Y_test)

mypreds_true=mypreds*std_target + mean_target
Y_test_true= Y_test*std_target + mean_target
plt.scatter(Y_test, mypreds)
