import tensorflow as tf
from energyflow.archs import PFN
from training_functions import *
from sklearn.preprocessing import StandardScaler
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
gpus = tf.config.experimental.list_physical_devices('GPU')
#print(gpus)
particle='pi+'
rate=5e-3
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

mirrored_strategy = tf.distribute.MirroredStrategy()
start_time=time.time()

if particle=='e-':
    h5_filename = '/media/miguel/Elements/Data_hcali/Data1/e-_623k_standalone_0_80GeV_train_split.hdf5'
    new_dir=f"standalone_model_623k_{particle}_1k_{rate}"
elif particle=='pi+':
    h5_filename = '/media/miguel/Elements/Data_hcali/Data1/pi+_495k_0_100GeV_train_split.hdf5'
    new_dir=f"aaastandalone_model_495k_{particle}_1k_{rate}"
else:
    print('WHAT IS PARTICLE')


#h5_filename = "/media/miguel/Elements/Data_hcali/Data1/total_train_split_1290k_pi+_gt50.hdf5"
#h5_filename = "/media/miguel/Elements/Data_hcali/Data1/total_train_split_1290k_pi+_lt50.hdf5"
#h5_filename = "/media/miguel/Elements/Data_hcali/Data1/total_train_split_300k_pi+.hdf5" 
#h5_filename = "/media/miguel/Elements/Data_hcali/Data1/total_train_split_658k_e-.hdf5"
#h5_filename ='/media/miguel/Elements/Data_hcali/Data1/e-_623k_standalone_0_80GeV_train_split.hdf5'
#h5_filename ='/media/miguel/Elements/Data_hcali/Data1/pi+_495k_0_100GeV_train_split.hdf5'
#h5_filename = "/media/miguel/Elements/Data_hcali/Data1/pi+_80k_standalone_train_split.hdf5"
h5_file = h5.File(h5_filename,'r')
working_dir=os.getcwd()
#new_dir=f"standalone_model_80k_pi+_1k_{rate}"
try:
    os.makedirs(f"{working_dir}/{new_dir}",exist_ok=True)
except OSError:
    print("Directory creation Error "%new_dir)
else:
    print("Directory %s is created"%new_dir)
label = new_dir  #Replace with your own variation!      
path = "./"+label


input_dim = h5_file['train_hcali'].shape[-2] #should be 4: Cell E,X,Y,Z, the number of features per particle
learning_rate =rate
dropout_rate = 0.05
batch_size = 1000
N_Epochs = 400
patience = 20
N_Latent = 128
shuffle_split = True #Turn FALSE for images!
train_shuffle = True #Turn TRUE for images!
Y_scalar = True
loss = 'mse' #'mae' and mse are two options #'swish'

Phi_sizes, F_sizes = (100, 100, N_Latent), (100, 100, 100)
output_act, output_dim = 'linear', 1 #Train to predict error

pfn = PFN(input_dim=input_dim, 
          Phi_sizes=Phi_sizes, 
          F_sizes=F_sizes, 
          output_act=output_act, 
          output_dim=output_dim, 
          loss=loss, 
          latent_dropout=dropout_rate,
          F_dropouts=dropout_rate,
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# Tensorflow CallBacks
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_decay,verbose=0)
early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience)
history_logger=tf.keras.callbacks.CSVLogger(path+"/log.csv", separator=",", append=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint( filepath=path, save_best_only=True)


train_generator = tf.data.Dataset.from_generator(
    training_generator(h5_filename,'train_hcali','train_mc',batch_size),
    output_shapes=(tf.TensorShape([None,None,None]),[None]),
    output_types=(tf.float64, tf.float64))


val_generator = tf.data.Dataset.from_generator(
    training_generator(h5_filename,'val_hcali','val_mc',batch_size),
    output_shapes=(tf.TensorShape([None,None,None]),[None]),
    output_types=(tf.float64, tf.float64))

test_generator = tf.data.Dataset.from_generator(
    test_generator(h5_filename,'test_hcali','test_mc',batch_size),
    output_shapes=(tf.TensorShape([None,None,None])),
    output_types=(tf.float64))

# training_generator.batch(batch_size)
# val_generator.batch(batch_size)
# test_generator.batch(batch_size)
the_fit = pfn.fit(
    train_generator,
    epochs=N_Epochs,
    batch_size=batch_size,
    callbacks=[lr_scheduler, early_stopping,history_logger,model_checkpoint],
    validation_data=val_generator,
    verbose=1
)

pfn.layers
pfn.save("%s/energy_regression.h5"%(path))
mypreds = pfn.predict(test_generator, batch_size=1000)
#mypreds=target_stdevs*mypreds_temp + target_means
np.save("%s/predictions.npy"%(path),mypreds)
#FIXME: un-norm the predictions

