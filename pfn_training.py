import tensorflow as tf
from energyflow.archs import PFN
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

import yaml
#Using YAML configuration File                                                                                                              
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
#'-----------------DONT FORGET TO CHANGE HERE---------------------------------------'
looking_for=config["looking_for"]
sub_label=config["sub_label"]
num_input=config["num_input"]

#"pi+_hcali" ## pi+_hcal, pi+_hcal, or e-_hcal
particle=looking_for.split('_')[0]
detector=looking_for.split('_')[1]


#'________CHANGE HERE IF NECESSARY______________'
label=f"log_loss_{detector}_{particle}_{sub_label}"  #Replace with your own variation!

print(looking_for,'    ',particle,'detector   ',detector,'  label', label)

'''
if(num_input=='4'):
    pick_training_function="/home/bishnu/EIC/regressiononly/functions/use_4_input_in_training.sh"
    input_dimension=4

elif(num_input=='5'):
    pick_training_function="/home/bishnu/EIC/regressiononly/functions/use_5_input_in_training.sh"
    input_dimension=5
subprocess.run(pick_training_function, shell=True)
print(input_dimension,'ccccccccccccccc')    
'''    
#h5_filename = config["h5_filename"]
#print("TRYING TO OPEN ",h5_filename)

Attributes={
    'e-_hcal':  {'File_Name':'log_uniform_e-_17deg_jan26_23_full.hdf5',
                 'PFN_Model_dir':'log_loss_hcal_e-_noSF',
                 'PFN_train_file':'log_uniform_e-_17deg_jan26_23_train_split.hdf5',
                 'PFN_train_file_SF':'log_loss_noFile'},
    
    'pi+_hcal': {'File_Name':'log_uniform_pi+_17deg_jan26_23_full.hdf5',
                 'PFN_Model_dir':'log_loss_hcal_pi+_noSF',
                 'PFN_train_file':'log_uniform_pi+_17deg_jan26_23_train_split.hdf5',
                 'PFN_train_file_SF': 'log_loss_hcal_pi+_SF'},
                
    'e-_hcali': {'File_Name':'log_uniform_e-_2.83deg_jan26_23_full.hdf5',
                 'PFN_Model_dir': 'log_loss_hcali_e-_noSF',
                 'PFN_train_file':'log_uniform_e-_2.83deg_jan26_23_train_split.hdf5',
                 'PFN_train_file_SF':'log_loss_noFile'},
        
    'pi+_hcali':{'File_Name':'log_uniform_pi+_2.83deg_jan26_23_full.hdf5',
                 'PFN_Model_dir':'log_loss_hcali_pi+_noSF',
                 'PFN_train_file':'log_uniform_pi+_2.83deg_jan26_23_train_split.hdf5',
                 'PFN_train_file_SF':'log_loss_hcali_SF'}
                 
          }
PFN_train_file=Attributes[looking_for]['PFN_train_file'] ## train split hdf5 file 
file_path="/media/miguel/Elements/Data_hcali/Data1/Jan_2023_log_space_Files"
h5_filename=f"{file_path}/{PFN_train_file}"
#h5_filename = config["h5_filename"]                                                                                                        
#print("TRYING TO OPEN ",h5_filename)

# h5_filename = "../generate_data/to_hdf5/Uniform_pi+_0-100GeV_standalone_TVT_Split.hdf5"
h5_file = h5.File(h5_filename,'r')

print('----------------',detector,'   ',particle)

do_normalization = True
#input_dim = h5_file[f'train_{detector}'].shape[-2] #should be 4: Cell E,X,Y,Z, the number of features per particle
input_dim=4    #input_dimension
learning_rate = 1e-4
dropout_rate = 0.1
batch_size = 1_000
N_Epochs = 5
patience = 10
N_Latent = 128
shuffle_split = True #Turn FALSE for images!
train_shuffle = True #Turn TRUE for images!
loss = 'mae' #'mae'
print(input_dim)

output_dir = "/home/bishnu/EIC/pfn_output/"

path=output_dir+label
#path = label
shutil.rmtree(path, ignore_errors=True)
os.makedirs(path)
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
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path, save_best_only=True)
callbacks=[lr_scheduler, early_stopping,history_logger,batch_history(),model_checkpoint]


train_generator = tf.data.Dataset.from_generator(
    training_generator(h5_filename,f'train_{detector}','train_mc',batch_size,do_normalization,path),
    output_shapes=(tf.TensorShape([None,None,None]),[None]),
    output_types=(tf.float64, tf.float64))

# train_generator = tf.data.Dataset.from_generator(
#     training_generator(h5_filename,'train_hcal','train_mc',batch_size,do_normalization,path),
#     tf.TensorSpec(shape=((None,None,None),(None)), dtype=tf.float64),
#     output_types=(tf.float64, tf.float64))

val_generator = tf.data.Dataset.from_generator(
    training_generator(h5_filename,f'val_{detector}','val_mc',batch_size,do_normalization,path),
    output_shapes=(tf.TensorShape([None,None,None]),[None]),
    output_types=(tf.float64, tf.float64))

test_generator = tf.data.Dataset.from_generator(
    test_generator(h5_filename,f'test_{detector}','test_mc',batch_size,do_normalization,path),
    output_shapes=(tf.TensorShape([None,None,None])),
    output_types=(tf.float64))


N_QA_Batches = 5 #number of batches for just plotting input data
pre_training_QA(h5_filename,path,N_QA_Batches,batch_size,do_normalization)

history = pfn.fit(
    train_generator,
    epochs=N_Epochs,
    batch_size=batch_size,
    callbacks=callbacks,
    validation_data=val_generator,
    verbose=1
)


#save batch loss
np.save("%s/batch_loss.npy"%(path),batch_history.batch_loss)

#save epoch loss
with open(path+'/history_file', 'wb') as hist_file:
    pickle.dump(history.history, hist_file)

pfn.layers
pfn.save("%s/energy_regression.h5"%(path))
mypreds = pfn.predict(test_generator, batch_size=1000)
np.save("%s/predictions.npy"%(path),mypreds)
#FIXME: un-norm the predictions



'''
import re
#print(h5_filename)
Last_name=h5_filename.split('/')[-1] 
print(Last_name)
angle_value=float(re.search(r'\d+', Last_name).group())
print(angle_value)                                                       
if angle_value>10:  
    detector='hcal' 
elif angle_value<10:  
    detector='hcali'
print(''__________________',detector)  
'''
