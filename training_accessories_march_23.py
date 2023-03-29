import os
import sys
import shutil
import pickle
import subprocess

import uproot3 as ur
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers


from energyflow.archs import PFN
from energyflow.utils import data_split
shuffle_split = True

#print(tf.config.experimental.list_physical_devices('GPU'))

## THis functions returns the root file based on particle and detector you select
def get_root_file(particle, detector):
    if(particle=="pi+"):
        if detector=="hcal":
            root_file = "log_uniform_pi+_17deg_jan26_23.root"
        elif detector=="hcal_insert":
            root_file = "log_uniform_pi+_2.83deg_jan26_23.root"


    elif particle=="e-": 
        if detector=="hcal":
            root_file = "log_uniform_e-_17deg_jan26_23.root"
        elif detector=="hcal_insert": 
            root_file = "log_uniform_e-_2.83deg_jan26_23.root"

    else: print("CHECK IF YOU PROVIDED THE RIGHT PARTICLE")   
    return root_file

## THIS FUNCTION READS THE GIVEN ROOTFILE AND OUTPUTS
## HITS,e,x,y,z,TIME AND GENERATE ENERGY
def read_root_file(FileName,N_Events, detector):
    if detector=="hcal":
        detector_name = "HcalEndcapPHitsReco"

    elif detector=="hcal_insert":
        detector_name= "HcalEndcapPInsertHitsReco"
    else:
        print("Please make sure you have picked right detector name")     
        print("Pick: hcal or hcal_insert for endcap calo/ hcal_insert for insert")
    
    ur_file = ur.open(FileName)
    ur_tree = ur_file['events']
    print("You are Analyzing :", N_Events,"/", ur_tree.numentries)
                                                                                                           
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
    #root_gen_Theta
    root_gen_Theta = np.arccos(genPz/root_gen_P)*180/np.pi
    return hit_e, PosRecoX ,PosRecoY ,PosRecoZ , time, gen_energy  


def read_start_stop(file_path, detector, entry_start, entry_stop):
    ur_file = ur.open(file_path)
    ur_tree = ur_file['events']
    num_entries = ur_tree.numentries
    #num_entries=int(train_frac*num_entriesss)

    #print(means.shape,'      ',stds.shape)
    #print("PRINT  DETECTOR ", detector)    
    if detector=="hcal":
        detector_name = "HcalEndcapPHitsReco"

    elif detector=="hcal_insert":
        detector_name= "HcalEndcapPInsertHitsReco"
    else:
        print("Please make sure you have picked right detector name")     
        print("Pick: hcal or hcal_insert for endcap calo/ hcal_insert for insert")
            
    if(entry_stop<entry_start):
        return
        
    genPx = ur_tree.array('MCParticles.momentum.x',entrystart=entry_start, entrystop=entry_stop)[:,2]
    genPy = ur_tree.array('MCParticles.momentum.y',entrystart=entry_start, entrystop=entry_stop)[:,2]
    genPz = ur_tree.array('MCParticles.momentum.z',entrystart=entry_start, entrystop=entry_stop)[:,2]
    mass = ur_tree.array("MCParticles.mass", entrystart=entry_start      , entrystop=entry_stop)[:,2]
    root_gen_P = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
    gen_energy=np.sqrt(root_gen_P**2 + mass**2)

    hit_e =ur_tree.array(f'{detector_name}.energy',entrystart=entry_start, entrystop=entry_stop)
    
    return hit_e, gen_energy


### Get Mean of from selected events of input array
def get_mean_std_input(input_variables,Events_for_mean):
    means=[]
    stds=[]
    
    for arr in input_variables:
        mean = np.nanmean(ak.flatten(arr[:Events_for_mean]))
        std=   np.nanstd(ak.flatten(arr[:Events_for_mean]))
        means.append(mean)
        stds.append(std)
        
    return means, stds

### Get log 10 of hits
def get_log10_hitE(my_jagged_array):
    for i in range(len(my_jagged_array)):
        for j in range(len(my_jagged_array[i])):
            if my_jagged_array[i][j] >0:
                my_jagged_array[i][j] = np.log10(my_jagged_array[i][j])
                
            else:
                my_jagged_array[i][j] = np.nan
                
    return my_jagged_array        




### Draw QA Plots before and after the normalization
def get_mean_std_target(target_variable,Events_for_mean):
    mean_target = np.nanmean(target_variable[:Events_for_mean])
    std_target = np.nanstd(target_variable[:Events_for_mean])
        
    return mean_target, std_target

def QA_plots(raw_input,target_data,Events_for_mean,title, without_zero=True):
    cell_vars = ["Cell Energy","Cell Z","Cell X","Cell Y", "time"]
    nbins=20
    fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(14,7))
    axes = axes.ravel()
    density = True
    for i in range(len(cell_vars)):
        if without_zero==False:
            axes[i].hist(ak.flatten(raw_input[i][:Events_for_mean]),alpha=0.5,label=title,\
                      bins=nbins,density=density)

        elif without_zero==True:
            axes[i].hist(raw_input[i][:Events_for_mean][raw_input[i][:Events_for_mean]!=0],alpha=0.5,label=title, 
                         bins=nbins,density=density)
            
        else:
            print("Whether you want to plot zero in zero padded data or not")
        
        axes[i].legend(fontsize=15)
        axes[i].set_title("%s"%(cell_vars[i]),fontsize=20)
        axes[i].xaxis.set_tick_params(labelsize=15)
        axes[i].yaxis.set_tick_params(labelsize=15)
#axes[0].set_xlim(-0.5,0.5)
        axes[0].set_yscale('log')
        axes[4].set_yscale('log')
    axes[5].set_title("Gen Energy",fontsize=20)    
    axes[5].hist(target_data,label=title,bins=100)
    axes[5].xaxis.set_tick_params(labelsize=15)
    axes[5].yaxis.set_tick_params(labelsize=15)
    axes[5].legend(fontsize=15)
        
#plt.savefig(f"./QAplots_{title}.png")        
### Data preprocessing

def data_preprocessing(input_hit_info_arr, means, stds, gen_energy, mean_target, std_target, max_length):
    normalized_input_data=[(arr - mean)/std for arr,mean,std in zip(input_hit_info_arr,means,stds)]
    processed_input_data=[pad_sequences(input_data, maxlen=max_length, padding='post', dtype='float64') for input_data in 
                         normalized_input_data]
   
    processed_target_data=(gen_energy - mean_target)/std_target
    
    #processed_target_data=normalized_target.reshape(-1,1)
    
    return processed_input_data, processed_target_data

def get_mean_std(means,stds,mean_target,std_target, max_dim):
    return means,stds,mean_target,std_target, max_dim


def data_generator(file_path, detector, entry_begin, stop, means, stds, mean_target, std_target, max_dim, max_length,batch_size, chunk_size, log10):
    ur_file = ur.open(file_path)
    ur_tree = ur_file['events']
    num_entries = ur_tree.numentries
    #num_entries=int(train_frac*num_entriesss)
    detector=detector.decode('utf-8')

    #print(means.shape,'      ',stds.shape)
    #print("PRINT  DETECTOR ", detector)    
    if detector=="hcal":
        detector_name = "HcalEndcapPHitsReco"

    elif detector=="hcal_insert":
        detector_name= "HcalEndcapPInsertHitsReco"
    else:
        print("Please make sure you have picked right detector name")     
        print("Pick: hcal or hcal_insert for endcap calo/ hcal_insert for insert")
            
     
    
    for entry_start in range(entry_begin,stop, chunk_size):
        entry_stop=entry_start + chunk_size
        if entry_stop>stop:
            break
        
        genPx = ur_tree.array('MCParticles.momentum.x',entrystart=entry_start, entrystop=entry_stop)[:,2]
        genPy = ur_tree.array('MCParticles.momentum.y',entrystart=entry_start, entrystop=entry_stop)[:,2]
        genPz = ur_tree.array('MCParticles.momentum.z',entrystart=entry_start, entrystop=entry_stop)[:,2]
        mass = ur_tree.array("MCParticles.mass", entrystart=entry_start      , entrystop=entry_stop)[:,2]
        root_gen_P = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        gen_energy=np.sqrt(root_gen_P**2 + mass**2)

        hit_e =ur_tree.array(f'{detector_name}.energy',entrystart=entry_start, entrystop=entry_stop)
        PosRecoX = ur_tree.array(f'{detector_name}.position.x',entrystart=entry_start, entrystop=entry_stop )
        PosRecoY = ur_tree.array(f'{detector_name}.position.y',entrystart=entry_start, entrystop=entry_stop )
        PosRecoZ= ur_tree.array(f'{detector_name}.position.z',entrystart=entry_start, entrystop=entry_stop )
        time= ur_tree.array(f'{detector_name}.time',entrystart=entry_start, entrystop=entry_stop)
        #root_gen_Theta
        #root_gen_Theta = np.arccos(genPz/root_gen_P)*180/np.pi
        if(log10==True):
            hit_e[hit_e<0.000005]=0.000005
            hit_e=get_log10_hitE(hit_e)
            gen_energy=np.log10(gen_energy)
        
 
        input_hit_info=[hit_e, PosRecoZ ,PosRecoX ,PosRecoY ,time]
        
        means_h,stds_h,mean_target_h,std_target_h, features=get_mean_std(means,stds,mean_target,std_target, max_dim)
        processed_input_data, processed_target_data=data_preprocessing(input_hit_info, means_h, stds_h,gen_energy, \
                                                                             mean_target_h, std_target_h, max_length)
        
        
        
        processed_input_data_arr=np.array(processed_input_data)
        
        #input_data_temp = np.concatenate([processed_input_data_arr[0:features,:, :, np.newaxis]], axis=2)
             
        #print(" max dim ", max_dim)
        #input_data=np.reshape(input_data_temp, (input_data_temp.shape[1],input_data_temp.shape[2],features))
        
        input_data_temp=reshape_for_dnn(processed_input_data_arr)
        input_data=input_data_temp[:,:,:features]
        yield input_data, processed_target_data
        
 
#### TAKE TRANSPOSE OF THE INPUT ARRAY AND MAKE SUITABLE TO FEED INTO TRAINING
def reshape_for_dnn(processed_input_data):
    processed_input_data_arr=np.array(processed_input_data)
    #print(processed_input_data_arr.shape)
    #input_data = np.concatenate([processed_input_data_arr[0:max_dim,:, :, np.newaxis]], axis=2)
    input_data=np.transpose(processed_input_data_arr,(1,2,0))
    return input_data

def write_Y_test(chunk_size_write, Y_test, output_path):
    num_chunks=int(np.ceil(len(Y_test)/chunk_size_write))
    for i in range (num_chunks):
        start_ch=i*chunk_size_write
        end_ch=min((i+1)*chunk_size_write, len(Y_test))
        chunk=Y_test[start_ch:end_ch]
        np.save("%s/Y_test_%d_%d.npy"%(output_path,i,num_chunks),chunk)
        
def read_Y_test(output_path, max_file_num):  
    print("I am here ---")
    Y_test_chunks = []
    num_chunks = 0
    while True:
        try:
            chunk = np.load(f"{output_path}/Y_test_{num_chunks}_{max_file_num}.npy")
            Y_test_chunks.append(chunk)
            num_chunks += 1
        except FileNotFoundError:
            break
    Y_test = np.concatenate(Y_test_chunks, axis=0)


    Y_test_final=np.ravel(Y_test)   
    return Y_test_final

