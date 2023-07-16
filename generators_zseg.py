#Imports
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import uproot as ur
import awkward as ak
import time
import sys
from multiprocessing import Process, Queue, Manager, set_start_method
# from multiprocess import Process, Queue, Manager, set_start_method
import compress_pickle as pickle
from scipy.stats import circmean
sys.path.insert(0, '/home/bishnu/EIC/regressiononly/functions')
from Clusterer import *
import random
MIP=0.0006 ## GeV
MIP_ECAL=0.13
time_TH=150  ## ns
energy_TH=0.5*MIP
energy_TH_ECAL=0.5*MIP_ECAL
NHITS_MIN=2
z_min=3820
z_max=5086
import uproot as ur2
#Change these for your usecase!
# data_dir = '/clusterfs/ml4hep_nvme2/ftoralesacosta/regressiononly/data/'
# out_dir = '/clusterfs/ml4hep_nvme2/ftoralesacosta/regressiononly/preprocessed_data/'

#data_dir = '/usr/workspace/hip/eic/log10_Uniform_03-23/log10_pi+_Uniform_0-140Gev_17deg_1/'
#out_dir = '/usr/WS2/karande1/eic/gitrepos/regressiononly/preprocessed_data/'



class MPGraphDataGenerator:
    def __init__(self,
                 file_list: list,
                 batch_size: int,
                 shuffle: bool = True,
                 num_procs: int = 32,
                 calc_stats: bool = False,
                 preprocess: bool = False,
                 already_preprocessed: bool = False,
                 is_val: bool = False,
                 data_set: str =None,
                 output_dir: str = None,
                 num_features: int = 4,
                 output_dim: int =1,
                 hadronic_detector: str =None,
                 include_ecal: bool = True,
                 n_Z_layers: int =10):
        """Initialization"""

        self.preprocess = preprocess
        self.already_preprocessed = already_preprocessed
        self.calc_stats = calc_stats
        self.is_val = is_val
        self.data_set=data_set
        self.hadronic_detector=hadronic_detector
        self.include_ecal=include_ecal
        self.output_dir = output_dir
        self.stats_dir = os.path.realpath(self.output_dir)
        # self.stats_dir = os.path.realpath(self.output_dir+'../')
        self.val_stat_dir = os.path.dirname(self.stats_dir)
        self.file_list = file_list
        self.num_files = len(self.file_list)
        self.output_dim=output_dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_Z_layers=n_Z_layers
        self.num_procs = num_procs
        self.procs = []


        if(self.hadronic_detector=='hcal'):
            self.detector_name = "HcalEndcapPHitsReco"
            self.sampling_fraction =0.0224
        elif(self.hadronic_detector=='hcal_insert'):    #'Insert' after the 'P'
            self.detector_name = "HcalEndcapPInsertHitsReco"
            self.sampling_fraction =0.0089
            
        
        self.nodeFeatureNames = [".energy",".position.z", ".position.x",".position.y",]
        self.nodeFeatureNames_ecal =['ecal_energy','ecal_posz', 'ecal_posx', 'ecal_posy']
        self.detector_ecal='EcalEndcapPHitsReco'
        self.num_nodeFeatures = num_features

        # Slice the nodeFeatureNames list to only include the first 'num_features' elements
        ## SET UP FOR ONE/TWO DIMENSION OUTPUT AND WITH/WITHOUT ECAL
        self.nodeFeatureNames = self.nodeFeatureNames[:num_features]
        self.nodeFeatureNames_ecal = self.nodeFeatureNames_ecal[:num_features]

                
        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.num_targetFeatures = output_dim   #Regression on Energy only (output dim =1)  Energy + theta for output_dim=2
        
        if ((self.num_targetFeatures==3) & (not self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + ["clusterE","genP","theta", "phi"]
            
        elif ((self.num_targetFeatures==3) & (self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + self.nodeFeatureNames_ecal+["clusterE","genP","theta","phi"]
            
        elif ((self.num_targetFeatures==1) & (not self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + ["clusterE","genP"]
            
        elif ((self.num_targetFeatures==1) & (self.include_ecal)):
            self.scalar_keys = self.nodeFeatureNames + self.nodeFeatureNames_ecal+ ["clusterE","genP"]    
            
            
        ## FOR VALIDATION MEAN AND STD FROM TRAINING SET IS USED
        if self.data_set!='val':
            # if not self.is_val and self.calc_stats:
            if (self.calc_stats):
                n_scalar_files = 8 #num files to use for scaler calculation
                if(not self.include_ecal):
                    self.preprocess_scalar(n_scalar_files)    ### 1st potential place

                elif (self.include_ecal):
                    self.preprocess_scalar_with_ecal(n_scalar_files)    ### 1st potential place    
            else:
                self.means_dict = pickle.load(open(f"{self.stats_dir}/means.p", 'rb'), compression='gzip')
                self.stdvs_dict = pickle.load(open(f"{self.stats_dir}/stdvs.p", 'rb'), compression='gzip')
                
        elif self.data_set=='val':
            self.means_dict = pickle.load(open(f"{self.val_stat_dir}/train/means.p", 'rb'), compression='gzip')
            self.stdvs_dict = pickle.load(open(f"{self.val_stat_dir}/train/stdvs.p", 'rb'), compression='gzip')
            
        
        if self.already_preprocessed and os.path.isdir(self.output_dir):
            self.file_list = [self.output_dir + f'data_{i:03d}.p' for i in range(self.num_files)]
        elif self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()
        else:
            print('Check preprocessing config!!')



        if self.shuffle: np.random.shuffle(self.file_list)





    def preprocess_scalar(self,n_calcs):
        print(f'\nCalcing Scalars and saving data to {self.stats_dir}')
        self.n_calcs = min(n_calcs,self.num_files)

        with Manager() as manager:
            means = manager.list()
            stdvs = manager.list()
            for i in range(self.n_calcs):
                p = Process(target=self.scalar_processor, args=(i,means,stdvs), daemon=True)
                p.start()
                self.procs.append(p)

            for p in self.procs:
                p.join()

            means = np.mean(means,axis=0) #avg means along file dimension
            stdvs = np.mean(stdvs,axis=0) #avg stdvs from files
            #stdvs[stdvs == 0] = 1
            self.means_dict = dict(zip(self.scalar_keys,means))
            self.stdvs_dict = dict(zip(self.scalar_keys,stdvs))
            print("MEANS = ",self.means_dict)
            print("STDVS = ",self.stdvs_dict)
            print(f"saving calc files to {self.stats_dir}/means.p\n")

            pickle.dump(self.means_dict, open(
                        self.stats_dir + '/means.p', 'wb'), compression='gzip')

            pickle.dump(self.stdvs_dict, open(
                        self.stats_dir + '/stdvs.p', 'wb'), compression='gzip')

        print(f"Finished Mean and Standard Deviation Calculation using { n_calcs } Files")
        
    def preprocess_scalar_with_ecal(self,n_calcs):
        print(f'\nCalcing Scalars and saving data to {self.stats_dir}')
        self.n_calcs = min(n_calcs,self.num_files)
        
        with Manager() as manager:
            means = manager.list()
            stdvs = manager.list()
            for i in range(self.n_calcs):
                p = Process(target=self.scalar_processor_with_ecal, args=(i,means,stdvs), daemon=True)
                p.start()
                self.procs.append(p)
                
            for p in self.procs:
                p.join()

            means = np.mean(means,axis=0) #avg means along file 
            stdvs = np.mean(stdvs,axis=0) #avg stdvs from files

            self.means_dict = dict(zip(self.scalar_keys,means))
            self.stdvs_dict = dict(zip(self.scalar_keys,stdvs))
            print("MEANS = ",self.means_dict)
            print("STDVS = ",self.stdvs_dict)
            print(f"saving calc files to {self.stats_dir}/means.p\n")
            
            pickle.dump(self.means_dict, open(
                        self.stats_dir + '/means.p', 'wb'), compression='gzip')

            pickle.dump(self.stdvs_dict, open(
                        self.stats_dir + '/stdvs.p', 'wb'), compression='gzip')

        print(f"Finished Mean and Standard Deviation Calculation using { n_calcs } Files")

        
    def scalar_processor(self,worker_id,means,stdvs):

        file_num = worker_id

        while file_num < self.num_files:
            print(f"Mean + Stdev Calc. file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur2.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward
            #print('xxxxxxxxxxxxxxxxxxx ',num_events)
            file_means = []
            file_stdvs = []
            cell_data=[]
            new_array=[]
            cell_E = event_data[self.detector_name+".energy"]
            time=event_data[self.detector_name+".time"]
            mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10) 


            for feature_name in self.nodeFeatureNames:
                feature_data = event_data[self.detector_name+feature_name][mask]
                #if "energy" in feature_name:
                #    feature_data = np.log10(feature_data)
                max_length = max(len(sublist) for sublist in feature_data)
                #print('            ', max_length)
                padded_array = np.zeros((len(feature_data), max_length))
                for i, sublist in enumerate(feature_data):
                    padded_array[i, :len(sublist)] = sublist

                cell_data.append(padded_array)    
                    
            cell_data=np.array(cell_data)
            #print('xxxxxxxxxxxxxxxxx ',cell_data.shape)
            cell_data_swaped=np.swapaxes(cell_data,0,1)
            #print('AAAAAAAAAAAAAAAAA ', cell_data_swaped.shape)
            ## Arrange as E, Z, X, Y for all events
            for row in cell_data_swaped:
                column=np.column_stack((row[0], row[1], row[2], row[3]))
                new_array.append(column)
            new_array=np.array(new_array, dtype=object)
            
            ## Get Z segmentation regrouuping
            z_seg_array=[]
            cluster_sum_arr=[]
            for row in new_array:
                
                #if row.shape[0]<1:
                #    continue
                if np.all(row==0):
                    continue
                new_array=self.get_regrouped_zseg_unique_xy(self.n_Z_layers, row)
                
                z_seg_array.append(new_array)
                
    
            z_seg_array=np.array(z_seg_array, dtype=object)
            
            for i in range(len(self.nodeFeatureNames)):
                means_temp=[]
                stds_temp=[]
                for row in z_seg_array:
                    if i<1:
                        sum=np.sum(row[:,0])
                        cluster_sum_arr.append(sum)
                    means_temp.append(np.mean(row[:,i])) 
                    stds_temp.append(np.mean(row[:,i])) 
                file_means.append(np.mean(means_temp))
                file_stdvs.append(np.std(means_temp))
   
            #print('8888888888888888888 ', file_means)  

            cluster_sum_arr=np.array(cluster_sum_arr)
            #file_means.append(np.mean(cluster_sum_arr))
            #file_stdvs.append(np.std(cluster_sum_arr))
                #if "energy" in feature_name:
                #    feature_data = np.log10(feature_data)
                    
               
            #unfortunatley, there's a version error so we can't use ak.nanmean...
            #cluster_sum_E = ak.sum(cell_E[mask],axis=-1) #global node feature later
                        
            mask = cluster_sum_arr > 0.0
            #cluster_calib_E  =np.log10(cluster_sum_arr[mask]/self.sampling_fraction)
            cluster_calib_E  =cluster_sum_arr[mask]/self.sampling_fraction
            
            #np.log10(cluster_sum_E[mask] / self.sampling_fraction)
            
                        
            file_means.append(np.mean(cluster_calib_E))
            file_stdvs.append(np.std(cluster_calib_E))
            
            genPx = event_data['MCParticles.momentum.x'][:,2]
            genPy = event_data['MCParticles.momentum.y'][:,2]
            genPz = event_data['MCParticles.momentum.z'][:,2]
            #genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
            genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
            #generation has the parent particle at index 2

            file_means.append(np.mean(genP))
            file_stdvs.append(np.std(genP))
            if self.num_targetFeatures==3:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*180/np.pi
                gen_phi=(np.arctan2(genPy,genPx))*180/np.pi
                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

                file_means.append(ak.mean(gen_phi))  ####
                file_stdvs.append(ak.std(gen_phi))   ####
            
            means.append(file_means)
            stdvs.append(file_stdvs)

            file_num += self.num_procs


    def scalar_processor_with_ecal(self,worker_id,means,stdvs):

        file_num = worker_id

        while file_num < self.num_files:
            print(f"Mean + Stdev Calc. file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur2.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward

            file_means = []
            file_stdvs = []

            cell_E = event_data[self.detector_name+".energy"]
            time=event_data[self.detector_name+".time"]
            mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)
            
            cell_E_ecal = event_data[self.detector_ecal+".energy"]
            time_ecal   = event_data[self.detector_ecal+".time"]
            mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<time_TH) & (cell_E_ecal<1e10) 
            
            
            for feature_name in self.nodeFeatureNames:
                feature_data = event_data[self.detector_name+feature_name][mask]
                                
                #if "energy" in feature_name:
                #    feature_data = np.log10(feature_data)
                
                                                
                file_means.append(ak.mean(feature_data))
                file_stdvs.append(ak.std(feature_data))
                
            ## ECAL MEANS AND STD AFTER HCAL     
            for feature_name in self.nodeFeatureNames:
                feature_data_ecal = event_data[self.detector_ecal+feature_name][mask_ecal]
                #if "energy" in feature_name:
                #    feature_data_ecal = np.log10(feature_data_ecal)
            ### ECAL    
                file_means.append(ak.mean(feature_data_ecal))
                file_stdvs.append(ak.std(feature_data_ecal))
                
                #unfortunatley, there's a version error so we can't use ak.nanmean...
            
            cluster_sum_E_hcal = ak.sum(cell_E[mask],axis=-1) #global node feature later
            cluster_sum_E_ecal=ak.sum(cell_E_ecal[mask_ecal],axis=-1)

            cluster_calib_E_hcal = cluster_sum_E_hcal / self.sampling_fraction
            cluster_calib_E_ecal  = cluster_sum_E_ecal ## sampling fractionn crrrection is already done

            total_calib_E= cluster_calib_E_hcal + cluster_calib_E_ecal
            mask = total_calib_E > 0.0
            #cluster_calib_E=np.log10(total_calib_E[mask])
            cluster_calib_E=total_calib_E[mask]
            file_means.append(np.mean(cluster_calib_E))
            file_stdvs.append(np.std(cluster_calib_E))
            

            genPx = event_data['MCParticles.momentum.x'][:,2]
            genPy = event_data['MCParticles.momentum.y'][:,2]
            genPz = event_data['MCParticles.momentum.z'][:,2]
            #genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
            genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
            #generation has the parent particle at index 2

            file_means.append(ak.mean(genP))
            file_stdvs.append(ak.std(genP))
            if self.num_targetFeatures==3:
                mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
                theta=np.arccos(genPz/mom)*180/np.pi
                gen_phi=(np.arctan2(genPy,genPx))*180/np.pi
                file_means.append(ak.mean(theta))  ####
                file_stdvs.append(ak.std(theta))   ####

                file_means.append(ak.mean(gen_phi))  ####
                file_stdvs.append(ak.std(gen_phi))   ####
            means.append(file_means)
            stdvs.append(file_stdvs)

            file_num += self.num_procs
            


    def get_regrouped_zseg_unique_xy(self, n_Z_layers, data):
        z_layers = get_Z_segmentation([z_min,z_max],n_Z_layers)
        z_centers = (z_layers[:-1] + z_layers[1:]) / 2
        z_mask =  get_Z_masks(data[:,1],z_layers) #mask for binning
        #print(z_layers)
        #print(z_centers)
        new_array = []
        # Iterate over the bins of column 1
        for i in range(len(z_layers) - 1):
            # Filter the data for the current bin of column 1
            bin_data = data[(data[:, 1] >= z_layers[i]) & (data[:, 1] < z_layers[i + 1])]
            #print(bin_data)
            # Calculate the sum of column 1 for unique combinations of column 2 and column 3
            unique_sum = {}
            for row in bin_data:
                #print(row[0])
                key = (row[2], row[3])
                if key not in unique_sum:
                    unique_sum[key] = 0
                unique_sum[key] += row[0]

            # Append the center value of the bin of column 1 and the sum values to the new array
            center_value = (z_layers[i] + z_layers[i + 1]) / 2
            for key, value in unique_sum.items():
                kera_array=np.column_stack((value, center_value, key[0], key[1]))
                new_array.append(kera_array)
        new_array=np.array(new_array)
        if new_array.shape[0]<1:
            
            print('Segmentation   ' , new_array.shape, 'data shape  ', data.shape)
            print(data)
                     
        new_array=np.swapaxes(new_array,1,2)
        #print(new_array.shape)
        new_array=np.reshape(new_array, (new_array.shape[0], new_array.shape[1]))
        #print(new_array.shape)
        return new_array

            
    def preprocess_data(self):
        print(f'\nPreprocessing and saving data to {os.path.realpath(self.output_dir)}')

        for i in range(self.num_procs):
            p = Process(target=self.preprocessor, args=(i,), daemon=True)
            p.start()
            self.procs.append(p)
        
        for p in self.procs:
            p.join()

        self.file_list = [self.output_dir + f'data_{i:03d}.p' for i in range(self.num_files)]


    def preprocessor(self, worker_id):

        file_num = worker_id

        while file_num < self.num_files:
            print(f"Processing file number {file_num}")
            f_name = self.file_list[file_num]

            event_tree = ur2.open(f_name)['events']
            num_events = event_tree.num_entries
            event_data = event_tree.arrays() #need to use awkward

            preprocessed_data = []

            for event_ind in range(num_events):
                
                nodess, global_node, cluster_num_nodes = self.get_nodes(event_data, event_ind)
                senders, receivers, edges = self.get_edges(cluster_num_nodes) #returns 'None'
                #print('I am shape     nodess  ', nodess.shape)
                #print(event_ind, ' event   ', nodess)
                #if (nodess.shape[0]<1):
                #    continue
                if np.all(nodess==0):
                    continue
                node=self.get_regrouped_zseg_unique_xy(self.n_Z_layers, nodess)
                nodes=self.scalar_preprocessor_zseg(node)
                
                #nhits_hcal = np.sum(nodes[:, -1] == 1)
                #print(nhits_hcal, '     events    ', event_ind)
                #if (not global_node) or (nhits_hcal<NHITS_MIN):
                    
                #    continue

                graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                    'senders': senders, 'receivers': receivers, 'edges': edges} 

                # graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                #     'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                #     'edges': edges.astype(np.float32)}
                if self.num_targetFeatures==3:
                    target = self.get_GenP_Theta(event_data,event_ind)    
                else:
                    target = self.get_GenP(event_data,event_ind)
                    
                meta_data = [f_name]
                meta_data.extend(self.get_meta(event_data, event_ind))

                preprocessed_data.append((graph, target, meta_data)) 

            random.shuffle(preprocessed_data) #should be done BEFORE multiple 'images' per geant event

            pickle.dump(preprocessed_data, open(self.output_dir + f'data_{file_num:03d}.p', 'wb'), compression='gzip')

            print(f"Finished processing file number {file_num}")
            file_num += self.num_procs



    def get_nodes(self,event_data,event_ind):
        if(not self.include_ecal):
            nodes = self.get_cell_data(event_data[event_ind])
            global_node = self.get_cluster_calib(event_data[event_ind])
        if(self.include_ecal):
            nodes = self.get_cell_data_with_ecal(event_data[event_ind])
            global_node = self.get_cluster_calib_with_ecal(event_data[event_ind])
            
        cluster_num_nodes = len(nodes)
        return nodes, np.array([global_node]), cluster_num_nodes
    
    def get_cell_data(self,event_data):

        cell_data = []
        cell_data_ecal = []

        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)
        
        
        for feature in self.nodeFeatureNames:
            feature_data = event_data[self.detector_name+feature][mask]
            
            #if "energy" in feature:  
            #    feature_data = np.log10(feature_data)
                
            #standard scalar transform
            #feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
            cell_data.append(feature_data)

        cell_data_swaped=np.swapaxes(cell_data,0,1)
        return cell_data_swaped
        #return np.swapaxes(cell_data,0,1) # returns [Events, Features]
        #alternative: cell_data = np.reshape(cell_data, (len(self.nodeFeatureNames), -1)).T
    def scalar_preprocessor_zseg(self, event_data):
        #print('hello hello ', event_data.shape)
        for index,feature in enumerate(self.nodeFeatureNames):
            if self.stdvs_dict['.position.z']==0:  ## This is to adrress the case where the std is 0, for instance with z=1
                event_data[:,index]= event_data[:,index]
            else:    
                event_data[:,index]= (event_data[:,index] - self.means_dict[feature]) / self.stdvs_dict[feature]
            
        return event_data    

    ### WITH ECAL AND HCAL 
    def get_cell_data_with_ecal(self,event_data):

        cell_data = []
        cell_data_ecal = []

        cell_E = event_data[self.detector_name+".energy"]
        time=event_data[self.detector_name+".time"]
        mask = (cell_E > energy_TH) & (time<time_TH) & (cell_E<1e10)
        

        cell_E_ecal = event_data[self.detector_ecal+".energy"]
        time_ecal=event_data[self.detector_ecal+".time"]
        mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<time_TH) & (cell_E_ecal<1e10)
        #mask_ecal = (cell_E_ecal > energy_TH_ECAL) & (time_ecal<time_TH) & (cell_E_ecalå<1e10)

        for feature in self.nodeFeatureNames:

            feature_data = event_data[self.detector_name+feature][mask]
            feature_data_ecal = event_data[self.detector_ecal+feature][mask_ecal]
            #if "energy" in feature:
            #    feature_data = np.log10(feature_data)
            #    feature_data_ecal = np.log10(feature_data_ecal)
            #standard scalar transform
            feature_data = (feature_data - self.means_dict[feature]) / self.stdvs_dict[feature]
            #print('Mean hcal ll ', self.means_dict[feature])
            cell_data.append(feature_data)

            
        for feature_ecal in self.nodeFeatureNames_ecal:            
            feature_data_ecal = (feature_data_ecal - self.means_dict[feature_ecal]) / self.stdvs_dict[feature_ecal]
            #print('Mean ECA:::::: ll ', self.means_dict[feature_ecal])
            cell_data_ecal.append(feature_data_ecal)

        cell_data_swaped=np.swapaxes(cell_data,0,1)

        cell_data_ecal_swaped=np.swapaxes(cell_data_ecal,0,1)
        #cell_data_total=np.vstack((cell_data_swaped,cell_data_ecal_swaped)) 
        col_with_zero_ecal=np.zeros((cell_data_ecal_swaped.shape[0],1))
        cell_data_ecal_label=np.hstack((cell_data_ecal_swaped, col_with_zero_ecal))

        col_with_one_hcal=np.ones((cell_data_swaped.shape[0],1))
        cell_data_hcal_label=np.hstack((cell_data_swaped, col_with_one_hcal))

        cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))

        

        return cell_data_total

    def get_cluster_calib(self, event_data):
        nodes=self.get_cell_data(event_data)
        if np.all(nodes==0):
            pass
        else:
            node=self.get_regrouped_zseg_unique_xy(self.n_Z_layers, nodes)
            cluster_sum_E=np.sum(node[:,0])
        
            """ Calibrate Clusters Energy """
        
            #cell_E = event_data[self.detector_name+".energy"]
            #cluster_sum_E = np.sum(cell_E,axis=-1) #global node feature later
            if cluster_sum_E <= 0:
                return None
            #cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))
        
            #cluster_calib_E  = np.log10(cluster_sum_E/self.sampling_fraction)
            cluster_calib_E  =cluster_sum_E/self.sampling_fraction
            cluster_calib_E = (cluster_calib_E - self.means_dict["clusterE"])/self.stdvs_dict["clusterE"]
        
            return(cluster_calib_E)
        


    ## WITH ECAL AND HCAL 
    def get_cluster_calib_with_ecal(self, event_data):
        """ Calibrate Clusters Energy """

        cell_E = event_data[self.detector_name+".energy"]
        cell_E_ecal = event_data[self.detector_ecal+".energy"]
        
        cluster_sum_E_hcal = np.sum(cell_E,axis=-1) #global node feature later
        cluster_sum_E_ecal = np.sum(cell_E_ecal,axis=-1) #global node feature later
        '''
        if cluster_sum_E_hcal <= 0:
            return None    
            
        if cluster_sum_E_ecal<=0:
            return None
        '''
        cluster_calib_E_hcal  = cluster_sum_E_hcal/self.sampling_fraction
        cluster_calib_E_ecal  = cluster_sum_E_ecal
        
        #cell_data_total=np.vstack((cell_data_hcal_label, cell_data_ecal_label))
        
        cluster_calib_E= cluster_calib_E_hcal + cluster_calib_E_ecal
        if cluster_calib_E<=0:
            return None
        #cluster_calib_E=np.log10(cluster_calib_E)
        cluster_calib_E=cluster_calib_E
        
        cluster_calib_E = (cluster_calib_E - self.means_dict["clusterE"])/self.stdvs_dict["clusterE"]

        return(cluster_calib_E)

    
    def get_edges(self, num_nodes):
        return None,None,None

    def get_GenP(self,event_data,event_ind):

        genPx = event_data['MCParticles.momentum.x'][event_ind,2]
        genPy = event_data['MCParticles.momentum.y'][event_ind,2]
        genPz = event_data['MCParticles.momentum.z'][event_ind,2]
        #the generation has the parent praticle always at index 2

        #genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
        genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]
        return genP

    def get_GenP_Theta(self,event_data,event_ind):

        genPx = event_data['MCParticles.momentum.x'][event_ind,2]
        genPy = event_data['MCParticles.momentum.y'][event_ind,2]
        genPz = event_data['MCParticles.momentum.z'][event_ind,2]
        mom=np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        theta=np.arccos(genPz/mom)*180/np.pi
        gen_phi=(np.arctan2(genPy,genPx))*180/np.pi
        #the generation has the parent praticle always at index 2
        
        #genP = np.log10(np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz))
        genP = np.sqrt(genPx*genPx + genPy*genPy + genPz*genPz)
        genP = (genP - self.means_dict["genP"]) / self.stdvs_dict["genP"]
        theta = (theta - self.means_dict["theta"]) / self.stdvs_dict["theta"]
        gen_phi = (gen_phi - self.means_dict["phi"]) / self.stdvs_dict["phi"]
        return genP, theta, gen_phi

    #FIXME: DELETE THIS AND TARGET SCALARS
    def get_cell_scalars(self,event_data):

        means = []
        stdvs = []

        for feature in nodeFeatureNames:
            means.append(np.nanmean(event_data[feature]))
            stdvs.append(np.nanstd( event_data[feature]))

        return means, stdvs


    def get_target_scalars(self,target):

        return np.nanmean(target), np.nanstd(target)


    def get_meta(self, event_data, event_ind):
        """ 
        Reading meta data
        Returns senders, receivers, and edges    
        """ 
        #For Now, only holds event id. Only one cluster per event, and no eta/phi
        meta_data = [] 
        meta_data.append(event_ind)

        return meta_data
    def preprocessed_worker(self, worker_id, batch_queue):
        batch_graphs = []
        batch_targets = []
        batch_meta = []

        file_num = worker_id
        while file_num < self.num_files:
            file_data = pickle.load(open(self.file_list[file_num], 'rb'), compression='gzip')

            #print("FILE DATA SHAPE = ",np.shape(file_data))

            for i in range(len(file_data)):
                batch_graphs.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
                batch_meta.append(file_data[i][2])
                #print('generator.py   shape of file    ',np.shape(file_data[i][1]))
                # batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
                '''need the above line if there are more than 1 cluster per event'''

                if len(batch_graphs) == self.batch_size:

                    batch_queue.put((batch_graphs, batch_targets, batch_meta))

                    batch_graphs = []
                    batch_targets = []
                    batch_meta = []

            file_num += self.num_procs

        if len(batch_graphs) > 0:
            # batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)

            batch_queue.put((batch_graphs, batch_targets, batch_meta))

    def worker(self, worker_id, batch_queue):
        if self.preprocess:
            self.preprocessed_worker(worker_id, batch_queue)
        else:
            raise Exception('Preprocessing is required for regression models.')

    def check_procs(self):
        for p in self.procs:
            if p.is_alive(): return True

        return False

    def kill_procs(self):
        for p in self.procs:
            p.kill()

        self.procs = []


    def generator(self):
        batch_queue = Queue(2 * self.num_procs)

        for i in range(self.num_procs):
            p = Process(target=self.worker, args=(i, batch_queue), daemon=True)
            p.start()
            self.procs.append(p)

        while self.check_procs() or not batch_queue.empty():
            try:
                batch = batch_queue.get(True, 0.0001)
            except:
                continue

            #FIXME: Print Batches here too
            yield batch

        for p in self.procs:
            p.join()


    
if __name__ == '__main__':
    pion_files = np.sort(glob.glob(data_dir+'*.root')) #dirs L14
    pion_files = pion_files[:20]
    # print("Pion Files = ",pion_files)

    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                    batch_size=32,
                                    shuffle=False,
                                    num_procs=32,
                                    preprocess=True,
                                    already_preprocessed=True,
                                    output_dir=out_dir,
                                    num_features=num_features,
                                    output_dim=output_dim,
                                    hadronic_detector=hadronic_detector,
                                    include_ecal= True)

    gen = data_gen.generator()

    print("\n~ DONE ~\n")
    exit()