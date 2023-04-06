import numpy as np
import os
import sys
import glob
import uproot as ur
import matplotlib.pyplot as plt
import time
import seaborn as sns
import tensorflow as tf
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
import sonnet as snt
import argparse
import yaml

from generators import MPGraphDataGenerator
import block as models
sns.set_context('poster')


### READ MODEL FROM THE YAML FILE 
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
args = parser.parse_args()
config = yaml.safe_load(open(args.config))

data_config = config['data']
model_config = config['model']
train_config = config['training']

#data_dir = data_config['data_dir']
num_test_files =10
#num_val_files = data_config['num_val_files']
batch_size = data_config['batch_size']
shuffle = data_config['shuffle']
num_procs = 2
preprocess = data_config['preprocess']
output_dir ='/home/bishnu/EIC/deepsets_output'
already_preprocessed = False
calc_stats = False 

concat_input = model_config['concat_input']





## PATH TO THE DATA AND MODEL DIRECTORY
path_to_result='/home/bishnu/EIC/regressiononly/results_all/Block_20230331_1758_concatTrue'
data_dir="/media/miguel/Elements/Data_hcali/Data1/log10_Uniform_03-23/log10_pi+_discrete_17deg_20_20k/"

if preprocess:
        train_output_dir = output_dir + '/train/'
        val_output_dir = output_dir + '/test/'


### MODEL READ 
model = models.BlockModel(global_output_size=1, model_config=model_config)
checkpoint = tf.train.Checkpoint(module=model)

last_ckpt_path = path_to_result + '/last_saved_model'



if os.path.exists(last_ckpt_path+'.index'):
        checkpoint.read(last_ckpt_path)
#print(type(checkpoint))




root_files = np.sort(glob.glob(data_dir+'*root'))
test_start = 0
test_end = test_start + num_test_files
#val_end = train_end + num_val_files
root_test_files = root_files[test_start:test_end]



def get_batch(data_iter):
        for graphs, targets, meta in data_iter:
            graphs = convert_to_tuple(graphs)
            targets = tf.convert_to_tensor(targets, dtype=tf.float32)
            
            yield graphs, targets


def convert_to_tuple(graphs):
        nodes = []
        edges = []
        globals = []
        senders = []
        receivers = []
        n_node = []
        n_edge = []
        offset = 0

        for graph in graphs:
            nodes.append(graph['nodes'])
            globals.append([graph['globals']])
            n_node.append(graph['nodes'].shape[:1])
            
            if graph['senders']:
                senders.append(graph['senders'] + offset)
            if graph['receivers']:
                receivers.append(graph['receivers'] + offset)
            if graph['edges']:
                edges.append(graph['edges'])
                n_edge.append(graph['edges'].shape[:1])
            else:
                n_edge.append([0])

            offset += len(graph['nodes'])

        nodes = tf.convert_to_tensor(np.concatenate(nodes))
        globals = tf.convert_to_tensor(np.concatenate(globals))
        n_node = tf.convert_to_tensor(np.concatenate(n_node))
        n_edge = tf.convert_to_tensor(np.concatenate(n_edge))

        if senders:
            senders = tf.convert_to_tensor(np.concatenate(senders))
        else:
            senders = tf.convert_to_tensor(senders)
        if receivers:
            receivers = tf.convert_to_tensor(np.concatenate(receivers))
        else:
            receivers = tf.convert_to_tensor(receivers)
        if edges:
            edges = tf.convert_to_tensor(np.concatenate(edges))
        else:
            edges = tf.convert_to_tensor(edges)
            edges = tf.reshape(edges, (-1, 1))

        graph = GraphsTuple(
                nodes=nodes,
                edges=edges,
                globals=globals,
                senders=senders,
                receivers=receivers,
                n_node=n_node,
                n_edge=n_edge
            )

        return graph
'''
data_gen_train = MPGraphDataGenerator(file_list=root_test_files,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_procs=num_procs,
                                          calc_stats=calc_stats,
                                          is_val=False,
                                          preprocess=preprocess,
                                          already_preprocessed=already_preprocessed,
                                          output_dir=train_output_dir)
'''
data_gen_val = MPGraphDataGenerator(file_list=root_test_files,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_procs=num_procs,
                                        calc_stats=calc_stats,
                                        is_val=True,
                                        preprocess=preprocess,
                                        already_preprocessed=already_preprocessed,
                                        output_dir=val_output_dir)



#samp_graph, samp_target = next(get_batch(data_gen_train.generator()))
samp_graph, samp_target = next(get_batch(data_gen_val.generator()))
data_gen_val.kill_procs()
graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)

mae_loss = tf.keras.losses.MeanAbsoluteError()

def loss_fn(targets, predictions):
        return mae_loss(targets, predictions) 


@tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,], dtype=tf.float32)])
def val_step(graphs, targets):
        predictions = model(graphs).globals
        #loss = loss_fn(targets, predictions)

        #return loss, predictions
        return predictions 



i = 1
all_targets = []
all_outputs = []

start = time.time()

for graph_data_val, targets_val in get_batch(data_gen_val.generator()):#val_iter):
        #losses_val, output_vals = val_step(graph_data_val, targets_val)
        #print('i -s ********   ', i)
        output_vals = val_step(graph_data_val, targets_val)

        targets_val = targets_val.numpy()
        output_vals = output_vals.numpy().squeeze()

        all_targets.append(targets_val)
        all_outputs.append(output_vals)
        
all_targets = np.concatenate(all_targets)
all_outputs = np.concatenate(all_outputs)
print("HELLO AI M DONE")
       
np.savez(output_dir+'/predictions_appended', targets=all_targets, outputs=all_outputs)
