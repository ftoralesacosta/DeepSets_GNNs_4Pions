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


from IPython import get_ipython
if 'IPKernelApp' in get_ipython().config:  # Check if running in IPython kernel
    args = argparse.Namespace(config='configs/default.yaml')  # Set default value for args.config
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()



### READ MODEL FROM THE YAML FILE                                                                                                                                                          
config = yaml.safe_load(open(args.config))

#data_config = config['data']
model_config = config['model']
#train_config = config['training']

### MODEL READ
model = models.BlockModel(global_output_size=1, model_config=model_config)




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



def loss_fn(targets, predictions):
        return mae_loss(targets, predictions)


#@tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,], dtype=tf.float32)])
def val_step(graphs, targets):
        predictions = model(graphs).globals
        #loss = loss_fn(targets, predictions)                                                                                                                                               

        #return loss, predictions                                                                                                                                                           
        return predictions    
